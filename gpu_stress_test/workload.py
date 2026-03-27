from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .helpers import get_gpu_mem_used_gib, get_gpu_snapshot, mean_or_none, percentile
from .models import LiveState, WorkloadSummary

# Standard FFN expansion ratio used by GPT/Llama-class models
FFN_MULTIPLIER = 4


def allocate_vram_pressure(
    *,
    target_gib: float,
    dtype: Any,
    torch_module: Any,
    renderer: Any,
) -> tuple[list[Any], float]:
    """Fill GPU memory in 0.5 GiB chunks until target_gib is reached.

    Returns the list of pressure buffers and the final measured usage in GiB.
    """
    buffers: list[Any] = []
    if target_gib <= 0:
        return buffers, get_gpu_mem_used_gib()

    bytes_per_elem = 2 if dtype in (torch_module.float16, torch_module.bfloat16) else 4
    chunk_gib = 0.5
    elems_per_chunk = int((chunk_gib * (1024 ** 3)) / bytes_per_elem)

    current_gib = get_gpu_mem_used_gib()
    while current_gib < target_gib:
        buf = torch_module.empty(elems_per_chunk, device="cuda", dtype=dtype)
        buf.fill_(1)
        buffers.append(buf)
        torch_module.cuda.synchronize()
        time.sleep(0.05)  # brief pause to let NVML report updated usage
        current_gib = get_gpu_mem_used_gib()
        renderer.vram_progress(current_gib, target_gib)

    renderer.vram_done(current_gib, target_gib)
    return buffers, current_gib


def run_workload(config: Any, outdir: Path, renderer: Any) -> WorkloadSummary:
    """Execute a realistic N-layer transformer forward pass in a loop and return metrics.

    Each iteration runs config.layers sequential transformer blocks, each consisting of:
      causal self-attention (pre-norm) → output projection → FFN/MLP (pre-norm)

    This matches the compute and memory-access pattern of a GPT/Llama-class decoder model.

    Note: 'tokens_per_sec' counts attention elements (batch × seq_len) per second, not
    decoded LLM tokens; use it for relative throughput comparison between runs only.
    """
    # Reduce allocator fragmentation so large FFN intermediate tensors can be served
    # from non-contiguous segments rather than requiring one contiguous block.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "PyTorch is required. Install a CUDA-enabled build, for example:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16

    if config.hidden_size % config.heads != 0:
        raise SystemExit("--hidden-size must be divisible by --heads")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    props = torch.cuda.get_device_properties(device)
    total_vram_gib = props.total_memory / (1024 ** 3)
    n_layers = config.layers
    hidden = config.hidden_size
    heads = config.heads
    head_dim = hidden // heads
    batch = config.batch_size
    seq = config.seq_len
    ffn_dim = hidden * config.ffn_multiplier
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # --- Per-layer weight matrices (decoder-only transformer block) ---
    # Q, K, V, O projections + two-layer FFN; LayerNorm params in float32 for stability.
    w_q  = [torch.randn(hidden, hidden,   device=device, dtype=dtype) for _ in range(n_layers)]
    w_k  = [torch.randn(hidden, hidden,   device=device, dtype=dtype) for _ in range(n_layers)]
    w_v  = [torch.randn(hidden, hidden,   device=device, dtype=dtype) for _ in range(n_layers)]
    w_o  = [torch.randn(hidden, hidden,   device=device, dtype=dtype) for _ in range(n_layers)]
    w_ff1 = [torch.randn(hidden, ffn_dim, device=device, dtype=dtype) for _ in range(n_layers)]
    w_ff2 = [torch.randn(ffn_dim, hidden, device=device, dtype=dtype) for _ in range(n_layers)]
    ln1_w = [torch.ones(hidden,  device=device, dtype=torch.float32)  for _ in range(n_layers)]
    ln1_b = [torch.zeros(hidden, device=device, dtype=torch.float32)  for _ in range(n_layers)]
    ln2_w = [torch.ones(hidden,  device=device, dtype=torch.float32)  for _ in range(n_layers)]
    ln2_b = [torch.zeros(hidden, device=device, dtype=torch.float32)  for _ in range(n_layers)]

    # Input activations
    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)

    # Approximate KV-cache footprint for the full model (informational)
    approx_cache_gib = (batch * heads * seq * head_dim * n_layers * 2 * bytes_per_elem) / (1024 ** 3)

    requested_target_gib = config.target_vram_gib or (total_vram_gib * config.target_vram_ratio)
    actual_target_gib = min(requested_target_gib, total_vram_gib * 0.98)
    pressure_buffers, reached_target_gib = allocate_vram_pressure(
        target_gib=actual_target_gib,
        dtype=dtype,
        torch_module=torch,
        renderer=renderer,
    )

    if pressure_buffers:
        torch.cuda.synchronize()

    start_time = time.time()
    warmup_end = start_time + config.warmup_seconds
    end_time = start_time + config.duration_seconds
    jsonl_path = outdir / "workload_metrics.jsonl"

    iterations_total = 0
    iterations_measured = 0
    tokens_measured = 0
    iter_ms_values: list[float] = []
    tps_values: list[float] = []
    max_nvml_used_gib = reached_target_gib
    last_ui = 0.0

    def one_step() -> None:
        """Full N-layer transformer forward pass: pre-norm attention + pre-norm FFN per layer."""
        nonlocal x
        for i in range(n_layers):
            # Pre-attention LayerNorm
            x_norm = F.layer_norm(x.float(), (hidden,), ln1_w[i], ln1_b[i]).to(dtype)
            # Q, K, V projections → (batch, heads, seq, head_dim)
            q = (x_norm @ w_q[i]).view(batch, seq, heads, head_dim).transpose(1, 2)
            k = (x_norm @ w_k[i]).view(batch, seq, heads, head_dim).transpose(1, 2)
            v = (x_norm @ w_v[i]).view(batch, seq, heads, head_dim).transpose(1, 2)
            # Causal self-attention + output projection with residual
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            x = x + attn.transpose(1, 2).reshape(batch, seq, hidden) @ w_o[i]
            # Pre-FFN LayerNorm + 2-layer MLP (expand → GELU → contract) with residual
            x_norm = F.layer_norm(x.float(), (hidden,), ln2_w[i], ln2_b[i]).to(dtype)
            x = x + F.gelu(x_norm @ w_ff1[i]) @ w_ff2[i]

    with jsonl_path.open("w", encoding="utf-8") as handle:
        while time.time() < end_time:
            t0 = time.time()
            one_step()
            torch.cuda.synchronize()
            t1 = time.time()

            elapsed_step = t1 - t0
            iterations_total += 1
            nvml_used_gib = get_gpu_mem_used_gib()
            max_nvml_used_gib = max(max_nvml_used_gib, nvml_used_gib)

            record: dict[str, Any] = {
                "ts": t1,
                "iteration": iterations_total,
                "iteration_ms": round(elapsed_step * 1000.0, 3),
                "allocated_gib": round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3),
                "reserved_gib": round(torch.cuda.memory_reserved(device) / (1024 ** 3), 3),
                "nvml_used_gib": round(nvml_used_gib, 3),
                "measured": 0,
            }

            if t1 >= warmup_end:
                iterations_measured += 1
                # "tokens" = attention elements per iteration (batch × seq_len), not decoded tokens
                tokens_this_iter = batch * seq
                tokens_measured += tokens_this_iter
                iter_ms = elapsed_step * 1000.0
                tps = tokens_this_iter / elapsed_step
                iter_ms_values.append(iter_ms)
                tps_values.append(tps)
                record.update({
                    "measured": 1,
                    "tokens_this_iter": tokens_this_iter,
                    "tokens_per_sec": round(tps, 2),
                })

            handle.write(json.dumps(record) + "\n")
            handle.flush()

            if time.monotonic() - last_ui >= 1.0 or iterations_total == 1:
                live_state = LiveState(
                    phase="WARMUP" if t1 < warmup_end else "RUNNING",
                    elapsed_seconds=max(0.0, t1 - start_time),
                    duration_seconds=config.duration_seconds,
                    iterations_total=iterations_total,
                    iterations_measured=iterations_measured,
                    avg_iter_ms=mean_or_none(iter_ms_values),
                    p50_iter_ms=percentile(iter_ms_values, 50),
                    p95_iter_ms=percentile(iter_ms_values, 95),
                    avg_tokens_per_sec=mean_or_none(tps_values),
                    max_nvml_memory_used_gib=max_nvml_used_gib,
                    torch_allocated_gib=torch.cuda.memory_allocated(device) / (1024 ** 3),
                    torch_reserved_gib=torch.cuda.memory_reserved(device) / (1024 ** 3),
                    warmup_remaining_seconds=max(0, int(warmup_end - t1)),
                )
                renderer.live(live_state, get_gpu_snapshot())
                last_ui = time.monotonic()

    return WorkloadSummary(
        gpu_name=props.name,
        gpu_class=config.gpu_class,
        profile=config.profile,
        total_vram_gib=round(total_vram_gib, 2),
        dtype=str(dtype).replace("torch.", ""),
        duration_seconds=config.duration_seconds,
        warmup_seconds=config.warmup_seconds,
        batch_size=batch,
        seq_len=seq,
        hidden_size=hidden,
        heads=heads,
        layers=n_layers,
        ffn_multiplier=config.ffn_multiplier,
        approx_cache_gib=round(approx_cache_gib, 2),
        requested_target_vram_gib=round(requested_target_gib, 2),
        actual_target_vram_gib=round(actual_target_gib, 2),
        iterations_total=iterations_total,
        iterations_measured=iterations_measured,
        tokens_measured=tokens_measured,
        avg_tokens_per_sec=mean_or_none(tps_values),
        p50_tokens_per_sec=percentile(tps_values, 50),
        p95_tokens_per_sec=percentile(tps_values, 95),
        avg_iter_ms=mean_or_none(iter_ms_values),
        p50_iter_ms=percentile(iter_ms_values, 50),
        p95_iter_ms=percentile(iter_ms_values, 95),
        max_memory_allocated_gib=round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
        max_memory_reserved_gib=round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3),
        max_nvml_memory_used_gib=round(max_nvml_used_gib, 3),
        power_floor_ratio=config.power_floor_ratio,
        util_floor_pct=config.util_floor_pct,
    )
