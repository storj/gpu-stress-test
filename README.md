# GPU-STRESS-TEST(1)

## NAME
gpu-stress-test — NVIDIA GPU stress harness for AI workloads

## SYNOPSIS
```
uv run python main.py [OPTIONS]
```

## DESCRIPTION
Runs a synthetic but realistic decoder-only transformer forward pass against an NVIDIA GPU,
monitoring telemetry and producing a PASS/WARN/FAIL verdict on completion.

Each iteration executes a full N-layer transformer stack:
causal self-attention (pre-norm) → output projection → FFN/MLP (pre-norm).
This matches the compute and memory-access pattern of GPT/Llama-class decoder models.

The GPU class is auto-detected from the nvidia-smi device name and a profile is selected
automatically. Use `--profile` to override.

Outputs a timestamped directory containing `summary.json`, `report.md`, per-iteration
JSONL metrics, nvidia-smi CSV telemetry, and before/after dmesg snapshots.

## INSTALL
Requires [uv](https://docs.astral.sh/uv/) and `nvidia-smi` on PATH.

```bash
uv sync
```

This creates `.venv/` and installs the CUDA-enabled PyTorch build automatically.
For systems running CUDA < 12.8, change `torch-backend = "cu128"` in `pyproject.toml`
to the appropriate tag (e.g. `"cu121"`) before running `uv sync`.

## PROFILES
Profiles set the workload shape and verdict thresholds. Individual flags override profile values.

| Profile | Target GPU | Layers | Batch | Seq | VRAM target |
|---------|-----------|--------|-------|-----|-------------|
| `consumer` | RTX 3090 / 4090 / 5090 (24–32 GiB) | 32 | 16 | 4096 | 75% |
| `datacenter` | A100 / H100 (40–80 GiB) | 40 | 32 | 4096 | 80% |
| `datacenter-max` | H200 / B100 / B200 (80–192 GiB) | 80 | 32 | 8192 | 82% |

`auto` (default) selects `consumer` or `datacenter` based on the detected GPU name.
`datacenter-max` must be requested explicitly.

## OPTIONS
```
--profile <name>          Workload preset: auto|consumer|datacenter|datacenter-max
                          (default: auto)
--gpu-class <class>       Override GPU class for auto profile selection:
                          auto|consumer|datacenter  (default: auto)
--duration <int>          Total runtime in minutes (default: 20)
--warmup <int>            Warmup seconds excluded from metrics (default: 30)
--dtype <bf16|fp16>       Tensor precision; overrides profile (default: profile)
--target-vram-ratio <f>   Fraction of total VRAM to fill; overrides profile
--target-vram-gb <f>      Absolute VRAM target in GiB; overrides ratio
--batch-size <int>        Batch size; overrides profile
--seq-len <int>           Sequence length; overrides profile
--hidden-size <int>       Model hidden dimension; overrides profile
--heads <int>             Attention heads; overrides profile
--layers <int>            Transformer layers (real stacked compute); overrides profile
--ffn-multiplier <int>    FFN hidden-dim expansion factor (default: 4)
--log-interval <int>      nvidia-smi polling interval in seconds (default: 1)
--outdir <path>           Output directory (default: ./gpu_stress_YYYYMMDD_HHMMSS)
--plain                   Line-oriented output; disables live dashboard
```

## EXAMPLES
```bash
# Standard 20-minute run — profile auto-selected from GPU name
uv run python main.py

# Explicit profile for H100
uv run python main.py --profile datacenter

# Maximum saturation for H200 / B200
uv run python main.py --profile datacenter-max --duration 30

# Quick 5-minute smoke test on RTX 5090
uv run python main.py --duration 5 --warmup 10

# Push VRAM harder on consumer GPU
uv run python main.py --target-vram-ratio 0.88

# Log-friendly output for CI
uv run python main.py --plain --outdir ./results
```

## OUTPUT FILES
| File | Contents |
|------|----------|
| `summary.json` | All metrics and verdict in machine-readable form |
| `report.md` | Human-readable run report |
| `workload_metrics.jsonl` | Per-iteration timing and memory (one JSON object per line) |
| `gpu_metrics.csv` | nvidia-smi telemetry sampled every `--log-interval` seconds |
| `gpu_list.txt` | `nvidia-smi -L` output |
| `nvidia_smi_q_before.txt` | Full `nvidia-smi -q` before run |
| `nvidia_smi_q_after.txt` | Full `nvidia-smi -q` after run |
| `dmesg_before.txt` | Kernel log (NVRM/Xid/PCIe/AER) before run |
| `dmesg_after.txt` | Kernel log after run |

## VERDICT
**PASS** — No driver/hardware faults; utilisation and power within profile thresholds.
**WARN** — Run completed but a quality check failed (low utilisation, power, or VRAM).
**FAIL** — Xid or NVRM errors detected in dmesg.

Power and utilisation thresholds are profile-specific. `dmesg` access may be restricted
on production Linux systems (`kernel.dmesg_restrict=1`); the error scan will be empty
but the workload and telemetry results remain valid.

## NOTES
`tokens/sec` counts attention elements (batch × seq\_len) per second, not decoded LLM tokens.
It is meaningful only for relative comparison between runs with identical configurations.

Layer norm weights are kept in float32 for numerical stability (standard practice).
Only `cuda:0` is tested; for multi-GPU nodes specify `CUDA_VISIBLE_DEVICES` before running.
