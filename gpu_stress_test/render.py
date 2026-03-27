from __future__ import annotations

import time
from typing import Any

from .helpers import clear_terminal, fmt, fmt_duration, get_gpu_snapshot, query_nvidia_smi
from .models import FullSummary, LiveState


class ConsoleRenderer:
    """Handles all console output: startup banner, live status, and final report."""

    def __init__(self, plain: bool) -> None:
        self.plain = plain
        self._last_render_monotonic = 0.0

    def startup(self, config: Any) -> None:
        """Print the pre-flight banner with GPU info and run parameters."""
        snap = get_gpu_snapshot()
        gpu_name = snap.get("name", "unknown")
        total_gib = snap.get("mem_total_gib")
        requested_target = config.target_vram_gib or ((total_gib or 0.0) * config.target_vram_ratio)

        print("Storj AI GPU Stress Test")
        print("─" * 64)
        print(f"GPU           : {gpu_name}")
        print(f"GPU Class     : {config.gpu_class}")
        print(f"Profile       : {config.profile}")
        print(f"Driver/CUDA   : {query_nvidia_smi('driver_version')} / reported by nvidia-smi")
        print(f"Duration      : {config.duration_minutes}m 0s")
        print("Workload      : PyTorch transformer (attention + FFN, all layers)")
        print(f"Precision     : {config.dtype.upper()}")
        print(f"Layers        : {config.layers}  (batch {config.batch_size}, seq {config.seq_len}, hidden {config.hidden_size})")
        print(f"Target VRAM   : {requested_target:.2f} GiB")
        print(f"Output Dir    : {config.outdir}")
        print("")
        print("Preflight")
        print("  CUDA available      yes")
        print("  NVML available      yes")
        print("  GPU visible         yes")
        print(f"  Initial temp        {fmt(snap.get('temp_c'), 0, ' C')}")
        print(f"  Initial power       {fmt(snap.get('power_w'), 0, ' W')}")
        print(f"  Initial mem used    {fmt(snap.get('mem_used_gib'), 2, ' GiB')}")
        print("")
        print("Starting workload...")
        print("")

    def vram_progress(self, current_gib: float, target_gib: float) -> None:
        """Print a VRAM ramp progress line during pre-allocation."""
        if self.plain:
            print(f"phase=vram_ramp mem_used_gib={current_gib:.2f} target_gib={target_gib:.2f}")
        else:
            print(f"\r[VRAM] {current_gib:.2f} / {target_gib:.2f} GiB", end="", flush=True)

    def vram_done(self, current_gib: float, target_gib: float) -> None:
        """Print a completion line once VRAM pre-allocation has reached the target."""
        if self.plain:
            print(f"phase=vram_ready mem_used_gib={current_gib:.2f} target_gib={target_gib:.2f}")
        else:
            print(f"\r[VRAM] {current_gib:.2f} / {target_gib:.2f} GiB reached{' ' * 20}")

    def live(self, state: LiveState, gpu_snapshot: dict[str, Any]) -> None:
        """Render a live status update; throttled to ~1 Hz in rich mode."""
        now = time.monotonic()
        if not self.plain and now - self._last_render_monotonic < 0.8:
            return
        self._last_render_monotonic = now

        if self.plain:
            print(
                " ".join([
                    f"phase={state.phase.lower()}",
                    f"elapsed={int(state.elapsed_seconds)}",
                    f"util={fmt(gpu_snapshot.get('gpu_util_pct'), 0)}",
                    f"mem_used_gib={fmt(gpu_snapshot.get('mem_used_gib'), 2)}",
                    f"power_w={fmt(gpu_snapshot.get('power_w'), 0)}",
                    f"temp_c={fmt(gpu_snapshot.get('temp_c'), 0)}",
                    f"tok_s={fmt(state.avg_tokens_per_sec, 0)}",
                ])
            )
            return

        clear_terminal()
        progress = min(1.0, state.elapsed_seconds / max(1, state.duration_seconds))
        bar_width = 34
        filled = int(progress * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        header = f"AI GPU Stress Test  [{state.phase}]"
        elapsed = f"Elapsed {fmt_duration(state.elapsed_seconds)} / {fmt_duration(state.duration_seconds)}"
        print(f"{header:<54}{elapsed}")
        print("")
        print(f"Progress  [{bar}]  {progress * 100:5.1f}%")
        if state.phase == "WARMUP":
            print(f"Warmup remaining: {state.warmup_remaining_seconds}s")
        print("")
        print("GPU      Util   Mem Util   VRAM Used      Power        Temp   SM Clock   Mem Clock")
        print(
            f"{str(gpu_snapshot.get('name', 'GPU'))[:10]:<10}"
            f"{fmt(gpu_snapshot.get('gpu_util_pct'), 0, '%'):>6}   "
            f"{fmt(gpu_snapshot.get('mem_util_pct'), 0, '%'):>7}   "
            f"{fmt(gpu_snapshot.get('mem_used_gib'), 1):>5} / {fmt(gpu_snapshot.get('mem_total_gib'), 1)}G   "
            f"{fmt(gpu_snapshot.get('power_w'), 0):>4} / {fmt(gpu_snapshot.get('power_limit_w'), 0)}W   "
            f"{fmt(gpu_snapshot.get('temp_c'), 0, 'C'):>4}   "
            f"{fmt(gpu_snapshot.get('graphics_mhz'), 0, ' MHz'):>8}   "
            f"{fmt(gpu_snapshot.get('mem_mhz'), 0, ' MHz'):>9}"
        )
        print("")
        print("Workload")
        print(f"Iterations       {state.iterations_total}")
        print(f"Measured iters   {state.iterations_measured}")
        print(f"Avg iter         {fmt(state.avg_iter_ms, 1, ' ms')}")
        print(f"P50 / P95        {fmt(state.p50_iter_ms, 1, ' ms')} / {fmt(state.p95_iter_ms, 1, ' ms')}")
        print(f"Tokens/sec       {fmt(state.avg_tokens_per_sec, 0)}")
        print("")
        print("Memory")
        print(f"NVML used max    {fmt(state.max_nvml_memory_used_gib, 2, ' GiB')}")
        print(f"Torch allocated  {fmt(state.torch_allocated_gib, 2, ' GiB')}")
        print(f"Torch reserved   {fmt(state.torch_reserved_gib, 2, ' GiB')}")
        print("")
        print("Health")
        print(f"PCIe link        Gen{fmt(gpu_snapshot.get('pcie_gen'), 0)} x{fmt(gpu_snapshot.get('pcie_width'), 0)}")
        print(f"GPU state        {gpu_snapshot.get('pstate', 'n/a')}")
        print("")

    def final(self, summary: FullSummary) -> None:
        """Print the final results table after the run completes."""
        if not self.plain:
            clear_terminal()
        gpu = summary.gpu
        wl = summary.workload
        verdict = summary.verdict
        print(f"AI GPU Stress Test  [{verdict.status}]")
        print("─" * 72)
        print(f"Result          {verdict.status}")
        print(f"Duration        {fmt_duration(wl.duration_seconds)}")
        print(f"GPU             {gpu.gpu_name or wl.gpu_name or 'n/a'}")
        print(f"GPU Class       {wl.gpu_class}")
        print(f"Profile         {wl.profile}")
        print(f"Driver          {gpu.driver_version or 'n/a'}")
        print(f"Precision       {wl.dtype.upper()}")
        print(f"Workload        {wl.layers}-layer transformer (FFN×{wl.ffn_multiplier})")
        print("")
        print("Performance")
        print(f"  Avg tokens/sec        {fmt(wl.avg_tokens_per_sec, 0)}")
        print(f"  Peak tokens/sec       {fmt(wl.p95_tokens_per_sec, 0)}")
        print(f"  Avg iter time         {fmt(wl.avg_iter_ms, 1, ' ms')}")
        print(f"  P50 / P95 iter        {fmt(wl.p50_iter_ms, 1, ' ms')} / {fmt(wl.p95_iter_ms, 1, ' ms')}")
        print(f"  Iterations measured   {wl.iterations_measured}")
        print("")
        print("Utilization")
        print(f"  GPU util avg          {fmt(gpu.avg_gpu_util_pct, 1, '%')}")
        print(f"  GPU util min/max      0% / {fmt(gpu.max_gpu_util_pct, 0, '%')}")
        print(f"  Memory util avg       {fmt(gpu.avg_mem_util_pct, 1, '%')}")
        print(f"  Max VRAM used         {fmt(gpu.max_mem_used_gib, 1)} / {fmt(wl.total_vram_gib, 1)} GiB")
        print("")
        print("Thermals and Power")
        print(f"  Temp avg/max          {fmt(gpu.avg_temp_c, 0, ' C')} / {fmt(gpu.max_temp_c, 0, ' C')}")
        print(f"  Power avg/max         {fmt(gpu.avg_power_w, 0, ' W')} / {fmt(gpu.max_power_w, 0, ' W')}")
        print(f"  SM clock avg          {fmt(gpu.avg_sm_mhz, 0, ' MHz')}")
        print(f"  Mem clock avg         {fmt(gpu.avg_mem_mhz, 0, ' MHz')}")
        print("")
        print("Health")
        print(f"  Xid errors            {summary.errors.xid_count}")
        print(f"  NVRM errors           {summary.errors.nvrm_count}")
        print(f"  PCIe/AER messages     {summary.errors.pcie_aer_count}")
        # Single snapshot call; reuse for both PCIe fields
        snap = get_gpu_snapshot()
        print(f"  PCIe link             Gen{fmt(snap.get('pcie_gen'), 0)} x{fmt(snap.get('pcie_width'), 0)}")
        print("")
        print(f"Assessment: {verdict.assessment}")
        print("")
