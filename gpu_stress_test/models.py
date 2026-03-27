from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GpuTelemetrySummary:
    """Aggregated GPU telemetry parsed from the nvidia-smi CSV log."""

    gpu_name: str | None = None
    driver_version: str | None = None
    pci_bus_id: str | None = None
    power_limit_w: float | None = None
    samples: int = 0
    avg_gpu_util_pct: float | None = None
    p95_gpu_util_pct: float | None = None
    max_gpu_util_pct: float | None = None
    avg_mem_util_pct: float | None = None
    avg_mem_used_gib: float | None = None
    max_mem_used_gib: float | None = None
    avg_temp_c: float | None = None
    max_temp_c: float | None = None
    avg_fan_pct: float | None = None
    max_fan_pct: float | None = None
    avg_power_w: float | None = None
    p95_power_w: float | None = None
    max_power_w: float | None = None
    avg_graphics_mhz: float | None = None
    p95_graphics_mhz: float | None = None
    avg_sm_mhz: float | None = None
    avg_mem_mhz: float | None = None


@dataclass
class WorkloadSummary:
    """Per-run workload configuration and measured performance metrics."""

    gpu_name: str | None = None
    gpu_class: str = "unknown"
    profile: str = "unknown"
    total_vram_gib: float | None = None
    dtype: str = "unknown"
    duration_seconds: int = 0
    warmup_seconds: int = 0
    batch_size: int = 0
    seq_len: int = 0
    hidden_size: int = 0
    heads: int = 0
    layers: int = 0
    ffn_multiplier: int = 4
    approx_cache_gib: float | None = None
    requested_target_vram_gib: float | None = None
    actual_target_vram_gib: float | None = None
    iterations_total: int = 0
    iterations_measured: int = 0
    # "tokens" here = batch_size × seq_len attention elements per iteration,
    # not real decoded LLM tokens; useful for relative throughput comparison only.
    tokens_measured: int = 0
    avg_tokens_per_sec: float | None = None
    p50_tokens_per_sec: float | None = None
    p95_tokens_per_sec: float | None = None
    avg_iter_ms: float | None = None
    p50_iter_ms: float | None = None
    p95_iter_ms: float | None = None
    max_memory_allocated_gib: float | None = None
    max_memory_reserved_gib: float | None = None
    max_nvml_memory_used_gib: float | None = None
    # Verdict thresholds derived from the active profile
    power_floor_ratio: float = 0.55
    util_floor_pct: float = 85.0


@dataclass
class ErrorSummary:
    """Counts of new GPU/driver error lines appearing in dmesg after the run."""

    xid_count: int = 0
    nvrm_count: int = 0
    pcie_aer_count: int = 0
    new_error_lines: list[str] | None = None


@dataclass
class Verdict:
    """Final PASS/WARN/FAIL verdict with supporting issues and assessment text."""

    status: str
    issues: list[str]
    assessment: str


@dataclass
class FullSummary:
    """Top-level container for all run results written to summary.json."""

    generated_at: str
    outdir: str
    config: dict[str, Any]
    gpu: GpuTelemetrySummary
    workload: WorkloadSummary
    errors: ErrorSummary
    verdict: Verdict


@dataclass
class LiveState:
    """Snapshot of workload progress for the live console display."""

    phase: str
    elapsed_seconds: float
    duration_seconds: int
    iterations_total: int
    iterations_measured: int
    avg_iter_ms: float | None
    p50_iter_ms: float | None
    p95_iter_ms: float | None
    avg_tokens_per_sec: float | None
    max_nvml_memory_used_gib: float | None
    torch_allocated_gib: float | None
    torch_reserved_gib: float | None
    warmup_remaining_seconds: int
