from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .helpers import (
    NVIDIA_SMI_QUERY_FIELDS,
    fmt,
    max_or_none,
    mean_or_none,
    maybe_float,
    percentile,
)
from .models import ErrorSummary, FullSummary, GpuTelemetrySummary, Verdict, WorkloadSummary


def parse_gpu_metrics(path: Path) -> GpuTelemetrySummary:
    """Parse the nvidia-smi CSV log and return aggregated telemetry statistics."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return GpuTelemetrySummary()

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < len(NVIDIA_SMI_QUERY_FIELDS):
                continue
            rows.append({
                "timestamp": row[0].strip(),
                "index": row[1].strip(),
                "name": row[2].strip(),
                "pci_bus_id": row[3].strip(),
                "driver_version": row[4].strip(),
                "pstate": row[5].strip(),
                "pcie_gen": maybe_float(row[6]),
                "pcie_width": maybe_float(row[7]),
                "temp_c": maybe_float(row[8]),
                "fan_pct": maybe_float(row[9]),
                "gpu_util_pct": maybe_float(row[10]),
                "mem_util_pct": maybe_float(row[11]),
                "mem_total_mib": maybe_float(row[12]),
                "mem_used_mib": maybe_float(row[13]),
                "power_w": maybe_float(row[14]),
                "power_limit_w": maybe_float(row[15]),
                "graphics_mhz": maybe_float(row[16]),
                "sm_mhz": maybe_float(row[17]),
                "mem_mhz": maybe_float(row[18]),
            })

    if not rows:
        return GpuTelemetrySummary()

    sample = rows[0]
    gpu_utils = [r["gpu_util_pct"] for r in rows if r["gpu_util_pct"] is not None]
    mem_utils = [r["mem_util_pct"] for r in rows if r["mem_util_pct"] is not None]
    temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
    fans = [r["fan_pct"] for r in rows if r["fan_pct"] is not None]
    powers = [r["power_w"] for r in rows if r["power_w"] is not None]
    mem_used = [r["mem_used_mib"] for r in rows if r["mem_used_mib"] is not None]
    graphics = [r["graphics_mhz"] for r in rows if r["graphics_mhz"] is not None]
    sm = [r["sm_mhz"] for r in rows if r["sm_mhz"] is not None]
    memclk = [r["mem_mhz"] for r in rows if r["mem_mhz"] is not None]

    return GpuTelemetrySummary(
        gpu_name=sample.get("name"),
        driver_version=sample.get("driver_version"),
        pci_bus_id=sample.get("pci_bus_id"),
        power_limit_w=sample.get("power_limit_w"),
        samples=len(rows),
        avg_gpu_util_pct=mean_or_none(gpu_utils),
        p95_gpu_util_pct=percentile(gpu_utils, 95),
        max_gpu_util_pct=max_or_none(gpu_utils),
        avg_mem_util_pct=mean_or_none(mem_utils),
        avg_mem_used_gib=(mean_or_none(mem_used) / 1024.0) if mem_used else None,
        max_mem_used_gib=(max_or_none(mem_used) / 1024.0) if mem_used else None,
        avg_temp_c=mean_or_none(temps),
        max_temp_c=max_or_none(temps),
        avg_fan_pct=mean_or_none(fans),
        max_fan_pct=max_or_none(fans),
        avg_power_w=mean_or_none(powers),
        p95_power_w=percentile(powers, 95),
        max_power_w=max_or_none(powers),
        avg_graphics_mhz=mean_or_none(graphics),
        p95_graphics_mhz=percentile(graphics, 95),
        avg_sm_mhz=mean_or_none(sm),
        avg_mem_mhz=mean_or_none(memclk),
    )


def parse_dmesg_errors(before_path: Path, after_path: Path) -> ErrorSummary:
    """Diff dmesg snapshots and categorise new GPU/driver error lines."""
    # Use errors="replace" to preserve line identity for non-UTF-8 bytes
    # rather than silently dropping them (errors="ignore"), which could
    # cause false diffs between before/after sets.
    before = before_path.read_text(encoding="utf-8", errors="replace") if before_path.exists() else ""
    after = after_path.read_text(encoding="utf-8", errors="replace") if after_path.exists() else ""

    before_set = set(before.splitlines())
    new_lines = [line for line in after.splitlines() if line not in before_set]

    xid = [line for line in new_lines if "xid" in line.lower()]
    nvrm = [line for line in new_lines if "nvrm" in line.lower()]
    pcie_aer = [line for line in new_lines if "aer" in line.lower() or "pcie" in line.lower()]

    return ErrorSummary(
        xid_count=len(xid),
        nvrm_count=len(nvrm),
        pcie_aer_count=len(pcie_aer),
        new_error_lines=new_lines,
    )


def build_verdict(gpu: GpuTelemetrySummary, workload: WorkloadSummary, errors: ErrorSummary) -> Verdict:
    """Evaluate run quality and return a PASS/WARN/FAIL verdict with issue details."""
    issues: list[str] = []

    if errors.xid_count > 0:
        issues.append("Xid errors detected")
    if errors.nvrm_count > 0:
        issues.append("NVRM errors detected")
    if errors.pcie_aer_count > 0:
        issues.append("PCIe/AER messages detected")
    if (gpu.avg_gpu_util_pct or 0.0) < workload.util_floor_pct:
        issues.append(f"Average GPU utilization below {workload.util_floor_pct:.0f}%")
    # Check power relative to the GPU's own reported power limit; fall back to 350 W if unknown
    _power_limit = gpu.power_limit_w or 0.0
    _floor_ratio = workload.power_floor_ratio
    _power_floor = _power_limit * _floor_ratio if _power_limit > 0 else 350.0
    if (gpu.max_power_w or 0.0) < _power_floor:
        issues.append(f"Power draw stayed below {_floor_ratio * 100:.0f}% of rated limit")
    if (gpu.max_mem_used_gib or 0.0) < 8.0:
        issues.append("VRAM usage stayed unexpectedly low")
    if workload.actual_target_vram_gib and (gpu.max_mem_used_gib or 0.0) < workload.actual_target_vram_gib * 0.85:
        issues.append("Observed VRAM usage fell materially below requested target")
    if workload.iterations_measured == 0:
        issues.append("No measured workload iterations captured")

    if errors.xid_count > 0 or errors.nvrm_count > 0:
        status = "FAIL"
        assessment = "Driver or hardware faults were detected during the run."
    elif issues:
        status = "WARN"
        assessment = "The run completed, but one or more quality or stability checks need review."
    else:
        status = "PASS"
        assessment = "Stable sustained AI load. No driver or hardware faults detected."

    return Verdict(status=status, issues=issues, assessment=assessment)


def build_markdown_report(summary: FullSummary) -> str:
    """Render the full run summary as a markdown document."""
    gpu = summary.gpu
    wl = summary.workload
    errs = summary.errors
    verdict = summary.verdict

    lines: list[str] = []
    lines.append("# NVIDIA AI Stress Test Report")
    lines.append("")
    lines.append(f"Generated: {summary.generated_at}")
    lines.append(f"Artifacts dir: `{summary.outdir}`")
    lines.append("")
    lines.append(f"## Verdict: {verdict.status}")
    lines.append(f"- {verdict.assessment}")
    for issue in verdict.issues:
        lines.append(f"- {issue}")

    lines.append("")
    lines.append("## GPU Summary")
    lines.append(f"- GPU: {gpu.gpu_name or wl.gpu_name or 'n/a'}")
    lines.append(f"- Driver: {gpu.driver_version or 'n/a'}")
    lines.append(f"- PCI Bus ID: {gpu.pci_bus_id or 'n/a'}")
    lines.append(f"- Avg GPU Util: {fmt(gpu.avg_gpu_util_pct, 1, '%')}")
    lines.append(f"- P95 GPU Util: {fmt(gpu.p95_gpu_util_pct, 1, '%')}")
    lines.append(f"- Avg Temp: {fmt(gpu.avg_temp_c, 1, ' C')}")
    lines.append(f"- Max Temp: {fmt(gpu.max_temp_c, 1, ' C')}")
    lines.append(f"- Avg Power: {fmt(gpu.avg_power_w, 1, ' W')}")
    lines.append(f"- Max Power: {fmt(gpu.max_power_w, 1, ' W')}")
    lines.append(f"- Avg VRAM Used: {fmt(gpu.avg_mem_used_gib, 2, ' GiB')}")
    lines.append(f"- Max VRAM Used: {fmt(gpu.max_mem_used_gib, 2, ' GiB')}")

    lines.append("")
    lines.append("## Workload Summary")
    lines.append(f"- GPU Class: {wl.gpu_class}")
    lines.append(f"- Profile: {wl.profile}")
    lines.append(f"- Dtype: {wl.dtype}")
    lines.append(f"- Duration: {wl.duration_seconds}s (warmup {wl.warmup_seconds}s)")
    lines.append(f"- Batch Size: {wl.batch_size}")
    lines.append(f"- Sequence Length: {wl.seq_len}")
    lines.append(f"- Hidden Size: {wl.hidden_size}")
    lines.append(f"- Heads: {wl.heads}")
    lines.append(f"- Layers: {wl.layers}")
    lines.append(f"- FFN Multiplier: {wl.ffn_multiplier}")
    lines.append(f"- Requested Target VRAM: {fmt(wl.requested_target_vram_gib, 2, ' GiB')}")
    lines.append(f"- Actual Target VRAM: {fmt(wl.actual_target_vram_gib, 2, ' GiB')}")
    lines.append(f"- Max NVML VRAM Used: {fmt(wl.max_nvml_memory_used_gib, 2, ' GiB')}")
    lines.append(f"- Iterations (measured): {wl.iterations_measured}")
    lines.append(f"- Avg Iteration Time: {fmt(wl.avg_iter_ms, 2, ' ms')}")
    lines.append(f"- P50 / P95 Iteration Time: {fmt(wl.p50_iter_ms, 2, ' ms')} / {fmt(wl.p95_iter_ms, 2, ' ms')}")
    lines.append(f"- Avg Tokens/sec: {fmt(wl.avg_tokens_per_sec, 2)}")
    lines.append(f"- Max Reserved VRAM: {fmt(wl.max_memory_reserved_gib, 2, ' GiB')}")
    lines.append(f"- Max Allocated VRAM: {fmt(wl.max_memory_allocated_gib, 2, ' GiB')}")

    lines.append("")
    lines.append("## Error Scan")
    lines.append(f"- New Xid lines: {errs.xid_count}")
    lines.append(f"- New NVRM lines: {errs.nvrm_count}")
    lines.append(f"- New PCIe/AER lines: {errs.pcie_aer_count}")
    if errs.new_error_lines:
        lines.append("")
        lines.append("### New driver/log lines captured after workload")
        lines.append("```text")
        lines.extend(errs.new_error_lines[:50])
        lines.append("```")

    lines.append("")
    lines.append("## Files")
    for name in [
        "gpu_metrics.csv",
        "workload_metrics.jsonl",
        "summary.json",
        "report.md",
        "dmesg_before.txt",
        "dmesg_after.txt",
        "nvidia_smi_q_before.txt",
        "nvidia_smi_q_after.txt",
        "gpu_list.txt",
    ]:
        lines.append(f"- `{name}`")

    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    """Serialise payload to pretty-printed JSON and write to path."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
