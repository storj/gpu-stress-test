#!/usr/bin/env python3
"""AI-oriented NVIDIA GPU stress harness with integrated terminal UI and reporting.

Supports both recent consumer GPUs and datacenter GPUs such as H100/H200/B100/B200.
The workload shape can be auto-tuned from simple profile definitions so future GPUs with
larger memory footprints are easy to support by extending PROFILE_PRESETS.

Examples:
    python3 gpu-stress-test.py --duration 20
    python3 gpu-stress-test.py --duration 20 --gpu-class datacenter --profile datacenter
    python3 gpu-stress-test.py --duration 30 --target-vram-ratio 0.82 --plain
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


QUERY_FIELDS = [
    "timestamp",
    "index",
    "name",
    "pci.bus_id",
    "driver_version",
    "pstate",
    "pcie.link.gen.current",
    "pcie.link.width.current",
    "temperature.gpu",
    "fan.speed",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "power.draw",
    "power.limit",
    "clocks.current.graphics",
    "clocks.current.sm",
    "clocks.current.memory",
]


@dataclass(frozen=True)
class ProfilePreset:
    name: str
    workload_name: str
    gpu_class: str
    dtype: str
    batch_size: int
    seq_len: int
    hidden_size: int
    heads: int
    layers: int
    target_vram_ratio: float
    min_target_vram_gb: float
    max_target_vram_gb: Optional[float]
    power_floor_ratio: float
    vram_floor_ratio: float
    util_floor_pct: float


PROFILE_PRESETS: dict[str, ProfilePreset] = {
    "consumer": ProfilePreset(
        name="consumer",
        workload_name="PyTorch attention+matmul",
        gpu_class="consumer",
        dtype="bf16",
        batch_size=16,
        seq_len=4096,
        hidden_size=4096,
        heads=32,
        layers=24,
        target_vram_ratio=0.74,
        min_target_vram_gb=8.0,
        max_target_vram_gb=28.0,
        power_floor_ratio=0.55,
        vram_floor_ratio=0.35,
        util_floor_pct=85.0,
    ),
    "datacenter": ProfilePreset(
        name="datacenter",
        workload_name="PyTorch attention+matmul",
        gpu_class="datacenter",
        dtype="bf16",
        batch_size=24,
        seq_len=8192,
        hidden_size=8192,
        heads=64,
        layers=40,
        target_vram_ratio=0.80,
        min_target_vram_gb=24.0,
        max_target_vram_gb=96.0,
        power_floor_ratio=0.45,
        vram_floor_ratio=0.50,
        util_floor_pct=85.0,
    ),
    "datacenter-max": ProfilePreset(
        name="datacenter-max",
        workload_name="PyTorch attention+matmul",
        gpu_class="datacenter",
        dtype="bf16",
        batch_size=32,
        seq_len=8192,
        hidden_size=8192,
        heads=64,
        layers=48,
        target_vram_ratio=0.84,
        min_target_vram_gb=32.0,
        max_target_vram_gb=120.0,
        power_floor_ratio=0.45,
        vram_floor_ratio=0.55,
        util_floor_pct=88.0,
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return statistics.mean(vals) if vals else None


def max_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return max(vals) if vals else None


def min_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return min(vals) if vals else None


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    vals = sorted(values)
    if not vals:
        return None
    idx = max(0, min(len(vals) - 1, math.ceil((p / 100.0) * len(vals)) - 1))
    return vals[idx]


def fmt_num(value: Optional[float], digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def run_capture(command: list[str], output_path: Path, allow_failure: bool = True) -> int:
    with output_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}")
    return proc.returncode


def color(text: str, code: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def detect_gpu_class(gpu_name: str) -> str:
    upper = gpu_name.upper()
    datacenter_markers = (
        "H100",
        "H200",
        "B100",
        "B200",
        "A100",
        "A800",
        "H800",
        "L40",
        "L4",
        "TESLA",
        "SXM",
        "HGX",
        "PCIe H100",
        "RTX PRO 6000 BLACKWELL SERVER",
    )
    return "datacenter" if any(marker in upper for marker in datacenter_markers) else "consumer"


@dataclass
class Config:
    duration_minutes: float
    warmup_seconds: int
    dtype: str
    target_vram_gb: float
    target_vram_ratio: float
    batch_size: int
    seq_len: int
    hidden_size: int
    heads: int
    layers: int
    log_interval: float
    outdir: Path
    plain: bool
    gpu_class: str
    profile: str
    workload_name: str
    power_floor_ratio: float
    vram_floor_ratio: float
    util_floor_pct: float

    @property
    def duration_seconds(self) -> int:
        return max(1, int(round(self.duration_minutes * 60)))


@dataclass
class GpuSample:
    timestamp: str = ""
    index: str = "0"
    name: str = ""
    pci_bus_id: str = ""
    driver_version: str = ""
    pstate: str = ""
    pcie_gen: Optional[float] = None
    pcie_width: Optional[float] = None
    temp_c: Optional[float] = None
    fan_pct: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    mem_util_pct: Optional[float] = None
    mem_total_mib: Optional[float] = None
    mem_used_mib: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    graphics_mhz: Optional[float] = None
    sm_mhz: Optional[float] = None
    mem_mhz: Optional[float] = None


@dataclass
class GpuTelemetrySummary:
    gpu_name: Optional[str] = None
    driver_version: Optional[str] = None
    pci_bus_id: Optional[str] = None
    power_limit_w: Optional[float] = None
    samples: int = 0
    avg_gpu_util_pct: Optional[float] = None
    min_gpu_util_pct: Optional[float] = None
    p95_gpu_util_pct: Optional[float] = None
    max_gpu_util_pct: Optional[float] = None
    avg_mem_util_pct: Optional[float] = None
    avg_mem_used_gib: Optional[float] = None
    max_mem_used_gib: Optional[float] = None
    avg_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    avg_fan_pct: Optional[float] = None
    max_fan_pct: Optional[float] = None
    avg_power_w: Optional[float] = None
    p95_power_w: Optional[float] = None
    max_power_w: Optional[float] = None
    avg_graphics_mhz: Optional[float] = None
    p95_graphics_mhz: Optional[float] = None
    avg_sm_mhz: Optional[float] = None
    avg_mem_mhz: Optional[float] = None
    pcie_gen: Optional[float] = None
    pcie_width: Optional[float] = None


@dataclass
class WorkloadSummary:
    gpu_name: Optional[str] = None
    gpu_class: str = "unknown"
    profile: str = "unknown"
    workload_name: str = "unknown"
    total_vram_gb: Optional[float] = None
    target_vram_gb: Optional[float] = None
    target_vram_ratio: Optional[float] = None
    dtype: str = "unknown"
    duration_seconds: int = 0
    warmup_seconds: int = 0
    batch_size: int = 0
    seq_len: int = 0
    hidden_size: int = 0
    heads: int = 0
    layers_hint: int = 0
    approx_cache_gb: Optional[float] = None
    iterations_total: int = 0
    iterations_measured: int = 0
    tokens_measured: int = 0
    avg_tokens_per_sec: Optional[float] = None
    peak_tokens_per_sec: Optional[float] = None
    p50_tokens_per_sec: Optional[float] = None
    p95_tokens_per_sec: Optional[float] = None
    avg_iter_ms: Optional[float] = None
    p50_iter_ms: Optional[float] = None
    p95_iter_ms: Optional[float] = None
    max_memory_allocated_gb: Optional[float] = None
    max_memory_reserved_gb: Optional[float] = None


@dataclass
class ErrorSummary:
    xid_count: int = 0
    nvrm_count: int = 0
    pcie_aer_count: int = 0
    new_error_lines: list[str] = field(default_factory=list)

    @property
    def has_driver_errors(self) -> bool:
        return bool(self.xid_count or self.nvrm_count or self.pcie_aer_count)


@dataclass
class Verdict:
    status: str
    issues: list[str]
    assessment: str


@dataclass
class FullSummary:
    generated_at: str
    outdir: str
    gpu: GpuTelemetrySummary
    workload: WorkloadSummary
    errors: ErrorSummary
    verdict: Verdict


@dataclass
class RuntimeState:
    phase: str = "STARTING"
    latest_sample: Optional[GpuSample] = None
    iteration_count: int = 0
    measured_iterations: int = 0
    avg_iter_ms: Optional[float] = None
    p50_iter_ms: Optional[float] = None
    p95_iter_ms: Optional[float] = None
    avg_tokens_per_sec: Optional[float] = None
    peak_tokens_per_sec: Optional[float] = None
    avg_gpu_util_pct: Optional[float] = None
    avg_power_w: Optional[float] = None
    max_power_w: Optional[float] = None
    avg_temp_c: Optional[float] = None
    max_temp_c: Optional[float] = None
    max_mem_used_gib: Optional[float] = None
    xid_count: int = 0
    nvrm_count: int = 0
    event_lines: list[str] = field(default_factory=list)
    failed: bool = False
    failure_reason: Optional[str] = None


class TelemetryCollector:
    def __init__(self, outdir: Path, interval_seconds: float) -> None:
        self.outdir = outdir
        self.interval_seconds = max(0.5, interval_seconds)
        self.csv_path = outdir / "gpu_metrics.csv"
        self._latest: Optional[GpuSample] = None
        self._samples: list[GpuSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def samples(self) -> list[GpuSample]:
        with self._lock:
            return list(self._samples)

    @property
    def latest(self) -> Optional[GpuSample]:
        with self._lock:
            return self._latest

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "timestamp",
                "index",
                "name",
                "pci_bus_id",
                "driver_version",
                "pstate",
                "pcie_gen",
                "pcie_width",
                "temp_c",
                "fan_pct",
                "gpu_util_pct",
                "mem_util_pct",
                "mem_total_mib",
                "mem_used_mib",
                "power_w",
                "power_limit_w",
                "graphics_mhz",
                "sm_mhz",
                "mem_mhz",
            ])
            while not self._stop.is_set():
                row = self._query_once()
                if row is not None:
                    writer.writerow([
                        row.timestamp,
                        row.index,
                        row.name,
                        row.pci_bus_id,
                        row.driver_version,
                        row.pstate,
                        row.pcie_gen,
                        row.pcie_width,
                        row.temp_c,
                        row.fan_pct,
                        row.gpu_util_pct,
                        row.mem_util_pct,
                        row.mem_total_mib,
                        row.mem_used_mib,
                        row.power_w,
                        row.power_limit_w,
                        row.graphics_mhz,
                        row.sm_mhz,
                        row.mem_mhz,
                    ])
                    handle.flush()
                    with self._lock:
                        self._latest = row
                        self._samples.append(row)
                self._stop.wait(self.interval_seconds)

    def _query_once(self) -> Optional[GpuSample]:
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        line = proc.stdout.strip().splitlines()[:1]
        if not line:
            return None
        parts = [part.strip() for part in line[0].split(",")]
        if len(parts) != len(QUERY_FIELDS):
            return None
        return GpuSample(
            timestamp=parts[0],
            index=parts[1],
            name=parts[2],
            pci_bus_id=parts[3],
            driver_version=parts[4],
            pstate=parts[5],
            pcie_gen=maybe_float(parts[6]),
            pcie_width=maybe_float(parts[7]),
            temp_c=maybe_float(parts[8]),
            fan_pct=maybe_float(parts[9]),
            gpu_util_pct=maybe_float(parts[10]),
            mem_util_pct=maybe_float(parts[11]),
            mem_total_mib=maybe_float(parts[12]),
            mem_used_mib=maybe_float(parts[13]),
            power_w=maybe_float(parts[14]),
            power_limit_w=maybe_float(parts[15]),
            graphics_mhz=maybe_float(parts[16]),
            sm_mhz=maybe_float(parts[17]),
            mem_mhz=maybe_float(parts[18]),
        )


class ConsoleUI:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.is_tty = sys.stdout.isatty() and not config.plain
        self.use_color = self.is_tty and os.environ.get("TERM", "dumb") != "dumb"

    def startup(self, sample: Optional[GpuSample]) -> None:
        initial_mem_gib = (sample.mem_used_mib / 1024.0) if sample and sample.mem_used_mib is not None else None
        lines = [
            color("AI GPU Stress Test", "1;36", self.use_color),
            "─" * 72,
            f"GPU           : {sample.name if sample else 'n/a'}",
            f"GPU Class     : {self.config.gpu_class}",
            f"Profile       : {self.config.profile}",
            f"Driver/CUDA   : {sample.driver_version if sample else 'n/a'} / detected via nvidia-smi",
            f"Device        : {sample.pci_bus_id if sample else 'n/a'}",
            f"Duration      : {self.config.duration_minutes:g} min",
            f"Workload      : {self.config.workload_name}",
            f"Precision     : {self.config.dtype.upper()}",
            f"Target VRAM   : {self.config.target_vram_gb:.1f} GiB ({self.config.target_vram_ratio * 100:.0f}% target)",
            f"Output Dir    : {self.config.outdir}",
            "",
            "Preflight",
            f"  GPU visible         {'yes' if sample else 'no'}",
            f"  Initial temp        {fmt_num(sample.temp_c if sample else None,0)} C",
            f"  Initial power       {fmt_num(sample.power_w if sample else None,0)} W",
            f"  Initial mem used    {fmt_num(initial_mem_gib,1)} GiB",
            "",
            color("Starting workload...", "36", self.use_color),
        ]
        print("\n".join(lines), flush=True)

    def live(self, state: RuntimeState, elapsed_s: float, total_s: float, warmup_s: int) -> None:
        sample = state.latest_sample
        status = "FAILED" if state.failed else state.phase
        util = fmt_num(sample.gpu_util_pct if sample else None, 0)
        mem_util = fmt_num(sample.mem_util_pct if sample else None, 0)
        mem_used_gib = (sample.mem_used_mib / 1024.0) if sample and sample.mem_used_mib is not None else None
        mem_total_gib = (sample.mem_total_mib / 1024.0) if sample and sample.mem_total_mib is not None else None
        progress = min(1.0, elapsed_s / total_s) if total_s > 0 else 0.0
        bar_width = 34
        filled = int(round(progress * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)
        gpu_label = (sample.name if sample and sample.name else "GPU")[:18]

        lines = [
            f"Storj: AI GPU Stress Test  [{status}]".ljust(54) + f"Elapsed {fmt_duration(elapsed_s)} / {fmt_duration(total_s)}",
            "",
            f"Progress  [{bar}]  {progress * 100:5.1f}%",
            "",
            "GPU                Util   Mem Util   VRAM Used        Power         Temp   SM Clock   Mem Clock",
            (
                f"{gpu_label:<18} {util:>3}%   {mem_util:>3}%        "
                f"{fmt_num(mem_used_gib,1):>5} / {fmt_num(mem_total_gib,1)}G   "
                f"{fmt_num(sample.power_w if sample else None,0):>4} / {fmt_num(sample.power_limit_w if sample else None,0)}W   "
                f"{fmt_num(sample.temp_c if sample else None,0):>3}C   "
                f"{fmt_num(sample.sm_mhz if sample else None,0):>4} MHz   "
                f"{fmt_num(sample.mem_mhz if sample else None,0):>4} MHz"
            ),
            "",
            "Workload",
            f"Profile          {self.config.profile}",
            f"GPU Class        {self.config.gpu_class}",
            f"Precision        {self.config.dtype.upper()}",
            f"Batch/Seq        {self.config.batch_size} / {self.config.seq_len}",
            f"Hidden/Heads     {self.config.hidden_size} / {self.config.heads}",
            f"Iterations       {state.iteration_count}",
            f"Avg iter         {fmt_num(state.avg_iter_ms,1)} ms",
            f"P50 / P95        {fmt_num(state.p50_iter_ms,1)} / {fmt_num(state.p95_iter_ms,1)} ms",
            f"Tokens/sec       {fmt_num(state.avg_tokens_per_sec,0)}",
            "",
            "Health",
            f"Driver errors    Xid={state.xid_count}  NVRM={state.nvrm_count}",
            f"PCIe link        Gen{fmt_num(sample.pcie_gen if sample else None,0)} x{fmt_num(sample.pcie_width if sample else None,0)}",
            f"GPU state        {sample.pstate if sample and sample.pstate else 'n/a'}",
            "",
            "Trends",
            f"Power avg/max    {fmt_num(state.avg_power_w,0)} / {fmt_num(state.max_power_w,0)} W",
            f"Temp avg/max     {fmt_num(state.avg_temp_c,0)} / {fmt_num(state.max_temp_c,0)} C",
            f"GPU util avg     {fmt_num(state.avg_gpu_util_pct,1)}%",
            f"VRAM max         {fmt_num(state.max_mem_used_gib,1)} GiB",
        ]
        if elapsed_s < warmup_s:
            lines.extend(["", f"Warmup remaining {fmt_duration(warmup_s - elapsed_s)}"])
        if state.event_lines:
            lines.extend(["", "Events"])
            for event in state.event_lines[-3:]:
                lines.append(f"  {event}")

        payload = "\n".join(lines)
        if self.is_tty:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(payload)
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            last_event = state.event_lines[-1] if state.event_lines else ""
            event_repr = json.dumps(last_event) if last_event else '""'
            print(
                " ".join(
                    [
                        f"phase={status}",
                        f"elapsed={int(elapsed_s)}s/{int(total_s)}s",
                        f"gpu_util={fmt_num(sample.gpu_util_pct if sample else None,0)}",
                        f"power_w={fmt_num(sample.power_w if sample else None,0)}",
                        f"temp_c={fmt_num(sample.temp_c if sample else None,0)}",
                        f"tok_s={fmt_num(state.avg_tokens_per_sec,0)}",
                        f"iter={state.iteration_count}",
                        f"event={event_repr}",
                    ]
                ),
                flush=True,
            )

    def final(self, summary: FullSummary) -> None:
        verdict_color = {"PASS": "1;32", "CHECK": "1;33", "FAIL": "1;31"}.get(summary.verdict.status, "1")
        gpu = summary.gpu
        wl = summary.workload
        errs = summary.errors
        lines = [
            color(f"Storj: AI GPU Stress Test  [{summary.verdict.status}]", verdict_color, self.use_color),
            "─" * 72,
            f"Result          {summary.verdict.status}",
            f"Duration        {fmt_duration(wl.duration_seconds)}",
            f"GPU             {gpu.gpu_name or wl.gpu_name or 'n/a'}",
            f"GPU Class       {wl.gpu_class}",
            f"Profile         {wl.profile}",
            f"Driver          {gpu.driver_version or 'n/a'}",
            f"Precision       {wl.dtype.upper()}",
            f"Workload        {wl.workload_name}",
            "",
            "Performance",
            f"  Avg tokens/sec        {fmt_num(wl.avg_tokens_per_sec,0)}",
            f"  Peak tokens/sec       {fmt_num(wl.peak_tokens_per_sec,0)}",
            f"  Avg iter time         {fmt_num(wl.avg_iter_ms,1)} ms",
            f"  P50 / P95 iter        {fmt_num(wl.p50_iter_ms,1)} / {fmt_num(wl.p95_iter_ms,1)} ms",
            f"  Iterations measured   {wl.iterations_measured}",
            "",
            "Utilization",
            f"  GPU util avg          {fmt_num(gpu.avg_gpu_util_pct,1)}%",
            f"  GPU util min/max      {fmt_num(gpu.min_gpu_util_pct,0)}% / {fmt_num(gpu.max_gpu_util_pct,0)}%",
            f"  Memory util avg       {fmt_num(gpu.avg_mem_util_pct,1)}%",
            f"  Max VRAM used         {fmt_num(gpu.max_mem_used_gib,1)} / {fmt_num(wl.total_vram_gb,1)} GiB",
            f"  Target VRAM           {fmt_num(wl.target_vram_gb,1)} GiB ({fmt_num((wl.target_vram_ratio or 0) * 100,0)}%)",
            "",
            "Thermals and Power",
            f"  Temp avg/max          {fmt_num(gpu.avg_temp_c,0)} C / {fmt_num(gpu.max_temp_c,0)} C",
            f"  Power avg/max         {fmt_num(gpu.avg_power_w,0)} W / {fmt_num(gpu.max_power_w,0)} W",
            f"  SM clock avg          {fmt_num(gpu.avg_sm_mhz,0)} MHz",
            f"  Mem clock avg         {fmt_num(gpu.avg_mem_mhz,0)} MHz",
            "",
            "Health",
            f"  Xid errors            {errs.xid_count}",
            f"  NVRM errors           {errs.nvrm_count}",
            f"  PCIe/AER messages     {errs.pcie_aer_count}",
            f"  PCIe link             Gen{fmt_num(gpu.pcie_gen,0)} x{fmt_num(gpu.pcie_width,0)}",
            "",
            f"Assessment: {summary.verdict.assessment}",
            "",
            "Artifacts",
            f"  Summary JSON          {Path(summary.outdir) / 'summary.json'}",
            f"  Markdown report       {Path(summary.outdir) / 'report.md'}",
            f"  GPU CSV               {Path(summary.outdir) / 'gpu_metrics.csv'}",
            f"  Workload JSONL        {Path(summary.outdir) / 'workload_metrics.jsonl'}",
        ]
        print("\n".join(lines), flush=True)


def parse_gpu_metrics_from_samples(samples: list[GpuSample]) -> GpuTelemetrySummary:
    if not samples:
        return GpuTelemetrySummary()
    sample = samples[-1]
    gpu_utils = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
    mem_utils = [s.mem_util_pct for s in samples if s.mem_util_pct is not None]
    temps = [s.temp_c for s in samples if s.temp_c is not None]
    fans = [s.fan_pct for s in samples if s.fan_pct is not None]
    powers = [s.power_w for s in samples if s.power_w is not None]
    mem_used = [s.mem_used_mib for s in samples if s.mem_used_mib is not None]
    graphics = [s.graphics_mhz for s in samples if s.graphics_mhz is not None]
    sm = [s.sm_mhz for s in samples if s.sm_mhz is not None]
    memclk = [s.mem_mhz for s in samples if s.mem_mhz is not None]
    return GpuTelemetrySummary(
        gpu_name=sample.name,
        driver_version=sample.driver_version,
        pci_bus_id=sample.pci_bus_id,
        power_limit_w=sample.power_limit_w,
        samples=len(samples),
        avg_gpu_util_pct=mean_or_none(gpu_utils),
        min_gpu_util_pct=min_or_none(gpu_utils),
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
        pcie_gen=sample.pcie_gen,
        pcie_width=sample.pcie_width,
    )


def parse_dmesg_errors(before_path: Path, after_path: Path) -> ErrorSummary:
    before = before_path.read_text(encoding="utf-8", errors="ignore") if before_path.exists() else ""
    after = after_path.read_text(encoding="utf-8", errors="ignore") if after_path.exists() else ""
    before_set = set(before.splitlines())
    new_lines = [line for line in after.splitlines() if line not in before_set]
    xid = [line for line in new_lines if "xid" in line.lower()]
    nvrm = [line for line in new_lines if "nvrm" in line.lower()]
    pcie_aer = [line for line in new_lines if "aer" in line.lower() or "pcie" in line.lower()]
    return ErrorSummary(xid_count=len(xid), nvrm_count=len(nvrm), pcie_aer_count=len(pcie_aer), new_error_lines=new_lines)


def build_verdict(config: Config, gpu: GpuTelemetrySummary, workload: WorkloadSummary, errors: ErrorSummary) -> Verdict:
    issues: list[str] = []
    if errors.xid_count > 0:
        issues.append("Xid errors detected")
    if errors.nvrm_count > 0:
        issues.append("NVRM errors detected")
    if errors.pcie_aer_count > 0:
        issues.append("PCIe/AER messages detected")
    if workload.iterations_measured == 0:
        issues.append("No measured workload iterations captured")
    if (gpu.avg_gpu_util_pct or 0.0) < config.util_floor_pct:
        issues.append(f"Average GPU utilization below {config.util_floor_pct:.0f}%")

    power_limit = gpu.power_limit_w or 0.0
    if power_limit > 0 and (gpu.max_power_w or 0.0) < power_limit * config.power_floor_ratio:
        issues.append(
            f"Power draw stayed below {config.power_floor_ratio * 100:.0f}% of power limit"
        )

    total_vram_gb = workload.total_vram_gb or 0.0
    if total_vram_gb > 0 and (gpu.max_mem_used_gib or 0.0) < total_vram_gb * config.vram_floor_ratio:
        issues.append(
            f"VRAM usage stayed below {config.vram_floor_ratio * 100:.0f}% of total memory"
        )

    if errors.xid_count > 0 or errors.nvrm_count > 0:
        return Verdict("FAIL", issues, "Driver or hardware faults were detected during the run.")
    if issues:
        return Verdict("CHECK", issues, "The test completed, but the workload did not fully saturate the GPU or health signals need review.")
    return Verdict("PASS", [], "Stable sustained AI load. No driver or hardware faults detected.")


def build_markdown_report(summary: FullSummary) -> str:
    gpu = summary.gpu
    wl = summary.workload
    errs = summary.errors
    lines = [
        "# Storj: NVIDIA AI Stress Test Report",
        "",
        f"Generated: {summary.generated_at}",
        f"Artifacts dir: `{summary.outdir}`",
        "",
        f"## Verdict: {summary.verdict.status}",
        f"{summary.verdict.assessment}",
    ]
    if summary.verdict.issues:
        lines.append("")
        lines.extend(f"- {issue}" for issue in summary.verdict.issues)
    lines.extend([
        "",
        "## Workload",
        f"- GPU class: {wl.gpu_class}",
        f"- Profile: {wl.profile}",
        f"- Workload: {wl.workload_name}",
        f"- Precision: {wl.dtype.upper()}",
        f"- Batch / Seq: {wl.batch_size} / {wl.seq_len}",
        f"- Hidden / Heads / Layers: {wl.hidden_size} / {wl.heads} / {wl.layers_hint}",
        f"- Target VRAM: {fmt_num(wl.target_vram_gb,1)} GiB ({fmt_num((wl.target_vram_ratio or 0) * 100,0)}%)",
        "",
        "## Performance",
        f"- Avg tokens/sec: {fmt_num(wl.avg_tokens_per_sec,0)}",
        f"- Peak tokens/sec: {fmt_num(wl.peak_tokens_per_sec,0)}",
        f"- Avg iter time: {fmt_num(wl.avg_iter_ms,1)} ms",
        f"- P50 / P95 iter: {fmt_num(wl.p50_iter_ms,1)} / {fmt_num(wl.p95_iter_ms,1)} ms",
        f"- Iterations measured: {wl.iterations_measured}",
        "",
        "## GPU Summary",
        f"- GPU: {gpu.gpu_name or wl.gpu_name or 'n/a'}",
        f"- Driver: {gpu.driver_version or 'n/a'}",
        f"- PCI Bus ID: {gpu.pci_bus_id or 'n/a'}",
        f"- Avg GPU Util: {fmt_num(gpu.avg_gpu_util_pct,1)}%",
        f"- Min / Max GPU Util: {fmt_num(gpu.min_gpu_util_pct,0)}% / {fmt_num(gpu.max_gpu_util_pct,0)}%",
        f"- Avg Temp / Max Temp: {fmt_num(gpu.avg_temp_c,0)} C / {fmt_num(gpu.max_temp_c,0)} C",
        f"- Avg Power / Max Power: {fmt_num(gpu.avg_power_w,0)} W / {fmt_num(gpu.max_power_w,0)} W",
        f"- Avg VRAM / Max VRAM: {fmt_num(gpu.avg_mem_used_gib,1)} GiB / {fmt_num(gpu.max_mem_used_gib,1)} GiB",
        f"- PCIe Link: Gen{fmt_num(gpu.pcie_gen,0)} x{fmt_num(gpu.pcie_width,0)}",
        "",
        "## Health",
        f"- Xid errors: {errs.xid_count}",
        f"- NVRM errors: {errs.nvrm_count}",
        f"- PCIe/AER messages: {errs.pcie_aer_count}",
    ])
    if errs.new_error_lines:
        lines.extend(["", "## New dmesg lines", "```text", *errs.new_error_lines[:50], "```"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Storj AI GPU stress harness with integrated terminal UI and reporting")
    parser.add_argument("--duration", type=float, default=20.0, help="Total runtime in minutes")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup duration in seconds")
    parser.add_argument("--gpu-class", choices=["auto", "consumer", "datacenter"], default="auto")
    parser.add_argument(
        "--profile",
        choices=["auto", *PROFILE_PRESETS.keys()],
        default="auto",
        help="Workload profile preset. Use datacenter or datacenter-max for H100/H200/B100-class GPUs.",
    )
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    parser.add_argument("--target-vram-gb", type=float, default=None, help="Absolute VRAM target in GiB; overrides ratio")
    parser.add_argument("--target-vram-ratio", type=float, default=None, help="Target fraction of total VRAM to reserve/use")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--log-interval", type=float, default=1.0, help="Telemetry polling interval in seconds")
    parser.add_argument("--plain", action="store_true", help="Disable the interactive terminal dashboard")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(f"./gpu_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Output directory",
    )
    return parser


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_runtime_state(state: RuntimeState, collector: TelemetryCollector, iter_ms_values: list[float], tps_values: list[float]) -> None:
    state.latest_sample = collector.latest
    samples = collector.samples
    util_vals = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
    power_vals = [s.power_w for s in samples if s.power_w is not None]
    temp_vals = [s.temp_c for s in samples if s.temp_c is not None]
    mem_vals = [(s.mem_used_mib / 1024.0) for s in samples if s.mem_used_mib is not None]
    state.avg_gpu_util_pct = mean_or_none(util_vals)
    state.avg_power_w = mean_or_none(power_vals)
    state.max_power_w = max_or_none(power_vals)
    state.avg_temp_c = mean_or_none(temp_vals)
    state.max_temp_c = max_or_none(temp_vals)
    state.max_mem_used_gib = max_or_none(mem_vals)
    state.avg_iter_ms = mean_or_none(iter_ms_values)
    state.p50_iter_ms = percentile(iter_ms_values, 50)
    state.p95_iter_ms = percentile(iter_ms_values, 95)
    state.avg_tokens_per_sec = mean_or_none(tps_values)
    state.peak_tokens_per_sec = max_or_none(tps_values)


def choose_profile(gpu_class: str, profile_name: str) -> ProfilePreset:
    if profile_name != "auto":
        return PROFILE_PRESETS[profile_name]
    return PROFILE_PRESETS["datacenter"] if gpu_class == "datacenter" else PROFILE_PRESETS["consumer"]


def resolve_target_vram_gb(total_vram_gb: float, ratio: float, absolute: Optional[float], preset: ProfilePreset) -> float:
    if absolute is not None:
        return min(max(1.0, absolute), total_vram_gb * 0.92)
    target = total_vram_gb * ratio
    target = max(target, preset.min_target_vram_gb)
    if preset.max_target_vram_gb is not None:
        target = min(target, preset.max_target_vram_gb)
    return min(target, total_vram_gb * 0.92)


def resolve_config(args: argparse.Namespace, initial_sample: Optional[GpuSample]) -> Config:
    gpu_name = initial_sample.name if initial_sample else ""
    detected_class = detect_gpu_class(gpu_name)
    gpu_class = detected_class if args.gpu_class == "auto" else args.gpu_class
    preset = choose_profile(gpu_class, args.profile)

    dtype = preset.dtype if args.dtype == "auto" else args.dtype
    target_ratio = preset.target_vram_ratio if args.target_vram_ratio is None else args.target_vram_ratio
    if not 0.1 <= target_ratio <= 0.95:
        raise SystemExit("--target-vram-ratio must be between 0.1 and 0.95")

    return Config(
        duration_minutes=args.duration,
        warmup_seconds=args.warmup,
        dtype=dtype,
        target_vram_gb=args.target_vram_gb or preset.min_target_vram_gb,
        target_vram_ratio=target_ratio,
        batch_size=args.batch_size or preset.batch_size,
        seq_len=args.seq_len or preset.seq_len,
        hidden_size=args.hidden_size or preset.hidden_size,
        heads=args.heads or preset.heads,
        layers=args.layers or preset.layers,
        log_interval=args.log_interval,
        outdir=args.outdir.resolve(),
        plain=args.plain,
        gpu_class=gpu_class,
        profile=preset.name if args.profile == "auto" else args.profile,
        workload_name=preset.workload_name,
        power_floor_ratio=preset.power_floor_ratio,
        vram_floor_ratio=preset.vram_floor_ratio,
        util_floor_pct=preset.util_floor_pct,
    )


def run_workload(config: Config, outdir: Path, collector: TelemetryCollector, ui: ConsoleUI) -> WorkloadSummary:
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
    if config.hidden_size % config.heads != 0:
        raise SystemExit("--hidden-size must be divisible by --heads")

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    props = torch.cuda.get_device_properties(device)
    total_vram_gb = props.total_memory / (1024 ** 3)
    head_dim = config.hidden_size // config.heads
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    target_vram_gb = resolve_target_vram_gb(total_vram_gb, config.target_vram_ratio, config.target_vram_gb, choose_profile(config.gpu_class, config.profile))
    config.target_vram_gb = target_vram_gb

    q = torch.randn(config.batch_size, config.heads, config.seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    proj = torch.randn(config.hidden_size, config.hidden_size, device=device, dtype=dtype)
    residual = torch.randn(config.batch_size, config.seq_len, config.hidden_size, device=device, dtype=dtype)
    ln_weight = torch.ones(config.hidden_size, device=device, dtype=torch.float32)
    ln_bias = torch.zeros(config.hidden_size, device=device, dtype=torch.float32)

    approx_cache_gb = (
        config.batch_size * config.heads * config.seq_len * head_dim * config.layers * 2 * bytes_per_elem / (1024 ** 3)
    )
    baseline_bytes = sum(t.numel() for t in (q, k, v, proj, residual)) * bytes_per_elem
    pressure_target_gb = min(target_vram_gb, total_vram_gb * 0.92)
    pressure_remaining_gb = max(0.0, pressure_target_gb - (baseline_bytes / (1024 ** 3)) - (approx_cache_gb * 0.15))
    pressure_chunks: list[Any] = []
    chunk_gb = 0.5 if total_vram_gb < 64 else 1.0
    elems_per_chunk = int((chunk_gb * (1024 ** 3)) / bytes_per_elem)
    while pressure_remaining_gb > 0:
        pressure_chunks.append(torch.empty(elems_per_chunk, device=device, dtype=dtype))
        pressure_remaining_gb -= chunk_gb

    torch.cuda.synchronize()

    state = RuntimeState(phase="WARMUP")
    jsonl_path = outdir / "workload_metrics.jsonl"
    start_time = time.time()
    warmup_end = start_time + config.warmup_seconds
    end_time = start_time + config.duration_seconds
    next_render = start_time
    warmup_announced = False

    iterations_total = 0
    iterations_measured = 0
    tokens_measured = 0
    iter_ms_values: list[float] = []
    tps_values: list[float] = []

    def one_step() -> None:
        nonlocal q, k, v, residual
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = attn.transpose(1, 2).reshape(config.batch_size, config.seq_len, config.hidden_size)
        x = F.layer_norm(x.float(), (config.hidden_size,), ln_weight, ln_bias).to(dtype)
        x = x @ proj
        x = x + residual
        x2 = F.gelu(x)
        x = x + x2
        residual = x
        qkv = x.reshape(config.batch_size, config.seq_len, config.heads, head_dim).permute(0, 2, 1, 3).contiguous()
        q = qkv
        k = torch.roll(qkv, shifts=1, dims=2)
        v = torch.roll(qkv, shifts=2, dims=2)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        while time.time() < end_time:
            now = time.time()
            if now >= warmup_end and not warmup_announced:
                warmup_announced = True
                state.phase = "RUNNING"
                state.event_lines.append(f"{datetime.now().strftime('%H:%M:%S')} Warmup complete; throughput now counted")
            t0 = time.time()
            one_step()
            torch.cuda.synchronize()
            t1 = time.time()

            elapsed = t1 - t0
            iterations_total += 1
            state.iteration_count = iterations_total
            record: dict[str, Any] = {
                "ts": t1,
                "iteration": iterations_total,
                "iteration_ms": round(elapsed * 1000.0, 3),
                "allocated_gb": round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3),
                "reserved_gb": round(torch.cuda.memory_reserved(device) / (1024 ** 3), 3),
                "measured": 0,
            }

            if t1 >= warmup_end:
                iterations_measured += 1
                state.measured_iterations = iterations_measured
                tokens_this_iter = config.batch_size * config.seq_len
                tokens_measured += tokens_this_iter
                iter_ms = elapsed * 1000.0
                tps = tokens_this_iter / elapsed
                iter_ms_values.append(iter_ms)
                tps_values.append(tps)
                record.update({"measured": 1, "tokens_this_iter": tokens_this_iter, "tokens_per_sec": round(tps, 2)})

            handle.write(json.dumps(record) + "\n")
            handle.flush()

            if t1 >= next_render:
                update_runtime_state(state, collector, iter_ms_values, tps_values)
                ui.live(state, t1 - start_time, config.duration_seconds, config.warmup_seconds)
                next_render = t1 + config.log_interval

    update_runtime_state(state, collector, iter_ms_values, tps_values)
    return WorkloadSummary(
        gpu_name=props.name,
        gpu_class=config.gpu_class,
        profile=config.profile,
        workload_name=config.workload_name,
        total_vram_gb=round(total_vram_gb, 2),
        target_vram_gb=round(target_vram_gb, 2),
        target_vram_ratio=config.target_vram_ratio,
        dtype=str(dtype).replace("torch.", ""),
        duration_seconds=config.duration_seconds,
        warmup_seconds=config.warmup_seconds,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        hidden_size=config.hidden_size,
        heads=config.heads,
        layers_hint=config.layers,
        approx_cache_gb=round(approx_cache_gb, 2),
        iterations_total=iterations_total,
        iterations_measured=iterations_measured,
        tokens_measured=tokens_measured,
        avg_tokens_per_sec=mean_or_none(tps_values),
        peak_tokens_per_sec=max_or_none(tps_values),
        p50_tokens_per_sec=percentile(tps_values, 50),
        p95_tokens_per_sec=percentile(tps_values, 95),
        avg_iter_ms=mean_or_none(iter_ms_values),
        p50_iter_ms=percentile(iter_ms_values, 50),
        p95_iter_ms=percentile(iter_ms_values, 95),
        max_memory_allocated_gb=round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
        max_memory_reserved_gb=round(torch.cuda.max_memory_reserved(device) / (1024 ** 3), 3),
    )


def main() -> int:
    args = build_parser().parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_capture(["nvidia-smi", "-L"], outdir / "gpu_list.txt", allow_failure=False)
    run_capture(["nvidia-smi", "-q"], outdir / "nvidia_smi_q_before.txt")
    run_capture(["bash", "-lc", "dmesg | grep -iE 'NVRM|Xid|RM|PCIe|AER'"], outdir / "dmesg_before.txt")

    collector = TelemetryCollector(outdir, args.log_interval)
    collector.start()
    time.sleep(min(1.0, args.log_interval))
    initial_sample = collector.latest
    config = resolve_config(args, initial_sample)
    config.outdir = outdir
    ui = ConsoleUI(config)

    def handle_signal(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ui.startup(initial_sample)

    try:
        workload = run_workload(config, outdir, collector, ui)
    finally:
        collector.stop()

    run_capture(["nvidia-smi", "-q"], outdir / "nvidia_smi_q_after.txt")
    run_capture(["bash", "-lc", "dmesg | grep -iE 'NVRM|Xid|RM|PCIe|AER'"], outdir / "dmesg_after.txt")

    gpu = parse_gpu_metrics_from_samples(collector.samples)
    errors = parse_dmesg_errors(outdir / "dmesg_before.txt", outdir / "dmesg_after.txt")
    verdict = build_verdict(config, gpu, workload, errors)
    summary = FullSummary(generated_at=utc_now_iso(), outdir=str(outdir), gpu=gpu, workload=workload, errors=errors, verdict=verdict)

    write_json(outdir / "summary.json", asdict(summary))
    (outdir / "report.md").write_text(build_markdown_report(summary), encoding="utf-8")
    ui.final(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
