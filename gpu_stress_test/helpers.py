from __future__ import annotations

import math
import shutil
import statistics
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NVIDIA_SMI_QUERY_FIELDS = [
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

NVIDIA_SMI_FILTER_REGEX = "NVRM|Xid|RM|PCIe|AER"


def utc_now_iso() -> str:
    """Return current UTC time as an ISO 8601 string (no microseconds)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def maybe_float(value: Any) -> float | None:
    """Parse a value to float, returning None for missing/N/A values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "[N/A]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean_or_none(values: Iterable[float]) -> float | None:
    """Return the arithmetic mean of values, or None if the sequence is empty."""
    vals = list(values)
    return statistics.mean(vals) if vals else None


def max_or_none(values: Iterable[float]) -> float | None:
    """Return the maximum of values, or None if the sequence is empty."""
    vals = list(values)
    return max(vals) if vals else None


def percentile(values: Iterable[float], p: float) -> float | None:
    """Return the p-th percentile (0–100) using the nearest-rank method."""
    vals = sorted(values)
    if not vals:
        return None
    # rank is 1-indexed; clamp to [1, n] then convert to 0-indexed
    rank = math.ceil(p / 100.0 * len(vals))
    return vals[max(0, rank - 1)]


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    """Format an optional float with fixed decimal places and an optional suffix."""
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS or MM:SS."""
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def clear_terminal() -> None:
    """Emit ANSI escape codes to clear the terminal when stdout is a TTY."""
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def terminal_supports_pretty() -> bool:
    """Return True when stdout is a TTY and --plain was not requested."""
    return sys.stdout.isatty() and "--plain" not in sys.argv


def run_capture(command: list[str], output_path: Path, allow_failure: bool = True) -> int:
    """Run a subprocess, writing combined stdout/stderr to output_path."""
    with output_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 and not allow_failure:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}")
    return proc.returncode


def query_nvidia_smi(field: str) -> str:
    """Query a single nvidia-smi field and return the first output line."""
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        text=True,
    )
    return output.splitlines()[0].strip()


def get_gpu_mem_used_gib() -> float:
    """Return current GPU memory usage in GiB via nvidia-smi."""
    return float(query_nvidia_smi("memory.used")) / 1024.0


def get_gpu_snapshot() -> dict[str, float | str | None]:
    """Query a snapshot of GPU state (utilization, memory, power, clocks, PCIe) in one call."""
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join([
                "name",
                "utilization.gpu",
                "utilization.memory",
                "memory.used",
                "memory.total",
                "power.draw",
                "power.limit",
                "temperature.gpu",
                "clocks.current.graphics",
                "clocks.current.memory",
                "pstate",
                "pcie.link.gen.current",
                "pcie.link.width.current",
            ]),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
    row = [part.strip() for part in raw.split(",")]
    mem_used = maybe_float(row[3])
    mem_total = maybe_float(row[4])
    return {
        "name": row[0],
        "gpu_util_pct": maybe_float(row[1]),
        "mem_util_pct": maybe_float(row[2]),
        "mem_used_gib": (mem_used / 1024.0) if mem_used is not None else None,
        "mem_total_gib": (mem_total / 1024.0) if mem_total is not None else None,
        "power_w": maybe_float(row[5]),
        "power_limit_w": maybe_float(row[6]),
        "temp_c": maybe_float(row[7]),
        "graphics_mhz": maybe_float(row[8]),
        "mem_mhz": maybe_float(row[9]),
        "pstate": row[10],
        "pcie_gen": maybe_float(row[11]),
        "pcie_width": maybe_float(row[12]),
    }


def ensure_command(name: str) -> None:
    """Raise SystemExit if a required external command is not on PATH."""
    if shutil.which(name) is None:
        raise SystemExit(f"Required command not found: {name}")
