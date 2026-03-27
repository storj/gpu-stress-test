#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gpu_stress_test.helpers import NVIDIA_SMI_FILTER_REGEX, ensure_command, query_nvidia_smi, run_capture, utc_now_iso
from gpu_stress_test.models import FullSummary
from gpu_stress_test.render import ConsoleRenderer
from gpu_stress_test.reporting import (
    build_markdown_report,
    build_verdict,
    parse_dmesg_errors,
    parse_gpu_metrics,
    write_json,
)
from gpu_stress_test.telemetry import NvidiaSmiLogger
from gpu_stress_test.workload import run_workload


# ---------------------------------------------------------------------------
# Profile presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProfilePreset:
    """Workload shape and verdict thresholds for a GPU class."""
    dtype: str
    batch_size: int
    seq_len: int
    hidden_size: int
    heads: int
    layers: int
    target_vram_ratio: float
    power_floor_ratio: float   # warn if max_power < this fraction of rated limit
    util_floor_pct: float      # warn if avg GPU util below this percentage


# Profile weight memory and peak FFN intermediate (bf16, ffn_multiplier=4):
#   consumer       32 layers × 384 MiB = 12 GiB weights;  FFN peak  2 GiB  → 14 GiB active
#   datacenter     40 layers × 384 MiB = 15 GiB weights;  FFN peak  4 GiB  → 19 GiB active
#   datacenter-max 80 layers × 384 MiB = 30 GiB weights;  FFN peak  8 GiB  → 38 GiB active
PROFILE_PRESETS: dict[str, ProfilePreset] = {
    "consumer": ProfilePreset(
        dtype="bf16",
        batch_size=16, seq_len=4096, hidden_size=4096, heads=32, layers=32,
        target_vram_ratio=0.70,
        power_floor_ratio=0.55,
        util_floor_pct=85.0,
    ),
    "datacenter": ProfilePreset(
        dtype="bf16",
        batch_size=32, seq_len=4096, hidden_size=4096, heads=32, layers=40,
        target_vram_ratio=0.80,
        power_floor_ratio=0.50,
        util_floor_pct=85.0,
    ),
    "datacenter-max": ProfilePreset(
        # Targets H200 / B100 / B200 (80–192 GiB). Also runs on H100 SXM (80 GiB).
        dtype="bf16",
        batch_size=32, seq_len=8192, hidden_size=4096, heads=32, layers=80,
        target_vram_ratio=0.82,
        power_floor_ratio=0.50,
        util_floor_pct=85.0,
    ),
}

# Substrings that identify datacenter-class GPU names reported by nvidia-smi
_DATACENTER_MARKERS = (
    "H100", "H200", "B100", "B200", "A100", "A800", "H800",
    "L40", "L4 ", "TESLA", "SXM", "HGX",
    "RTX PRO 6000 BLACKWELL SERVER",
)


def detect_gpu_class(gpu_name: str) -> str:
    """Return 'datacenter' if the GPU name matches known server SKUs, else 'consumer'."""
    upper = gpu_name.upper()
    return "datacenter" if any(m in upper for m in _DATACENTER_MARKERS) else "consumer"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Fully-resolved run parameters after profile merging and CLI overrides."""

    gpu_class: str
    profile: str
    duration_minutes: int
    warmup_seconds: int
    dtype: str
    target_vram_ratio: float
    target_vram_gib: float | None
    batch_size: int
    seq_len: int
    hidden_size: int
    heads: int
    layers: int
    ffn_multiplier: int
    power_floor_ratio: float
    util_floor_pct: float
    log_interval: int
    outdir: Path
    plain: bool

    @property
    def duration_seconds(self) -> int:
        return self.duration_minutes * 60


def _resolve_config(args: argparse.Namespace, gpu_name: str) -> Config:
    """Merge profile preset with explicit CLI overrides; explicit flags always win."""
    gpu_class = detect_gpu_class(gpu_name) if args.gpu_class == "auto" else args.gpu_class

    if args.profile == "auto":
        profile_name = "datacenter" if gpu_class == "datacenter" else "consumer"
    else:
        profile_name = args.profile

    preset = PROFILE_PRESETS[profile_name]

    return Config(
        gpu_class=gpu_class,
        profile=profile_name,
        duration_minutes=args.duration,
        warmup_seconds=args.warmup,
        dtype=args.dtype if args.dtype is not None else preset.dtype,
        target_vram_ratio=args.target_vram_ratio if args.target_vram_ratio is not None else preset.target_vram_ratio,
        target_vram_gib=args.target_vram_gb,
        batch_size=args.batch_size if args.batch_size is not None else preset.batch_size,
        seq_len=args.seq_len if args.seq_len is not None else preset.seq_len,
        hidden_size=args.hidden_size if args.hidden_size is not None else preset.hidden_size,
        heads=args.heads if args.heads is not None else preset.heads,
        layers=args.layers if args.layers is not None else preset.layers,
        ffn_multiplier=args.ffn_multiplier,
        power_floor_ratio=preset.power_floor_ratio,
        util_floor_pct=preset.util_floor_pct,
        log_interval=args.log_interval,
        outdir=args.outdir.resolve(),
        plain=args.plain,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Storj: AI GPU stress harness with integrated reporting")
    parser.add_argument("--duration", type=int, default=20, help="Total runtime in minutes (default: 20)")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup duration in seconds (default: 30)")
    parser.add_argument(
        "--profile",
        choices=["auto", *PROFILE_PRESETS],
        default="auto",
        help="Workload preset. auto selects consumer or datacenter based on GPU (default: auto)",
    )
    parser.add_argument(
        "--gpu-class",
        choices=["auto", "consumer", "datacenter"],
        default="auto",
        help="Override GPU class used for auto profile selection (default: auto)",
    )
    # Profile-governed params — None means 'use profile default'
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default=None)
    parser.add_argument("--target-vram-ratio", type=float, default=None, help="Target fraction of total VRAM")
    parser.add_argument("--target-vram-gb", type=float, default=None, help="Absolute VRAM target in GiB; overrides ratio")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    # Fixed-default params
    parser.add_argument("--ffn-multiplier", type=int, default=4, help="FFN hidden-dim expansion factor (default: 4)")
    parser.add_argument("--log-interval", type=int, default=1, help="Telemetry interval in seconds (default: 1)")
    parser.add_argument("--plain", action="store_true", help="Emit line-oriented console output")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(f"./gpu_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
        help="Output directory",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse args, detect GPU, resolve profile, run workload, produce report."""
    args = build_parser().parse_args()

    ensure_command("nvidia-smi")

    # Query the GPU name early so profile auto-detection can use it
    gpu_name = query_nvidia_smi("name")
    config = _resolve_config(args, gpu_name)
    config.outdir.mkdir(parents=True, exist_ok=True)

    renderer = ConsoleRenderer(plain=config.plain)
    logger = NvidiaSmiLogger(config.outdir / "gpu_metrics.csv", config.log_interval)

    def cleanup(signum: int | None = None, frame: Any = None) -> None:  # noqa: ARG001
        # NvidiaSmiLogger.stop() is idempotent; the finally block below also calls it.
        logger.stop()
        if signum is not None:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    renderer.startup(config)

    run_capture(["nvidia-smi", "-L"], config.outdir / "gpu_list.txt", allow_failure=False)
    run_capture(["nvidia-smi", "-q"], config.outdir / "nvidia_smi_q_before.txt")
    run_capture(["bash", "-lc", f"dmesg | grep -iE '{NVIDIA_SMI_FILTER_REGEX}'"], config.outdir / "dmesg_before.txt")

    try:
        logger.start()
        workload = run_workload(config, config.outdir, renderer)
    finally:
        logger.stop()

    run_capture(["nvidia-smi", "-q"], config.outdir / "nvidia_smi_q_after.txt")
    run_capture(["bash", "-lc", f"dmesg | grep -iE '{NVIDIA_SMI_FILTER_REGEX}'"], config.outdir / "dmesg_after.txt")

    gpu = parse_gpu_metrics(config.outdir / "gpu_metrics.csv")
    errors = parse_dmesg_errors(config.outdir / "dmesg_before.txt", config.outdir / "dmesg_after.txt")
    verdict = build_verdict(gpu, workload, errors)

    summary = FullSummary(
        generated_at=utc_now_iso(),
        outdir=str(config.outdir),
        config={
            "gpu_class": config.gpu_class,
            "profile": config.profile,
            "duration_minutes": config.duration_minutes,
            "warmup_seconds": config.warmup_seconds,
            "dtype": config.dtype,
            "target_vram_ratio": config.target_vram_ratio,
            "target_vram_gib": config.target_vram_gib,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "hidden_size": config.hidden_size,
            "heads": config.heads,
            "layers": config.layers,
            "ffn_multiplier": config.ffn_multiplier,
            "log_interval": config.log_interval,
            "plain": config.plain,
        },
        gpu=gpu,
        workload=workload,
        errors=errors,
        verdict=verdict,
    )

    write_json(config.outdir / "summary.json", asdict(summary))
    (config.outdir / "report.md").write_text(build_markdown_report(summary), encoding="utf-8")
    renderer.final(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
