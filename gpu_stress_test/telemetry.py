from __future__ import annotations

import subprocess
from pathlib import Path

from .helpers import NVIDIA_SMI_QUERY_FIELDS


class NvidiaSmiLogger:
    """Background nvidia-smi poller that streams CSV telemetry to a file."""

    def __init__(self, output_path: Path, interval_seconds: int) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self._handle: object = None
        self._process: subprocess.Popen[str] | None = None

    def start(self) -> None:
        """Spawn nvidia-smi in continuous-loop mode, writing CSV rows to output_path."""
        self._handle = self.output_path.open("w", encoding="utf-8")
        command = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS),
            "--format=csv,noheader,nounits",
            "-l",
            str(self.interval_seconds),
        ]
        self._process = subprocess.Popen(command, stdout=self._handle, stderr=subprocess.STDOUT, text=True)

    def stop(self) -> None:
        """Terminate the nvidia-smi process and flush/close the output file. Idempotent."""
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
        if self._handle is not None:
            self._handle.close()  # type: ignore[union-attr]
            self._handle = None
