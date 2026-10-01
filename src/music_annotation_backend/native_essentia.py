"""Optional out-of-process Essentia worker: no Python extension ABI dependency."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryFile

import numpy as np


class NativeEssentia:
    def __init__(self, executable: Path, timeout_seconds: float = 1800, *, command_prefix=None, expected_runtime="essentia-cpp", signature_paths=None):
        self.executable = executable.resolve()
        self.timeout_seconds = timeout_seconds
        self.signature = None
        self.info = None
        self.command_prefix = command_prefix or [str(self.executable)]
        self.expected_runtime = expected_runtime
        self.signature_paths = signature_paths or [self.executable, self.executable.parent / "tensorflow.dll"]

    def _run(self, arguments, *, stdin=None, timeout=30):
        try:
            result = subprocess.run(
                [*self.command_prefix, *map(str, arguments)], stdin=stdin,
                capture_output=True, timeout=timeout, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"},
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Native Essentia could not run: {error}") from error
        if result.returncode:
            diagnostic = result.stderr.decode("utf-8", errors="replace")[-2000:].strip()
            raise RuntimeError(f"Native Essentia exited with {result.returncode}: {diagnostic or 'Check the TensorFlow DLL and C++ runtime.'}")
        try:
            return json.loads(result.stdout)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Native Essentia returned invalid JSON") from error

    def probe(self) -> dict:
        signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in self.signature_paths if path.is_file())
        if signature != self.signature or self.info is None:
            info = self._run(["--capabilities"])
            if not isinstance(info, dict) or info.get("protocol") != 1 or info.get("runtime") != self.expected_runtime:
                raise RuntimeError("Unsupported native Essentia worker protocol")
            self.info, self.signature = info, signature
        return self.info

    def curve(self, audio: np.ndarray, embedding: Path, regression: Path, window: float, hop: float, duration: float) -> list[dict]:
        with TemporaryFile() as pcm:
            np.asarray(audio, dtype="<f4").tofile(pcm)
            pcm.seek(0)
            result = self._run(["--mood", embedding.resolve(), regression.resolve(), window, hop, duration], stdin=pcm, timeout=self.timeout_seconds)
        expected = math.floor(duration / hop) + 1
        if not isinstance(result, dict) or result.get("protocol") != 1 or not isinstance(result.get("points"), list) or len(result["points"]) != expected:
            raise RuntimeError("Native Essentia returned an invalid mood curve")
        for index, point in enumerate(result["points"]):
            if not isinstance(point, dict):
                raise RuntimeError("Native Essentia returned an invalid mood point")
            values = [point.get(key) for key in ("timeSeconds", "valence", "arousal")]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise RuntimeError("Native Essentia returned non-finite mood values")
            if abs(values[0] - min(duration, index * hop)) > .00001 or any(abs(value) > 1 for value in values[1:]):
                raise RuntimeError("Native Essentia returned out-of-range mood values")
        return result["points"]
