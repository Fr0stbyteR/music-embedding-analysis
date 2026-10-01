"""Select an isolated native C++ executable or official C++ Python wheel."""
import importlib.util
from pathlib import Path
import platform
import sys

from .native_essentia import NativeEssentia


def native_executable(settings):
    if settings.essentia_native_executable is not None:
        return settings.essentia_native_executable
    root = settings.vendor_root.parent
    candidates = [root / ".tools/essentia-runtime/essentia-worker.exe", root / "build/essentia/Release/essentia-worker.exe"]
    return next((path for path in candidates if path.is_file()), candidates[-1])


def feature_runtime(settings):
    executable = native_executable(settings)
    if settings.essentia_native_executable is not None or executable.is_file() or platform.system() == "Windows":
        return NativeEssentia(executable, settings.essentia_native_timeout_seconds)
    spec = importlib.util.find_spec("essentia")
    paths = [Path(__file__).with_name(name) for name in ("essentia_python_worker.py", "essentia_tf_worker.py")]
    if spec and spec.origin:
        paths.extend(Path(spec.origin).parent.glob("*.so"))
        paths.append(Path(spec.origin))
    return NativeEssentia(Path(sys.executable), settings.essentia_native_timeout_seconds,
        command_prefix=[sys.executable, "-m", "music_annotation_backend.essentia_python_worker"],
        expected_runtime="essentia-python", signature_paths=paths)
