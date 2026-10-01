"""Guard the wheel's SDL 1.2 shim before it can open a fatal macOS dialog.

The pinned ARM wheel bundles sdl12-compat, but omits its dlopen dependency
SDL2. The launcher installs the official universal SDL2 binary beside the
shim, at its first @loader_path lookup. No Homebrew/DYLD environment is used.
"""
import ctypes
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import platform

SDL2_NAME = "libSDL2-2.0.0.dylib"
# Unmodified universal x86_64/arm64 binary from the verified SDL2 2.32.10 DMG.
SDL2_SHA256 = "bc96277325b2e1dc75a13cf70f5ddcf63005d29e93f54a8b7adc7f5c3c017b91"


def sdl_compat_directory():
    """Locate without importing essentia: importing already runs the constructor."""
    if platform.system() != "Darwin":
        return None
    spec = importlib.util.find_spec("essentia")
    if not spec or not spec.origin:
        return None
    directory = Path(spec.origin).parent / ".dylibs"
    shim = directory / "libSDL-1.2.0.dylib"
    if shim.is_file() and b"sdl12-compat" in shim.read_bytes():
        return directory
    return None  # Other wheels may use classic SDL 1.2, with no SDL2 dependency.


@lru_cache(maxsize=4)
def _load_sdl2(path, modified, size):
    # Keep the CDLL alive and avoid repeatedly hashing/loading in the API process.
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SDL2_SHA256:
            raise RuntimeError(f"Essentia SDL2 checksum mismatch: {path}. Move this file aside and restart with bash start.command; it was not overwritten.")
    try:
        return ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    except OSError as error:
        raise RuntimeError(f"Essentia SDL2 could not load for {platform.machine()}: {error}. Restart with bash start.command.") from error


def require_macos_sdl2():
    """Fail in Python, not inside the shim's blocking NSAlert/abort constructor."""
    directory = sdl_compat_directory()
    if directory is None:
        return
    path = directory / SDL2_NAME
    if not path.is_file():
        raise RuntimeError("Essentia wheel requires SDL2, which is missing. Restart with bash start.command to install the verified project-local runtime (no Homebrew needed).")
    stat = path.stat()
    _load_sdl2(str(path), stat.st_mtime_ns, stat.st_size)
