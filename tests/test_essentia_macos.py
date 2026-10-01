"""SDL2 regression: prevent the wheel's NSAlert/abort before any native import."""
import hashlib
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from music_annotation_backend import essentia_macos as mac
from music_annotation_backend import essentia_setup as setup
from music_annotation_backend import essentia_python_worker as dsp
from music_annotation_backend import essentia_tf_worker as tf
from music_annotation_backend.config import Settings
from music_annotation_backend.mood import MoodAnalyzer


@pytest.fixture
def wheel(tmp_path, monkeypatch):
    package = tmp_path / "essentia"
    directory = package / ".dylibs"
    directory.mkdir(parents=True)
    shim = directory / "libSDL-1.2.0.dylib"
    shim.write_bytes(b"sdl12-compat: Failed loading SDL2 library.")
    monkeypatch.setattr(mac.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mac.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(package / "__init__.py")))
    mac._load_sdl2.cache_clear()
    yield directory
    mac._load_sdl2.cache_clear()


def test_locates_shim_without_importing_native_package(wheel):
    assert mac.sdl_compat_directory() == wheel


@pytest.mark.parametrize("system", ["Windows", "Linux"])
def test_other_platforms_do_not_touch_sdl(monkeypatch, system):
    monkeypatch.setattr(mac.platform, "system", lambda: system)
    monkeypatch.setattr(mac.importlib.util, "find_spec", lambda _: pytest.fail("Unexpected package lookup"))
    assert mac.sdl_compat_directory() is None
    mac.require_macos_sdl2()


def test_classic_sdl_wheel_does_not_need_sdl2(wheel):
    (wheel / "libSDL-1.2.0.dylib").write_bytes(b"classic SDL 1.2")
    assert mac.sdl_compat_directory() is None
    mac.require_macos_sdl2()


@pytest.mark.parametrize("entry", [
    lambda: dsp.probe(),
    lambda: dsp.analyze(None, None, None),
    lambda: tf.predict(None, None, None, None),
    lambda: tf.backbone(None, None, None, None),
    lambda: tf.head(None, None, None, None, None, None),
])
def test_missing_sdl2_fails_before_all_native_imports(wheel, entry):
    with pytest.raises(RuntimeError, match="Restart with bash start.command"):
        entry()  # No essentia installed here; the guard must run before import.


def test_mood_capability_cannot_open_fatal_dialog(wheel, tmp_path):
    analyzer = MoodAnalyzer(Settings(vendor_root=tmp_path / "vendor", essentia_native_executable=None))
    capability = analyzer.capabilities()
    assert not capability["available"]
    assert "requires SDL2" in capability["reason"]


def test_guard_checks_checksum_before_loading(wheel, monkeypatch):
    target = wheel / mac.SDL2_NAME
    target.write_bytes(b"corrupted")
    monkeypatch.setattr(mac.ctypes, "CDLL", lambda *a, **k: pytest.fail("Loaded unverified binary"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        mac.require_macos_sdl2()
    assert target.read_bytes() == b"corrupted"


def test_verified_library_is_loaded_once_and_kept_alive(wheel, monkeypatch):
    payload = b"verified-universal-library"
    target = wheel / mac.SDL2_NAME
    target.write_bytes(payload)
    monkeypatch.setattr(mac, "SDL2_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return object()
    monkeypatch.setattr(mac.ctypes, "CDLL", load)
    mac.require_macos_sdl2()
    mac.require_macos_sdl2()
    assert calls == [(str(target), {"mode": mac.ctypes.RTLD_LOCAL})]


def test_load_failure_is_actionable_python_error(wheel, monkeypatch):
    target = wheel / mac.SDL2_NAME
    target.write_bytes(b"library")
    monkeypatch.setattr(mac, "SDL2_SHA256", hashlib.sha256(b"library").hexdigest())
    def fail(*a, **k):
        raise OSError("incompatible architecture")
    monkeypatch.setattr(mac.ctypes, "CDLL", fail)
    with pytest.raises(RuntimeError, match="SDL2 could not load.*incompatible architecture"):
        mac.require_macos_sdl2()


@pytest.fixture
def mounted_image(wheel, tmp_path, monkeypatch):
    payload = b"verified-universal-library"
    checksum = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(setup, "SDL2_SHA256", checksum)
    downloads, commands = [], []
    def download(entry, folder):
        downloads.append(entry)
        return tmp_path / entry.name
    def run(command, **kwargs):
        commands.append((command, kwargs))
        if "attach" in command:
            mount = Path(command[-1])
            binary = mount / "SDL2.framework/Versions/A/SDL2"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(payload)
            (mount / "License.txt").write_text("SDL2 zlib license", encoding="utf-8")
    monkeypatch.setattr(setup, "download", download)
    monkeypatch.setattr(setup.subprocess, "run", run)
    return payload, downloads, commands


def test_setup_installs_beside_shim_with_license_then_reuses(wheel, mounted_image, tmp_path, monkeypatch):
    payload, downloads, commands = mounted_image
    setup.prepare_mac_sdl2(tmp_path)
    assert (wheel / mac.SDL2_NAME).read_bytes() == payload
    assert (wheel / "SDL2-LICENSE.txt").read_text() == "SDL2 zlib license"
    assert downloads == [setup.SDL2_MAC]
    assert commands[0][0][:2] == ["/usr/bin/hdiutil", "attach"]
    assert "-readonly" in commands[0][0] and "-nobrowse" in commands[0][0]
    assert commands[1][0][:2] == ["/usr/bin/hdiutil", "detach"]
    assert list(wheel.glob("*.partial")) == []
    monkeypatch.setattr(setup, "download", lambda *a: pytest.fail("Repeated download"))
    setup.prepare_mac_sdl2(tmp_path)
    assert len(commands) == 2


def test_setup_does_not_overwrite_unrecognized_library(wheel, tmp_path, monkeypatch):
    target = wheel / mac.SDL2_NAME
    target.write_bytes(b"user-library")
    monkeypatch.setattr(setup, "download", lambda *a: pytest.fail("Unexpected download"))
    with pytest.raises(RuntimeError, match="not overwritten"):
        setup.prepare_mac_sdl2(tmp_path)
    assert target.read_bytes() == b"user-library"


def test_bad_image_binary_is_rejected_and_always_detached(wheel, mounted_image, tmp_path, monkeypatch):
    _, _, commands = mounted_image
    monkeypatch.setattr(setup, "SDL2_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="binary checksum mismatch"):
        setup.prepare_mac_sdl2(tmp_path)
    assert commands[-1][0][1] == "detach"
    assert not (wheel / mac.SDL2_NAME).exists()


def test_install_failure_detaches_and_cleans_partial(wheel, mounted_image, tmp_path, monkeypatch):
    _, _, commands = mounted_image
    def fail(*a):
        raise OSError("no space")
    monkeypatch.setattr(setup.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="no space"):
        setup.prepare_mac_sdl2(tmp_path)
    assert commands[-1][0][1] == "detach"
    assert not (wheel / mac.SDL2_NAME).exists()
    assert list(wheel.glob("*.partial")) == []


def test_existing_wheel_is_repaired_before_first_probe(tmp_path, monkeypatch):
    settings = Settings(vendor_root=tmp_path / "vendor", essentia_native_executable=None,
        essentia_setup="auto", essentia_setup_mood=False, essentia_setup_tensorflow=False)
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    events = []
    monkeypatch.setattr(setup, "prepare_mac_sdl2", lambda _: events.append("sdl"))
    class Runtime:
        signature = "test"
        executable = tmp_path / "worker"
        def probe(self):
            events.append("probe")
            assert events[0] == "sdl"
            return {"runtime": "test-native", "features": sorted(setup.ALGORITHMS)}
    monkeypatch.setattr(setup, "feature_runtime", lambda _: Runtime())
    monkeypatch.setattr(setup, "verify_features", lambda _: None)
    setup.ensure_essentia(settings, tmp_path, "uv", basic=True)
    assert events[:2] == ["sdl", "probe"]


@pytest.mark.skipif(platform.system() != "Darwin", reason="Actual macOS SDL2/Essentia import checked on both Mac CI runners")
def test_real_mac_library_loads_before_essentia_probe():
    # CI prepares the runtime before pytest. The ARM wheel must find our library.
    directory = mac.sdl_compat_directory()
    if platform.machine() == "arm64":
        assert directory is not None
    if directory is not None:
        assert (directory / mac.SDL2_NAME).is_file()
        mac.require_macos_sdl2()
    assert dsp.probe()["tensorflowFeatures"] == 1
