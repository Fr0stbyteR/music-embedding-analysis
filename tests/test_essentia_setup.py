"""First-run safety, platform routing and reuse without network/toolchain installs."""
import hashlib
import io
from pathlib import Path
import zipfile

import pytest

from music_annotation_backend.config import Settings
from music_annotation_backend import essentia_setup as setup
from music_annotation_backend.essentia_api import ALGORITHMS
from music_annotation_backend.essentia_python_worker import regions


class Runtime:
    def __init__(self, root, available=True, tensorflow=True):
        self.executable = root / "custom-worker"
        self.signature = "unchanged"
        self.available = available
        self.tensorflow = tensorflow

    def probe(self):
        if not self.available:
            raise RuntimeError("Missing runtime")
        return {"runtime": "test-native", "features": sorted(ALGORITHMS), "tensorflow": self.tensorflow, "tensorflowFeatures": int(self.tensorflow)}


@pytest.fixture
def install(tmp_path, monkeypatch):
    settings = Settings(vendor_root=tmp_path / "vendor", model_root=tmp_path / "models",
        essentia_native_executable=None, essentia_setup="auto", essentia_setup_mood=False, essentia_setup_tensorflow=False)
    runtime = Runtime(tmp_path)
    checks = []
    monkeypatch.setattr(setup, "feature_runtime", lambda _: runtime)
    monkeypatch.setattr(setup, "verify_features", lambda _: checks.append("features"))
    return settings, runtime, checks, tmp_path


@pytest.mark.parametrize("system,builder", [("Windows", "build_windows"), ("Darwin", "install_mac")])
def test_first_run_installs_then_checks_all_features_and_reuses(install, monkeypatch, system, builder):
    settings, runtime, checks, root = install
    runtime.available = False
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    builds = []
    def build(*args):
        builds.append(system)
        runtime.available = True
    monkeypatch.setattr(setup, builder, build)
    setup.ensure_essentia(settings, root, "uv")
    setup.ensure_essentia(settings, root, "uv")
    assert builds == [system]
    assert checks == ["features"]
    assert (root / ".tools/essentia-runtime/ready.json").is_file()


def test_off_does_not_probe_or_install(install, monkeypatch):
    settings, _, checks, root = install
    settings.essentia_setup = "off"
    monkeypatch.setattr(setup, "feature_runtime", lambda _: pytest.fail("Unexpected probe"))
    setup.ensure_essentia(settings, root, "uv")
    assert not checks


def test_broken_explicit_runtime_is_not_silently_replaced(install, monkeypatch):
    settings, runtime, _, root = install
    settings.essentia_native_executable = root / "user.exe"
    runtime.available = False
    monkeypatch.setattr(setup.platform, "system", lambda: "Windows")
    monkeypatch.setattr(setup, "build_windows", lambda *a: pytest.fail("Overrode custom runtime"))
    with pytest.raises(RuntimeError, match="Configured MAB_ESSENTIA_NATIVE_EXECUTABLE"):
        setup.ensure_essentia(settings, root, "uv")


def test_broken_ready_manifest_is_rechecked(install, monkeypatch):
    settings, _, checks, root = install
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    manifest = root / ".tools/essentia-runtime/ready.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"\xff")
    setup.ensure_essentia(settings, root, "uv")
    assert checks == ["features"]
    assert '"runtime"' in manifest.read_text()


def test_basic_runs_dsp_without_downloading_mood_weights(install, monkeypatch):
    settings, _, checks, root = install
    settings.essentia_setup_mood = True
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(setup, "download", lambda *a: pytest.fail("Downloaded VA weights in basic mode"))
    setup.ensure_essentia(settings, root, "uv", basic=True)
    assert checks == ["features"]


def test_full_setup_checks_real_mood_inference_before_marking_ready(install, monkeypatch):
    settings, _, checks, root = install
    settings.essentia_setup_mood = True
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    def download(entry, folder):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / entry.name
        path.write_bytes(entry.name.encode())
        return path
    monkeypatch.setattr(setup, "download", download)
    class Mood:
        native_selected = False
        def __init__(self, _):
            self.embedding_path, self.regression_path = [settings.model_root / "essentia" / entry.name for entry in setup.MODELS]
        def _load(self):
            checks.append("load-va")
        def predict(self, audio):
            assert len(audio) == 48000
            checks.append("infer-va")
    monkeypatch.setattr(setup, "MoodAnalyzer", Mood)
    setup.ensure_essentia(settings, root, "uv")
    setup.ensure_essentia(settings, root, "uv")
    assert checks == ["features", "load-va", "infer-va"]


def test_self_test_failure_does_not_write_ready_manifest(install, monkeypatch):
    settings, _, _, root = install
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    def fail(_):
        raise RuntimeError("GFCC failed")
    monkeypatch.setattr(setup, "verify_features", fail)
    with pytest.raises(RuntimeError, match="GFCC"):
        setup.ensure_essentia(settings, root, "uv")
    assert not (root / ".tools/essentia-runtime/ready.json").exists()


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_tensorflow_setup_downloads_checks_and_reuses_on_each_platform(install, monkeypatch, system):
    settings, _, checks, root = install
    settings.essentia_setup_tensorflow = True
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    downloaded = []
    def download(entry, folder):
        downloaded.append(entry.name)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / entry.name
        target.write_bytes(entry.name.encode())
        return target
    monkeypatch.setattr(setup, "download", download)
    monkeypatch.setattr(setup, "verify_tensorflow", lambda _: checks.append("tf-inference"))
    setup.ensure_essentia(settings, root, "uv")
    setup.ensure_essentia(settings, root, "uv")
    assert checks == ["features", "tf-inference"]
    assert len(set(downloaded)) == 34
    assert '"tensorflow": true' in (root / ".tools/essentia-runtime/ready.json").read_text()


def test_tensorflow_failed_inference_never_marks_install_ready(install, monkeypatch):
    settings, _, _, root = install
    settings.essentia_setup_tensorflow = True
    monkeypatch.setattr(setup.platform, "system", lambda: "Darwin")
    def download(entry, folder):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / entry.name
        target.write_bytes(entry.name.encode())
        return target
    monkeypatch.setattr(setup, "download", download)
    def fail(_):
        raise RuntimeError("TF inference failed")
    monkeypatch.setattr(setup, "verify_tensorflow", fail)
    with pytest.raises(RuntimeError, match="TF inference failed"):
        setup.ensure_essentia(settings, root, "uv")
    assert not (root / ".tools/essentia-runtime/ready.json").exists()


@pytest.mark.parametrize("machine", ["x86_64", "arm64"])
def test_mac_installs_architecture_specific_pinned_wheel(tmp_path, monkeypatch, machine):
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    monkeypatch.setattr(setup.sys, "version_info", (3, 11, 14))
    downloads, commands = [], []
    def download(entry, folder):
        downloads.append(entry)
        return folder / entry.name
    monkeypatch.setattr(setup, "download", download)
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kwargs: commands.append(command))
    setup.install_mac(tmp_path, "uv")
    assert downloads == [setup.MAC_WHEELS[machine]]
    assert len(downloads[0].sha256) == 64
    assert commands[0][1:3] == ["pip", "uninstall"]
    assert "numpy>=1.26,<2" in commands[1]
    assert commands[1][-1].endswith(downloads[0].name)


def test_verified_download_is_cached_and_mismatch_preserves_existing_file(tmp_path, monkeypatch):
    payload = b"official-binary"
    entry = setup.Download("binary.zip", "https://example.invalid/binary", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(setup, "urlopen", lambda *a, **k: io.BytesIO(payload))
    target = setup.download(entry, tmp_path)
    assert target.read_bytes() == payload
    monkeypatch.setattr(setup, "urlopen", lambda *a, **k: pytest.fail("Repeated download"))
    assert setup.download(entry, tmp_path) == target
    target.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="not overwritten"):
        setup.download(entry, tmp_path)
    assert target.read_bytes() == b"corrupted"


def test_bad_download_does_not_leave_partial_or_final_file(tmp_path, monkeypatch):
    entry = setup.Download("binary.zip", "https://example.invalid/binary", "0" * 64)
    monkeypatch.setattr(setup, "urlopen", lambda *a, **k: io.BytesIO(b"bad"))
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        setup.download(entry, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../outside", "bad")
    with pytest.raises(RuntimeError, match="Unsafe archive"):
        setup.extract(archive, tmp_path / "runtime")
    assert not (tmp_path / "outside").exists()


def test_python_worker_marker_regions_clamp_to_audio_duration():
    intervals, labels = regions([-1, 60, 60, -1, 61, 61], .55, .1, .15, str)
    assert intervals == [[.1, pytest.approx(.3)], [.4, .55]]
    assert labels == ["60", "61"]
