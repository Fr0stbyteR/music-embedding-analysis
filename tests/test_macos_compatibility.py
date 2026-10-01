"""Regression guards for the actual Apple Silicon/Intel CI failure paths."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement

from music_annotation_backend.config import Settings
from music_annotation_backend import essentia_runtime
from music_annotation_backend.mood import MoodAnalyzer


ROOT = Path(__file__).resolve().parents[1]


def test_intel_mac_jit_pins_do_not_downgrade_other_platforms():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [Requirement(value) for value in config["tool"]["uv"]["constraint-dependencies"]]
    for name, version in (("numba", "0.62.1"), ("llvmlite", "0.45.1")):
        pin = next(value for value in pins if value.name == name)
        assert str(pin.specifier) == f"=={version}"
        assert pin.marker.evaluate({"sys_platform": "darwin", "platform_machine": "x86_64"})
        for system, machine in (("darwin", "arm64"), ("win32", "AMD64"), ("linux", "x86_64")):
            assert not pin.marker.evaluate({"sys_platform": system, "platform_machine": machine})


def test_locked_intel_jit_stack_contains_official_wheels_for_both_python_versions():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for name, version in (("numba", "0.62.1"), ("llvmlite", "0.45.1")):
        package = next(value for value in lock["package"] if value["name"] == name and value["version"] == version)
        for python in ("cp311", "cp312"):
            assert any(f"{python}-{python}-macosx_" in wheel["url"] and wheel["url"].endswith("x86_64.whl") for wheel in package["wheels"])


def test_configured_probe_timeout_reaches_cpp_python_and_mood_workers(monkeypatch, tmp_path):
    settings = Settings(vendor_root=tmp_path / "vendor", model_root=tmp_path / "models",
        essentia_native_executable=None, essentia_probe_timeout_seconds=240)
    monkeypatch.setattr(essentia_runtime.platform, "system", lambda: "Windows")
    assert essentia_runtime.feature_runtime(settings).probe_timeout_seconds == 240
    monkeypatch.setattr(essentia_runtime.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(essentia_runtime.importlib.util, "find_spec", lambda _: None)
    worker = essentia_runtime.feature_runtime(settings)
    assert worker.expected_runtime == "essentia-python"
    assert worker.probe_timeout_seconds == 240
    assert MoodAnalyzer(settings).native.probe_timeout_seconds == 240
