"""Exercise first-run configuration and failure paths without downloading models."""
import importlib.util
from pathlib import Path
import socket
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location("launcher", Path(__file__).parents[1] / "scripts" / "launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "scripts" / "launch.py"))
    monkeypatch.chdir(tmp_path)
    for key in list(launcher.os.environ):
        if key.startswith("MAB_") or key == "HF_HUB_OFFLINE":
            monkeypatch.delenv(key)
    (tmp_path / ".env.example").write_text(
        "MAB_PORT=0\nMAB_AUTO_LOAD_PROVIDER=laion_clap_music_htsat_base\n"
        "MAB_AUTO_LOAD_ALLOW_DOWNLOAD=true\n", encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["launch.py"])
    return tmp_path


def test_first_run_copies_config_and_installs_selected_extra(workspace, monkeypatch):
    installed, started = [], []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: installed.append(command))
    monkeypatch.setattr(subprocess, "call", lambda command, **kwargs: started.append(command) or 0)
    assert launcher.main() == 0
    assert (workspace / ".env").read_bytes() == (workspace / ".env.example").read_bytes()
    assert installed[0][-2:] == ["--extra", "laion"]
    assert "--locked" in installed[0]
    assert started[0][-2:] == ["-m", "music_annotation_backend.main"]


def test_basic_mode_preserves_config_and_skips_model_install(workspace, monkeypatch):
    original = "MAB_PORT=0\nMAB_AUTO_LOAD_PROVIDER=muq_mulan_large\n"
    (workspace / ".env").write_text(original, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["launch.py", "--basic"])
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Model installation in basic mode"))
    child_env = {}
    monkeypatch.setattr(subprocess, "call", lambda command, **kwargs: child_env.update(kwargs["env"]) or 0)
    assert launcher.main() == 0
    assert (workspace / ".env").read_text() == original
    assert launcher.Settings().auto_load_provider == ""
    assert child_env["MAB_AUTO_LOAD_PROVIDER"] == ""


def test_environment_override_and_offline_mode(workspace, monkeypatch):
    monkeypatch.setenv("MAB_AUTO_LOAD_PROVIDER", "muq_mulan_large")
    monkeypatch.setenv("MAB_AUTO_LOAD_ALLOW_DOWNLOAD", "false")
    installed = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: installed.append(command))
    monkeypatch.setattr(subprocess, "call", lambda command, **kwargs: 0)
    assert launcher.main() == 0
    assert installed[0][-1] == "muq"
    assert launcher.os.environ["HF_HUB_OFFLINE"] == "1"


def test_unknown_provider_fails_before_install(workspace, monkeypatch):
    monkeypatch.setenv("MAB_AUTO_LOAD_PROVIDER", "typo")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected install"))
    assert launcher.main() == 1


def test_port_in_use_fails_before_download(workspace, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setenv("MAB_PORT", str(listener.getsockname()[1]))
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected install"))
        assert launcher.main() == 1


def test_failed_install_never_starts_backend(workspace, monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(subprocess, "call", lambda *a, **k: pytest.fail("Started after failed install"))
    with pytest.raises(subprocess.CalledProcessError):
        launcher.main()
