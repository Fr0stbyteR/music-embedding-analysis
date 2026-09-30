"""Download routing must never fall back to another host implicitly."""
import sys
from types import SimpleNamespace

import pytest

from music_annotation_backend.clap_music import MODEL_ID, TransformersClap, resolve_snapshot
from music_annotation_backend.config import Settings
from music_annotation_backend.providers import ProviderRegistry


def settings(tmp_path, **kwargs):
    return Settings(_env_file=None, model_root=tmp_path / "models", **kwargs)


def test_modelscope_is_default_and_offline_is_forwarded(tmp_path, monkeypatch):
    calls = []
    def download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return str(tmp_path / "snapshot")
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=download))
    assert resolve_snapshot(settings(tmp_path), None, False) == tmp_path / "snapshot"
    assert calls[0][0] == MODEL_ID
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["cache_dir"].endswith("modelscope")
    assert "pytorch_model.bin" in calls[0][1]["allow_patterns"]


def test_huggingface_selection_and_cache_override(tmp_path, monkeypatch):
    calls = []
    def download(model_id, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    cfg = settings(tmp_path, model_download_source="huggingface", huggingface_cache=tmp_path / "hf")
    resolve_snapshot(cfg, None, True)
    assert calls[0]["local_files_only"] is False
    assert calls[0]["cache_dir"] == str(tmp_path / "hf")


def test_modelscope_failure_does_not_contact_huggingface(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("selected host unavailable")
    def unexpected(*args, **kwargs):
        pytest.fail("Unexpected Hugging Face fallback")
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=fail))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=unexpected))
    with pytest.raises(OSError, match="selected host"):
        resolve_snapshot(settings(tmp_path), None, True)


def test_explicit_directory_skips_hub_and_legacy_file_is_rejected(tmp_path):
    assert resolve_snapshot(settings(tmp_path), str(tmp_path), False) == tmp_path
    legacy = tmp_path / "model.pt"
    legacy.touch()
    with pytest.raises(FileNotFoundError, match="directory"):
        resolve_snapshot(settings(tmp_path), str(legacy), True)


def test_transformers_loads_only_local_files(tmp_path, monkeypatch):
    calls = []
    class LocalLoader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((path, kwargs))
            return cls()
        def to(self, device):
            return self
        def eval(self):
            return self
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(ClapModel=LocalLoader, ClapProcessor=LocalLoader))
    TransformersClap(tmp_path, "cpu")
    assert len(calls) == 2
    assert all(kwargs == {"local_files_only": True} for _, kwargs in calls)


def test_download_source_validation_and_provider_registration(tmp_path):
    with pytest.raises(ValueError):
        settings(tmp_path, model_download_source="unknown")
    assert ProviderRegistry(settings(tmp_path)).get("clap_music").provider_id == "clap_music"
