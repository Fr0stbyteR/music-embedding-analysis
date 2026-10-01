"""Configure first-run startup after uv has installed the base dependencies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from music_annotation_backend.config import Settings


PROVIDER_EXTRAS = {
    "clap_music": "clap",
    "laion_clap_music_htsat_base": "laion",
    "muq_mulan_large": "muq",
    "m2d_clap_2025": "m2d",
}


def prepare_essentia(*args, **kwargs):
    # Import after uv sync: optional provider installation can update NumPy.
    from music_annotation_backend.essentia_setup import ensure_essentia
    return ensure_essentia(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic", action="store_true", help="Start without an embedding model")
    parser.add_argument("--uv", default="uv", help="uv executable path")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and verify dependencies without starting HTTP")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    env_file = root / ".env"
    if not env_file.exists():
        shutil.copyfile(root / ".env.example", env_file)
        print("Created .env from .env.example; future launches keep your settings.", flush=True)
    if args.basic:
        os.environ["MAB_AUTO_LOAD_PROVIDER"] = ""
    settings = Settings()
    provider = settings.auto_load_provider
    with socket.socket() as probe:
        try:
            probe.bind((settings.host, settings.port))
        except OSError as exc:
            print(f"Cannot listen on {settings.host}:{settings.port}: {exc}. "
                  "Stop the other service or change MAB_PORT in .env.", file=sys.stderr)
            return 1
    if provider and provider != "mock":
        extra = PROVIDER_EXTRAS.get(provider)
        if extra is None:
            print(f"Unknown auto-load provider: {provider}. Check .env.", file=sys.stderr)
            return 1
        print(f"Installing dependencies for {provider}...", flush=True)
        subprocess.run([args.uv, "sync", "--locked", "--python", "3.11", "--inexact", "--extra", extra], check=True)
        if provider == "clap_music":
            print(f"Music CLAP download source: {settings.model_download_source}", flush=True)
        elif provider in {"muq_mulan_large", "laion_clap_music_htsat_base"}:
            print("This legacy provider uses Hugging Face, including nested encoders. "
                  "For ModelScope downloads select MAB_AUTO_LOAD_PROVIDER=clap_music.", flush=True)
        if not settings.auto_load_allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        else:
            print("First model load may download several GB. Wait for 'Application startup complete'.", flush=True)
    prepare_essentia(settings, root, args.uv, basic=args.basic)
    if args.prepare_only:
        print("Backend dependencies and Essentia are ready.", flush=True)
        return 0
    print(f"API documentation: http://{settings.host}:{settings.port}/docs", flush=True)
    print("Authorize API requests with the token in the upcoming JSON handshake. Stop with Ctrl+C.", flush=True)
    python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    # Windows putenv removes empty values; an explicit environment preserves the
    # empty provider override used by --basic in the child process.
    return subprocess.call([str(python), "-m", "music_annotation_backend.main"], env=dict(os.environ))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
