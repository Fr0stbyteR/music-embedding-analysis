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
    "laion_clap_music_htsat_base": "laion",
    "muq_mulan_large": "muq",
    "m2d_clap_2025": "m2d",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic", action="store_true", help="Start without an embedding model")
    parser.add_argument("--uv", default="uv", help="uv executable path")
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
        if not settings.auto_load_allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        else:
            print("First model load may download several GB. Wait for 'Application startup complete'.", flush=True)
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
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
