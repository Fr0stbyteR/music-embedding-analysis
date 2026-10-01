"""Run the same Essentia preparation used by start scripts, without starting HTTP."""
import argparse
from pathlib import Path
import os

from music_annotation_backend.config import Settings
from music_annotation_backend.essentia_setup import ensure_essentia

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--uv", default="uv")
parser.add_argument("--basic", action="store_true")
parser.add_argument("--force-portable-build", action="store_true", help="Rebuild with the project-local compiler, even if a developer worker exists")
arguments = parser.parse_args()
root = Path(__file__).resolve().parent.parent
os.chdir(root)
ensure_essentia(Settings(), root, arguments.uv, basic=arguments.basic, force_build=arguments.force_portable_build)
