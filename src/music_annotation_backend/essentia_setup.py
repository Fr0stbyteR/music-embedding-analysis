"""Project-local, checksum-verified first-run setup. Never installs system toolchains."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryFile, TemporaryDirectory
import time
from urllib.request import urlopen
from uuid import uuid4
import zipfile

import numpy as np

from .essentia_api import ALGORITHMS, EssentiaAnalyzer, EssentiaOptions
from .essentia_runtime import feature_runtime
from .mood import MoodAnalyzer

REVISION = "7320015a1cad3ac1dc038b52ef94803587d09986"


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    sha256: str


SOURCE = Download("essentia-7320015.tar.gz", f"https://codeload.github.com/MTG/essentia/tar.gz/{REVISION}", "bd087a181f1ffedbae318eea458566ce1e63c7012162a785d3c3aa7bcad18af5")
EIGEN = Download("eigen-3.4.0.tar.gz", "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz", "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")
LLVM = Download("llvm-mingw-20260922-ucrt-x86_64.zip", "https://github.com/mstorsjo/llvm-mingw/releases/download/20260922/llvm-mingw-20260922-ucrt-x86_64.zip", "e3ad77d117a4bea19a7a3b333341824d79a5a371004a10e25b8504e7b3047666")
TENSORFLOW = Download("libtensorflow-2.18.1.zip", "https://storage.googleapis.com/tensorflow/versions/2.18.1/libtensorflow-cpu-windows-x86_64.zip", "28acdcea6c6b34828cf0e95e67802b0f3577d51bc2e8915de811b7aa0b04452d")
VC_RUNTIME = Download("Microsoft.VCLibs.x64.14.00.Desktop.appx", "https://download.microsoft.com/download/4/7/c/47c6134b-d61f-4024-83bd-b9c9ea951c25/Microsoft.VCLibs.x64.14.00.Desktop.appx", "b56a9101f706f9d95f815f5b7fa6efbac972e86573d378b96a07cff5540c5961")
MAC_WHEELS = {
    "x86_64": Download("essentia_tensorflow-2.1b6.dev1110-cp311-cp311-macosx_10_9_x86_64.whl", "https://files.pythonhosted.org/packages/25/02/28409a08eb938e212a29c3fdbc6638e7ed1fb54c8ebbb00e4c5981c788d3/essentia_tensorflow-2.1b6.dev1110-cp311-cp311-macosx_10_9_x86_64.whl", "23107204efda0bec2d2ad52db0faffbbe2b80cd00d9e93c79badbbe70d2ddee2"),
    "arm64": Download("essentia_tensorflow-2.1b6.dev1110-cp311-cp311-macosx_11_0_arm64.whl", "https://files.pythonhosted.org/packages/5f/bc/8ab3c74f700ed243833663cef1bad9341a1d7321d983db6fbeb51ec5ec75/essentia_tensorflow-2.1b6.dev1110-cp311-cp311-macosx_11_0_arm64.whl", "e0a24d41af205e2c3af866a347b987a39b43a497e6bb6ca771a8d8e81048d58f"),
}
MODELS = [
    Download("msd-musicnn-1.pb", "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb", "cdea0722bcee7f731286843f2233e3aa69887bb5c3e2dce011eff55f38d04f3e"),
    Download("deam-msd-musicnn-2.pb", "https://essentia.upf.edu/models/classification-heads/deam/deam-msd-musicnn-2.pb", "beb5eeb0909266eeb78b8d6bb1323b10829cf2fe55e3c01a13fa1846fa98b371"),
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(entry, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / entry.name
    if target.is_file():
        if digest(target) != entry.sha256:
            raise RuntimeError(f"Checksum mismatch: {target}. Move this file aside and retry; it was not overwritten.")
        return target
    print(f"Downloading {entry.name} from {entry.url.split('/')[2]}...", flush=True)
    partial = target.with_name(f"{target.name}.{uuid4().hex}.partial")
    try:
        with urlopen(entry.url, timeout=60) as response, partial.open("wb") as output:
            total, last = 0, time.monotonic()
            while chunk := response.read(1024 * 1024):
                output.write(chunk); total += len(chunk)
                if time.monotonic() - last > 10:
                    print(f"  {entry.name}: {total / 1048576:.0f} MiB downloaded", flush=True)
                    last = time.monotonic()
        if digest(partial) != entry.sha256:
            raise RuntimeError(f"Checksum mismatch for {entry.name}; refusing to use the download")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    boundary = destination.resolve()
    def check(name):
        if not (boundary / name).resolve().is_relative_to(boundary):
            raise RuntimeError(f"Unsafe archive path: {name}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            for name in package.namelist():
                check(name)
            package.extractall(destination)
    else:
        with tarfile.open(archive) as package:
            for member in package.getmembers():
                check(member.name)
            package.extractall(destination, filter="data")


def source_fingerprint(root):
    return hashlib.sha256(b"".join((root / "native/essentia" / name).read_bytes() for name in ("worker.cpp", "features_native.h", "mood_native.h", "tensorflow_native.h", "generate.py", "CMakeLists.txt"))).hexdigest()


def build_windows(root, uv):
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Automatic Windows Essentia setup requires x64 Python; use the x64 start launcher.")
    print("Preparing project-local Essentia C++ toolchain (no Visual Studio/admin installation)...", flush=True)
    cache, tools, vendor = root / "vendor/downloads", root / ".tools", root / "vendor"
    compiler = tools / LLVM.name.removesuffix(".zip")
    if not (compiler / "bin/clang++.exe").is_file():
        extract(download(LLVM, cache), tools)
    source = vendor / f"essentia-{REVISION}"
    if not (source / "src/essentia/essentia.h").is_file():
        extract(download(SOURCE, cache), vendor)
    if not (vendor / "eigen-3.4.0/Eigen/Core").is_file():
        extract(download(EIGEN, cache), vendor)
    tensorflow = vendor / "libtensorflow"
    if not (tensorflow / "lib/tensorflow.dll").is_file():
        extract(download(TENSORFLOW, cache), tensorflow)
    scripts = Path(sys.executable).parent
    subprocess.run([uv, "pip", "install", "--python", sys.executable, "--only-binary", ":all:", "cmake==4.1.3", "ninja==1.13.0"], check=True)
    cmake, ninja = scripts / "cmake.exe", scripts / "ninja.exe"
    build = root / "build/essentia-portable"
    env = {**os.environ, "PATH": str(compiler / "bin") + os.pathsep + str(scripts) + os.pathsep + os.environ.get("PATH", "")}
    arguments = [str(cmake), "-S", str(root / "native/essentia"), "-B", str(build), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_MAKE_PROGRAM={ninja.as_posix()}",
        f"-DCMAKE_C_COMPILER={(compiler / 'bin/clang.exe').as_posix()}", f"-DCMAKE_CXX_COMPILER={(compiler / 'bin/clang++.exe').as_posix()}",
        f"-DPython3_EXECUTABLE={Path(sys.executable).as_posix()}", f"-DESSENTIA_ROOT={source.as_posix()}",
        f"-DESSENTIA_REVISION={REVISION}", f"-DEIGEN_ROOT={(vendor / 'eigen-3.4.0').as_posix()}",
        f"-DTENSORFLOW_ROOT={tensorflow.as_posix()}", f"-DTENSORFLOW_LIBRARY={(tensorflow / 'lib/tensorflow.lib').as_posix()}"]
    subprocess.run(arguments, check=True, env=env)
    print("Building native Essentia; first launch can take several minutes...", flush=True)
    subprocess.run([str(cmake), "--build", str(build), "--parallel", "4"], check=True, env=env)
    installed = tools / "essentia-runtime"
    installed.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "essentia-worker.exe", installed / "essentia-worker.exe")
    shutil.copy2(tensorflow / "lib/tensorflow.dll", installed / "tensorflow.dll")
    # App-local official MSVC runtime required by the TensorFlow DLL, not a
    # system-wide redist installation. The worker itself statically links libc++.
    with zipfile.ZipFile(download(VC_RUNTIME, cache)) as runtime:
        for name in runtime.namelist():
            if '/' not in name and name.lower().endswith('.dll') and not name.lower().startswith('mfc'):
                (installed / name).write_bytes(runtime.read(name))
    (installed / "build.json").write_text(json.dumps({"sources": source_fingerprint(root), "compiler": LLVM.sha256}), encoding="utf-8")


def install_mac(root, uv):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The macOS start script must use Python 3.11 for the pinned Essentia wheel")
    machine = platform.machine().lower()
    if machine not in MAC_WHEELS:
        raise RuntimeError(f"Unsupported macOS Python architecture: {machine}")
    wheel = download(MAC_WHEELS[machine], root / "vendor/downloads")
    # Avoid two distributions owning the same 'essentia' package directory.
    subprocess.run([uv, "pip", "uninstall", "--python", sys.executable, "essentia", "essentia-tensorflow"], check=True)
    subprocess.run([uv, "pip", "install", "--python", sys.executable, "--only-binary", ":all:", "numpy>=1.26,<2", str(wheel)], check=True)


def verify_features(runtime):
    options = EssentiaOptions()
    rate = options.sample_rate
    audio = (.2 * np.sin(2 * np.pi * 440 * np.arange(rate * 3) / rate)).astype(np.float32)
    audio[rate:rate * 2] = 0
    for index, algorithm in enumerate(sorted(ALGORITHMS), 1):
        print(f"Essentia first-run check {index}/{len(ALGORITHMS)}: {algorithm}", flush=True)
        with TemporaryFile() as pcm:
            audio.astype("<f4").tofile(pcm); pcm.seek(0)
            result = runtime._run(["--features", algorithm, rate, options.frame_length, options.hop_length, options.bands, options.coefficients, options.roll_percent, options.threshold_db, options.minimum_duration, options.confidence, options.key_window, options.tuning], stdin=pcm, timeout=120)
        EssentiaAnalyzer.validate_result(result, algorithm, options, 3)


def verify_tensorflow(settings):
    from types import SimpleNamespace
    import soundfile as sf
    from .essentia_tf import ALGORITHMS as TF_ALGORITHMS, TensorflowAnalyzer
    from .schemas import InteractiveLibrosaRequest
    with TemporaryDirectory() as folder:
        directory = Path(folder)
        audio = (.1 * np.sin(2 * np.pi * 440 * np.arange(16000 * 3) / 16000)).astype(np.float32)
        path = directory / "self-test.wav"; sf.write(path, audio, 16000)
        analyzer = TensorflowAnalyzer(settings); analyzer.cache_root = directory / "cache"
        asset = SimpleNamespace(path=str(path), duration_seconds=3, content_hash="tf-startup-test-v1")
        for index, algorithm in enumerate(sorted(TF_ALGORITHMS), 1):
            print(f"Essentia TensorFlow first-run check {index}/{len(TF_ALGORITHMS)}: {algorithm}", flush=True)
            analyzer.analyze(asset, InteractiveLibrosaRequest(algorithm=algorithm))

def ensure_essentia(settings, root, uv, *, basic=False, force_build=False):
    if settings.essentia_setup == "off":
        print("Essentia automatic setup disabled by MAB_ESSENTIA_SETUP=off.", flush=True)
        return
    system = platform.system()
    if system not in {"Windows", "Darwin"}:
        raise RuntimeError("Automatic Essentia setup supports Windows x64 and macOS Intel/Apple Silicon")
    with_mood = settings.essentia_setup_mood and not basic
    with_tensorflow = settings.essentia_setup_tensorflow and not basic
    runtime = feature_runtime(settings)
    ready = False
    try:
        capability = runtime.probe()
        ready = ALGORITHMS.issubset(capability.get("features", [])) and (not with_mood or bool(capability.get("tensorflow")))
        ready = ready and (not with_tensorflow or capability.get("tensorflowFeatures") == 1)
    except RuntimeError:
        pass
    installed = root / ".tools/essentia-runtime"
    if system == "Windows" and runtime.executable.parent == installed.resolve():
        try:
            ready = ready and json.loads((installed / "build.json").read_text())["sources"] == source_fingerprint(root)
        except (OSError, ValueError, KeyError):
            ready = False
    if force_build or not ready:
        if settings.essentia_native_executable is not None and not force_build:
            raise RuntimeError("Configured MAB_ESSENTIA_NATIVE_EXECUTABLE failed its feature check; fix that path or remove the override to install automatically.")
        if system == "Windows":
            build_windows(root, uv)
        else:
            install_mac(root, uv)
        runtime = feature_runtime(settings)
        print(f"Checking installed Essentia capabilities (up to {settings.essentia_probe_timeout_seconds:g}s for cold native import)...", flush=True)
        capability = runtime.probe()
    if not ALGORITHMS.issubset(capability.get("features", [])):
        raise RuntimeError("Essentia runtime is missing one or more of the 29 required analyses")
    if with_mood and not capability.get("tensorflow"):
        raise RuntimeError("Essentia TensorFlow algorithms are missing; VA cannot run with this runtime. Set MAB_ESSENTIA_SETUP_MOOD=false to run DSP only.")
    if with_mood:
        print("Preparing official DEAM/MusiCNN weights (CC BY-NC-SA 4.0, non-commercial).", flush=True)
        for custom, entry in zip((settings.mood_embedding_graph, settings.mood_regression_graph), MODELS):
            if custom is not None:
                if not custom.is_file():
                    raise RuntimeError(f"Configured mood graph is missing: {custom}")
            else:
                download(entry, settings.model_root / "essentia")
    manifest = installed / "ready.json"
    if with_tensorflow:
        from .essentia_tf import MANIFEST
        print("Preparing official Essentia TensorFlow models (check model weight licenses).", flush=True)
        for entry in MANIFEST:
            download(Download(**entry), settings.model_root / "essentia")
        if capability.get("tensorflowFeatures") != 1:
            raise RuntimeError("Native runtime is missing Essentia TensorFlow support")
    identity = {"runtime": capability, "signature": runtime.signature, "setup": digest(Path(__file__)), "adapter": digest(Path(__file__).with_name("essentia_python_worker.py")), "mood": with_mood}
    identity["tensorflow"] = with_tensorflow
    if with_tensorflow:
        identity["tensorflowGraphs"] = [digest(settings.model_root / "essentia" / entry["name"]) for entry in MANIFEST]
        identity["tensorflowAdapter"] = digest(Path(__file__).with_name("essentia_tf_worker.py"))
    if with_mood:
        analyzer = MoodAnalyzer(settings)
        identity["graphs"] = [digest(path) for path in (analyzer.embedding_path, analyzer.regression_path)]
    serialized = json.dumps(identity, sort_keys=True)
    try:
        verified = manifest.read_text(encoding="utf-8") == serialized
    except (OSError, UnicodeError):
        verified = False
    if not verified:
        verify_features(runtime)
        if with_tensorflow:
            verify_tensorflow(settings)
        if with_mood:
            analyzer._load()
            audio = (.1 * np.sin(2 * np.pi * 440 * np.arange(48000) / 16000)).astype(np.float32)
            if analyzer.native_selected:
                analyzer.native.curve(audio, analyzer.embedding_path, analyzer.regression_path, 3, 3, 3)
            else:
                analyzer.predict(audio)
            print("Essentia VA first-run inference passed.", flush=True)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temporary = manifest.with_name(f"ready.{uuid4().hex}.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(manifest)
    print(f"Essentia ready: {len(ALGORITHMS)} feature modules · {capability['runtime']}" + (" · VA model ready" if with_mood else "") + (" · 21 TensorFlow modules ready" if with_tensorflow else ""), flush=True)
