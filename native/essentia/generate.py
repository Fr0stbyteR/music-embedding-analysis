"""Generate a reduced upstream registry; do not modify the vendor checkout."""
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

root, output = map(lambda value: Path(value).resolve(), sys.argv[1:3])
spec = importlib.util.spec_from_file_location("upstream_algorithms", root / "utils/algorithms_info.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
algorithms = module.get_all_algorithms(str(root / "src/algorithms"), str(root / "src"))
pending = ["RMS", "Energy", "Loudness", "ZeroCrossingRate", "Centroid", "RollOff", "Flatness", "Crest", "Flux", "Entropy", "SpectralComplexity", "HFC", "CentralMoments", "DistributionShape", "SpectralPeaks", "Dissonance", "PitchYinFFT", "MelBands", "BarkBands", "ERBBands", "MFCC", "GFCC", "HPCP", "Key", "OnsetDetection", "Onsets", "FrameCutter", "TensorflowInputMusiCNN", "TensorflowPredict"]
pending.append("TensorflowInputTempoCNN")
selected = {}

def write_changed(path, text):
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")

def standard_only(text):
    # Tokenize strings/comments too, so braces inside docs do not affect nesting.
    tokens = list(re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|namespace\s+streaming\s*\{|[{}]', text))
    ranges = []
    depth = 0
    start = None
    for token in tokens:
        value = token.group()
        if start is None and value.startswith("namespace streaming"):
            start, depth = token.start(), 1
        elif start is not None and value in ("{", "}"):
            depth += 1 if value == "{" else -1
            if depth == 0:
                ranges.append((start, token.end()))
                start = None
    if start is not None:
        raise ValueError("Unclosed streaming namespace")
    for start, end in reversed(ranges):
        text = text[:start] + text[end:]
    return text

while pending:
    name = pending.pop()
    if name == "FFT":
        name = "FFTK"
    if name in selected:
        continue
    info = algorithms[name]
    if not info["has_standard"]:
        raise ValueError(f"Not a standard algorithm: {name}")
    selected[name] = info
    for kind in ("header", "source"):
        source = standard_only((root / "src" / info[kind]).read_text(encoding="utf-8"))
        target = output / info[kind]
        target.parent.mkdir(parents=True, exist_ok=True)
        write_changed(target, source)
        pending.extend(re.findall(r'\bcreate\(\s*"([A-Za-z0-9]+)"', source))
sha = sys.argv[3] if len(sys.argv) > 3 else subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
with TemporaryDirectory(dir=output) as temporary:
    registry = Path(temporary) / "registry.cpp"
    version = Path(temporary) / "version.h"
    module.create_registration_cpp(selected, str(registry), use_streaming=False)
    write_changed(output / "registry.cpp", registry.read_text() + "\nnamespace essentia { namespace streaming { ESSENTIA_API void registerAlgorithm() {} }}\n")
    module.create_version_h(str(version), (root / "VERSION").read_text().strip(), sha)
    write_changed(output / "version.h", version.read_text())
sources = sorted({(output / info["source"]).as_posix() for info in selected.values()})
write_changed(output / "algorithms.cmake", "set(ALGORITHM_SOURCES\n" + "\n".join(f'  "{path}"' for path in sources) + "\n)\n")
print("Native algorithm subset:", ", ".join(sorted(selected)))

# The old explicit singleton specializations are rejected by modern Clang on
# Windows after dllexport has instantiated the factory class. Defining the
# primary template has identical null initialization and is standard C++.
core = (root / "src/essentia/essentia.cpp").read_text(encoding="utf-8")
old = "template<> standard::AlgorithmFactory* standard::AlgorithmFactory::_instance = 0;\ntemplate<> streaming::AlgorithmFactory* streaming::AlgorithmFactory::_instance = 0;"
if old not in core:
    raise ValueError("Unexpected upstream factory singleton definitions")
write_changed(output / "essentia.cpp", core.replace(old, "template<typename BaseAlgorithm> EssentiaFactory<BaseAlgorithm>* EssentiaFactory<BaseAlgorithm>::_instance = 0;"))
