"""HOMR 0.7 CPU adapter. Runs in its own Python/NumPy environment."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

PREFIX = "AUDIO_TOOLKIT_OMR "


def emit(**event):
    print(PREFIX + json.dumps(event, ensure_ascii=True), flush=True)


def merge_pages(paths, output):
    roots = [ET.parse(path).getroot() for path in paths]
    if any(root.tag != "score-partwise" for root in roots):
        raise ValueError("HOMR did not produce partwise MusicXML")
    root = roots[0]
    parts = root.findall("part")
    if not parts:
        raise ValueError("No score parts were recognized")
    if any(not page.findall(".//note") for page in roots):
        raise ValueError("No score notes were recognized on a page; use a clearer scan")
    for page in roots[1:]:
        next_parts = page.findall("part")
        if len(next_parts) != len(parts):
            raise ValueError("Different part counts across PDF pages; import pages separately and correct missing staves")
        for target, source in zip(parts, next_parts, strict=True):
            for measure in source.findall("measure"):
                target.append(deepcopy(measure))
    for part in parts:
        for index, measure in enumerate(part.findall("measure"), 1):
            measure.set("number", str(index))
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def render_input(source, limit):
    from PIL import Image
    if source.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(source) as pdf:
            if not 1 <= len(pdf) <= limit:
                raise ValueError(f"PDF must contain 1–{limit} pages; split longer scores first")
            paths = []
            for index in range(len(pdf)):
                page = pdf[index]
                try:
                    width, height = page.get_size()
                    scale = min(200 / 72, (20000000 / max(1, width * height)) ** .5)
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil().convert("RGB")
                        path = source.with_name(f"page-{index + 1:04}.png")
                        image.save(path); image.close(); paths.append(path)
                    finally:
                        bitmap.close()
                finally:
                    page.close()
            return paths
    with Image.open(source) as image:
        if image.width * image.height > 40000000:
            raise ValueError("Score image exceeds 40 million pixels; resize it first")
        image.load()
        path = source.with_name("page-0001.png")
        image.convert("RGB").save(path)
    return [path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    source, model_root = args.source.resolve(), args.models.resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    os.chdir(model_root)
    paths = render_input(source, 20)
    emit(type="progress", progress=.05, message="Score pages prepared")
    # Relocate upstream's model cache before importing its inference modules.
    import homr.segmentation.config as segmentation
    segmentation.segnet_path_onnx = str(model_root / Path(segmentation.segnet_path_onnx).name)
    segmentation.segnet_path_onnx_fp16 = str(model_root / Path(segmentation.segnet_path_onnx_fp16).name)
    import homr.transformer.configs as transformer
    original_paths = transformer.FilePaths.__init__
    def relocate(paths):
        for field in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16"):
            setattr(paths, field, str(model_root / Path(getattr(paths, field)).name))
    def model_paths(paths):
        original_paths(paths)
        relocate(paths)
    transformer.FilePaths.__init__ = model_paths
    relocate(transformer.default_config.filepaths)
    from homr.main import ProcessingConfig, download_weights, process_image
    from homr.music_xml_generator import XmlGeneratorArguments
    download_weights(False, False, False)
    emit(type="progress", progress=.1, message="HOMR models prepared")
    config = ProcessingConfig(False, False, False, False, -1, False, False, False)
    xmls = []
    for index, path in enumerate(paths):
        process_image(str(path), config, XmlGeneratorArguments())
        xml = path.with_suffix(".musicxml")
        if not xml.is_file():
            raise RuntimeError(f"No MusicXML produced for page {index + 1}")
        xmls.append(xml)
        emit(type="progress", progress=.1 + .85 * (index + 1) / len(paths), message=f"Recognized page {index + 1}/{len(paths)}")
    output = source.with_name("recognized.musicxml")
    merge_pages(xmls, output)
    emit(type="result", pageCount=len(paths))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit(type="error", message=str(error))
        raise SystemExit(1)
