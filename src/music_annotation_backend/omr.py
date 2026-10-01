"""Authenticated, bounded OMR jobs; the model does not share the audio venv."""
import asyncio
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
from uuid import uuid4

VERSION = "homr-0.7.0-adapter-v1"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".webp", ".tif", ".tiff"}


class OmrService:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path.cwd()
        self.lock = asyncio.Lock()

    @property
    def python(self):
        return self.settings.omr_python or self.root / ".tools/homr" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def capabilities(self):
        supported = not (platform.system() == "Darwin" and platform.machine().lower() in {"x86_64", "amd64"})
        return {"engine": "homr", "version": "0.7.0", "installed": self.python.is_file(), "autoInstall": self.settings.omr_auto_install,
            "available": self.python.is_file() or self.settings.omr_auto_install and supported, "extensions": sorted(EXTENSIONS), "maximumBytes": 50 * 1024 * 1024, "maximumPages": 20, "license": "AGPL-3.0",
            "reason": None if supported else "HOMR 0.7 needs ONNX Runtime 1.24+, which has no macOS Intel wheel; use a remote Windows/Linux/Apple Silicon backend or MAB_OMR_PYTHON with a compatible custom build"}

    async def prepare(self, progress):
        marker = self.root / ".tools/homr/ready.json"
        if self.python.is_file() and (self.settings.omr_python is not None or marker.is_file() and marker.read_text() == VERSION):
            probe = await asyncio.create_subprocess_exec(str(self.python), "-c", "import homr, onnxruntime, cv2, PIL, pypdfium2", stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                await asyncio.wait_for(probe.wait(), 30)
                if probe.returncode == 0:
                    return
            finally:
                if probe.returncode is None:
                    probe.kill(); await probe.wait()
            if self.settings.omr_python is not None:
                raise RuntimeError("Configured HOMR environment is missing dependencies (homr, onnxruntime, Pillow, OpenCV, pypdfium2)")
        if self.settings.omr_python is not None or not self.settings.omr_auto_install:
            raise RuntimeError("HOMR is not installed; enable MAB_OMR_AUTO_INSTALL or configure MAB_OMR_PYTHON")
        if not self.capabilities()["available"]:
            raise RuntimeError(self.capabilities()["reason"])
        uv = shutil.which("uv") or str(self.root / ".tools/uv" / ("uv.exe" if os.name == "nt" else "uv"))
        progress("omr-setup", .01, "Preparing isolated HOMR environment")
        commands = []
        if not self.python.is_file():
            commands.append([uv, "venv", "--python", "3.12", str(self.root / ".tools/homr")])
        commands.append([uv, "pip", "install", "--python", str(self.python), "homr==0.7.0", "pypdfium2>=4,<6", "numpy==2.4.2", "onnxruntime==1.24.1", "opencv-python==4.13.0.92", "opencv-python-headless==4.13.0.92"])
        logs = self.settings.data_root / "omr-logs"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "install.log").open("wb") as log:
            for command in commands:
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log)
                try:
                    await asyncio.wait_for(process.wait(), 900)
                    if process.returncode:
                        raise RuntimeError("HOMR installation failed; see backend omr-logs/install.log (requires internet and compatible ONNX CPU wheels)")
                finally:
                    if process.returncode is None:
                        process.kill(); await process.wait()
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(VERSION)

    async def recognize(self, source, name, progress):
        async with self.lock:
            await self.prepare(progress)
            digest = hashlib.sha256(source.read_bytes() + VERSION.encode() + Path(__file__).with_name("omr_worker.py").read_bytes()).hexdigest()
            cache = self.settings.data_root / "omr-cache"
            cache.mkdir(parents=True, exist_ok=True)
            saved = cache / f"{digest}.json"
            if saved.is_file():
                return {**json.loads(saved.read_text(encoding="utf-8")), "name": Path(name).stem + ".musicxml", "cached": True}
            folder = source.parent
            stderr = folder / "homr.log"
            with stderr.open("wb") as log:
                process = await asyncio.create_subprocess_exec(str(self.python), "-B", str(Path(__file__).with_name("omr_worker.py")), str(source.resolve()), "--models", str((self.settings.model_root / "homr").resolve()), stdout=asyncio.subprocess.PIPE, stderr=log)
                async def read_events():
                    pages, error = 0, None
                    async for raw in process.stdout:
                        text = raw.decode("utf-8", errors="replace")
                        if not text.startswith("AUDIO_TOOLKIT_OMR "): continue
                        event = json.loads(text.removeprefix("AUDIO_TOOLKIT_OMR "))
                        if event["type"] == "progress": progress("omr", event["progress"], event["message"])
                        elif event["type"] == "result": pages = event["pageCount"]
                        elif event["type"] == "error": error = event["message"]
                    await process.wait()
                    if process.returncode or not pages:
                        raise RuntimeError(error or f"HOMR exited with {process.returncode}; see {stderr}")
                    return pages
                try:
                    pages = await asyncio.wait_for(read_events(), self.settings.omr_timeout_seconds)
                except TimeoutError as error:
                    raise RuntimeError("HOMR recognition timed out; try fewer pages or increase MAB_OMR_TIMEOUT_SECONDS") from error
                finally:
                    if process.returncode is None:
                        process.kill(); await process.wait()
            xml = folder / "recognized.musicxml"
            if not xml.is_file() or xml.stat().st_size > 10 * 1024 * 1024:
                raise RuntimeError("HOMR produced missing or oversized MusicXML")
            result = {"musicxml": xml.read_text(encoding="utf-8"), "name": Path(name).stem + ".musicxml", "pageCount": pages, "cached": False,
                "warnings": ["OMR is an estimate: review pitches, rhythm, voices, key signatures and page joins before DTW alignment."]}
            temporary = saved.with_name(f"{digest}.{uuid4().hex}.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8"); temporary.replace(saved)
            return result
