from __future__ import annotations

import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from music_annotation_backend.librosa_engine import analyze


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.audio_path = Path(cls.temp_dir.name) / "clicks.wav"
        sample_rate = 22050
        duration = 4
        samples: list[int] = []
        for index in range(sample_rate * duration):
            phase_in_beat = index % (sample_rate // 2)
            envelope = max(0.0, 1.0 - phase_in_beat / 800.0)
            sample = 0.7 * envelope * math.sin(2 * math.pi * 880 * index / sample_rate)
            samples.append(round(sample * 32767))
        with wave.open(str(cls.audio_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(struct.pack(f"<{len(samples)}h", *samples))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp_dir.cleanup()

    def request(self, algorithm: str, **options: object) -> dict[str, object]:
        return analyze({"path": str(self.audio_path), "algorithm": algorithm, "options": options})

    def test_beats(self) -> None:
        result = self.request("beats", hopLength=256, startBpm=120)
        self.assertGreater(len(result["values"]), 2)  # type: ignore[arg-type]
        self.assertGreater(result["metadata"]["tempo"], 90)  # type: ignore[index,operator]

    def test_onsets(self) -> None:
        result = self.request("onsets", hopLength=256)
        self.assertGreater(len(result["values"]), 2)  # type: ignore[arg-type]

    def test_non_silent_regions(self) -> None:
        result = self.request("nonSilent", topDb=40)
        self.assertTrue(result["intervals"])

    def test_vector_features(self) -> None:
        for algorithm in (
            "rms", "spectralCentroid", "spectralBandwidth", "spectralRolloff",
            "spectralFlatness", "zeroCrossingRate", "onsetStrength", "pitch"
        ):
            with self.subTest(algorithm=algorithm):
                result = self.request(algorithm, frameLength=1024, hopLength=256)
                self.assertTrue(result["vectors"])
                self.assertTrue(result["vectors"][0])  # type: ignore[index]

    def test_matrix_features(self) -> None:
        for algorithm in ("melSpectrogram", "chroma", "mfcc"):
            with self.subTest(algorithm=algorithm):
                result = self.request(algorithm, frameLength=1024, hopLength=256, melBins=32)
                self.assertTrue(result["matrix"])
                self.assertGreater(len(result["matrix"][0]), 1)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()

