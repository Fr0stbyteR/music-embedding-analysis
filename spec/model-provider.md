# Model provider protocol

Python implementations live behind a stable provider interface so checkpoints
can change without changing frontend or project data.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Literal, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_id: str
    provider_version: str
    sample_rate: int
    supports_frame_embeddings: bool
    supports_text_embeddings: bool
    supports_training: bool
    supports_cpu: bool
    supported_devices: tuple[str, ...]
    weight_license: str
    commercial_use: bool | None


@dataclass(frozen=True)
class AudioBatch:
    waveforms: np.ndarray       # float32 [batch, samples]
    sample_rate: int
    valid_samples: np.ndarray   # int64 [batch]


@dataclass(frozen=True)
class FrameEmbeddings:
    values: np.ndarray          # float16/float32 [batch, frames, dimension]
    frame_start_samples: np.ndarray
    frame_end_samples: np.ndarray
    receptive_field_samples: int


@dataclass(frozen=True)
class PredictionEvent:
    start_sample: int
    end_sample: int
    label_scores: dict[str, float]
    ood_score: float | None


class ModelProvider(Protocol):
    async def load(self, device: str) -> None: ...

    def capabilities(self) -> ProviderCapabilities: ...

    async def encode_audio(
        self,
        batch: AudioBatch,
        *,
        output: Literal["frame", "clip", "both"],
        layer: int | str = "recommended",
    ) -> FrameEmbeddings | np.ndarray | tuple[FrameEmbeddings, np.ndarray]: ...

    async def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...

    async def predict_events(
        self,
        frames: FrameEmbeddings,
        *,
        taxonomy_version: str,
        model_version: str,
    ) -> Sequence[PredictionEvent]: ...

    async def train(
        self,
        snapshot_path: Path,
        output_path: Path,
        config: dict,
    ) -> AsyncIterator[dict]: ...

    async def unload(self) -> None: ...
```

## Contract rules

1. Provider methods never write project metadata directly.
2. Every tensor result includes timing/receptive-field metadata.
3. Resampling is explicit and recorded in the feature manifest.
4. Providers return raw scores; calibration and project thresholds are versioned
   separately.
5. A provider cannot publish its own model version. The model registry publishes
   only after evaluation gates pass.
6. Loading and training must be cancellable at phase boundaries.
7. Model code runs in a worker process when GPU memory leaks or native-library
   crashes could destabilize the API process.
8. Providers expose license metadata before a weight download begins.

## Initial implementations

- `M2DClapProvider`: default temporal feature and training provider.
- `LaionClapProvider`: global zero-shot compatibility baseline.
- `MuQMulanProvider`: optional bilingual research provider.
- `MockProvider`: deterministic CI provider with no large weights.

