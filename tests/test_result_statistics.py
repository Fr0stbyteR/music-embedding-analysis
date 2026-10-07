import base64
import math

import numpy as np
import pytest

from music_annotation_backend.result_statistics import result_statistics, summarize


def test_vector_summaries_mask_missing_values_but_keep_real_zero():
    result = result_statistics({"vectors": [[0, 1000, 6, float("nan")], []],
                                "metadata": {"validity.0": base64.b64encode(bytes([0b0101])).decode(),
                                             "statistics.0.median": 3, "unit": "Hz"}})
    metadata = result["metadata"]
    assert metadata["statistics.version"] == 1
    assert metadata["statistics.0.count"] == 2
    assert metadata["statistics.0.mean"] == 3
    assert metadata["statistics.0.std"] == 3
    assert metadata["statistics.0.rms"] == pytest.approx(math.sqrt(18))
    assert metadata["statistics.0.min"] == 0
    assert metadata["statistics.1.count"] == 0
    assert "statistics.1.mean" not in metadata
    assert metadata["statistics.0.median"] == 3
    assert metadata["unit"] == "Hz"
    assert result_statistics(result) is result


def test_matrix_aggregate_and_bins_have_separate_namespaces_and_validity():
    result = result_statistics({"matrix": [[1, 10], [2, 1000], [3, 30]],
                                "metadata": {"statistics.0.mean": 2,
                                             "validity.1": base64.b64encode(bytes([0b0101])).decode()}})
    metadata = result["metadata"]
    expected = summarize([1, 2, 3, 10, 30])
    for key, value in expected.items():
        assert metadata[f"statistics.matrix.0.{key}"] == pytest.approx(value)
    assert metadata["statistics.matrix.0.bin.0.mean"] == 2
    assert metadata["statistics.matrix.0.bin.1.mean"] == 20
    assert metadata["statistics.matrix.0.bin.1.count"] == 2
    assert metadata["statistics.0.mean"] == 2, "legacy LPCC coefficient summaries are preserved"


def test_no_numeric_result_and_finite_only_statistics():
    markers = {"markers": [{"time": 1}]}
    assert result_statistics(markers) is markers
    assert "metadata" not in markers
    assert summarize([float("nan"), float("inf")]) == {"count": 0}
    assert summarize(np.array([3, 4]))["rms"] == pytest.approx(math.sqrt(12.5))
    with pytest.raises(ValueError, match="validity mask"):
        result_statistics({"vectors": [[0] * 9], "metadata": {"validity.0": "AQ=="}})
