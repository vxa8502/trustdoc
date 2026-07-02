"""Unit tests for the pipeline's own glue logic (calibration application, threshold flagging).

Model loading/inference is mocked -- that's HF Hub network I/O, verified manually against real
models instead rather than re-tested here, which would make CI slow and dependent on network
availability for no extra correctness signal.
"""
from unittest.mock import patch

import torch
from PIL import Image

from src import pipeline

# Labels are now read off the loaded model's config (see src/classify/predict.py:label_names),
# not a static importable constant -- so tests mock label_names() directly, same as classify().
FAKE_LABELS = ["letter", "form", "email", "handwritten"]

# Every test below mocks yaml.safe_load with this instead of letting run() read the real
# configs/calibration.yaml off disk. Without this, these tests silently depend on whatever the
# live production temperature/threshold currently are -- a legitimate recalibration (the
# temperature changed twice already this project) could spuriously break them, and a real bug in
# _load_calibration() couldn't be pinned down against a known value.
FAKE_CALIBRATION = {"temperature": 1.0, "confidence_threshold": 0.9}


def _make_test_image(tmp_path):
    path = tmp_path / "doc.png"
    Image.new("RGB", (100, 100), color="white").save(path)
    return str(path)


def test_run_returns_expected_structure_and_high_confidence_not_flagged(tmp_path):
    fake_logits = torch.zeros(1, len(FAKE_LABELS))
    fake_logits[0, 3] = 10.0  # heavily favors class 3 -> high confidence after softmax

    with patch("src.pipeline.extract_words_boxes", return_value=(["word"], [[0, 0, 10, 10]])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", return_value=[{"word": "word", "label": "B-ANSWER"}]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION):
        pipeline._models.clear()
        result = pipeline.run(_make_test_image(tmp_path))

    assert result["document_type"] == FAKE_LABELS[3]
    assert result["extracted_fields"] == [{"word": "word", "label": "B-ANSWER"}]
    assert result["flagged_for_review"] is False
    # softmax([0, 0, 0, 10]) at T=1.0 -> class 3 gets ~0.9999; a known, precise value now that
    # temperature is controlled, not whatever configs/calibration.yaml currently happens to say.
    assert result["confidence"] > 0.999


def test_run_flags_low_confidence_predictions(tmp_path):
    fake_logits = torch.zeros(1, len(FAKE_LABELS))  # uniform -> low confidence

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", return_value=[]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION):
        pipeline._models.clear()
        result = pipeline.run(_make_test_image(tmp_path))

    assert result["flagged_for_review"] is True
    # softmax of 4 equal logits is exactly 1/4, regardless of temperature -- an exact, known
    # value now that this doesn't depend on whatever the real config currently says.
    assert result["confidence"] == 0.25


def test_run_caches_models_across_calls(tmp_path):
    fake_logits = torch.zeros(1, len(FAKE_LABELS))

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")) as mock_load_clf, \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS) as mock_label_names, \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")) as mock_load_ext, \
         patch("src.pipeline.extract_fields", return_value=[]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION) as mock_safe_load:
        pipeline._models.clear()
        image_path = _make_test_image(tmp_path)
        pipeline.run(image_path)
        pipeline.run(image_path)

    assert mock_load_clf.call_count == 1  # second run() must reuse the cached model
    assert mock_load_ext.call_count == 1
    assert mock_label_names.call_count == 1  # second run() must reuse the cached label list
    assert mock_safe_load.call_count == 1  # second run() must not re-read/re-parse the config
