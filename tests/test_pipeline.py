"""Unit tests for the pipeline's own glue logic (calibration application, threshold flagging).

Model loading/inference is mocked -- that's HF Hub network I/O, verified manually against real
models instead rather than re-tested here, which would make CI slow and dependent on network
availability for no extra correctness signal.
"""
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from src import pipeline
from src.calibrate.temperature_scaling import StaleCalibrationError
from src.pipeline import PipelineStageError

# Labels are now read off the loaded model's config (see src/classify/predict.py:label_names),
# not a static importable constant -- so tests mock label_names() directly, same as classify().
FAKE_LABELS = ["letter", "form", "email", "handwritten"]

# Every test below mocks yaml.safe_load with this instead of letting run() read the real
# configs/calibration.yaml off disk. Without this, these tests silently depend on whatever the
# live production temperature/threshold currently are -- a legitimate recalibration (the
# temperature changed twice already this project) could spuriously break them, and a real bug in
# _load_calibration() couldn't be pinned down against a known value. Nested per-head, matching
# the real config's structure -- classifier and extraction are never allowed to share a
# temperature/threshold.
#
# classifier/extraction deliberately use DIFFERENT temperature and confidence_threshold values
# (not just different model_revision) -- identical values across heads would make a real bug that
# cross-wires classifier_cal/extraction_cal in src/pipeline.py invisible to these tests (confirmed
# via mutation testing).
FAKE_CALIBRATION = {
    "classifier": {"temperature": 1.0, "confidence_threshold": 0.9, "model_revision": "fake-clf-rev"},
    "extraction": {"temperature": 2.0, "confidence_threshold": 0.7, "model_revision": "fake-ext-rev"},
}


def _make_test_image(tmp_path):
    path = tmp_path / "doc.png"
    Image.new("RGB", (100, 100), color="white").save(path)
    return str(path)


def test_run_returns_expected_structure_and_high_confidence_not_flagged(tmp_path):
    fake_logits = torch.zeros(1, len(FAKE_LABELS))
    fake_logits[0, 3] = 10.0  # heavily favors class 3 -> high confidence after softmax
    fake_fields = [{"word": "word", "label": "B-ANSWER", "confidence": 0.95}]

    with patch("src.pipeline.extract_words_boxes", return_value=(["word"], [[0, 0, 10, 10]])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", return_value=fake_fields) as mock_extract_fields, \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION), \
         patch("src.pipeline.check_model_revision"):
        pipeline._models.clear()
        result = pipeline.run(_make_test_image(tmp_path))

    assert result["document_type"] == FAKE_LABELS[3]
    assert result["flagged_for_review"] is False
    # softmax([0, 0, 0, 10]) at T=1.0 -> class 3 gets ~0.9999; a known, precise value now that
    # temperature is controlled, not whatever configs/calibration.yaml currently happens to say.
    assert result["confidence"] > 0.999
    # A field above the extraction threshold (0.7) gets its own flagged_for_review=False, added
    # by run() -- independent of the document-level classification confidence.
    assert result["extracted_fields"] == [
        {"word": "word", "label": "B-ANSWER", "confidence": 0.95, "flagged_for_review": False}
    ]
    # extract_fields() must be temperature-scaled with the extraction head's own temperature, not
    # the classifier's -- the two are 1.0 vs. 2.0 here specifically so a cross-wired bug (passing
    # classifier_cal["temperature"] instead) would be caught (confirmed via mutation testing).
    assert mock_extract_fields.call_args.kwargs["temperature"] == FAKE_CALIBRATION["extraction"]["temperature"]


def test_run_flags_low_confidence_predictions_for_both_classification_and_extraction(tmp_path):
    fake_logits = torch.zeros(1, len(FAKE_LABELS))  # uniform -> low confidence
    # One field above, one below the extraction threshold -- proves each field is flagged
    # independently, not by a single document-level decision. "confident_word" is deliberately
    # 0.8 -- between the extraction threshold (0.7, not flagged) and the classifier's (0.9, would
    # be wrongly flagged if should_flag() were cross-wired to the wrong head's threshold) -- so
    # this test is actually sensitive to that bug, not just to a below/above-both-thresholds case.
    fake_fields = [
        {"word": "confident_word", "label": "B-QUESTION", "confidence": 0.8},
        {"word": "unsure_word", "label": "B-ANSWER", "confidence": 0.4},
    ]

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", return_value=fake_fields), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION), \
         patch("src.pipeline.check_model_revision"):
        pipeline._models.clear()
        result = pipeline.run(_make_test_image(tmp_path))

    assert result["flagged_for_review"] is True
    # softmax of 4 equal logits is exactly 1/4, regardless of temperature -- an exact, known
    # value now that this doesn't depend on whatever the real config currently says.
    assert result["confidence"] == 0.25
    assert result["extracted_fields"][0]["flagged_for_review"] is False
    assert result["extracted_fields"][1]["flagged_for_review"] is True


def test_run_caches_models_across_calls(tmp_path):
    fake_logits = torch.zeros(1, len(FAKE_LABELS))

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")) as mock_load_clf, \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS) as mock_label_names, \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")) as mock_load_ext, \
         patch("src.pipeline.extract_fields", return_value=[]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION) as mock_safe_load, \
         patch("src.pipeline.check_model_revision"):
        pipeline._models.clear()
        image_path = _make_test_image(tmp_path)
        pipeline.run(image_path)
        pipeline.run(image_path)

    assert mock_load_clf.call_count == 1  # second run() must reuse the cached model
    assert mock_load_ext.call_count == 1
    assert mock_label_names.call_count == 1  # second run() must reuse the cached label list
    assert mock_safe_load.call_count == 1  # second run() must not re-read/re-parse the config


def test_run_raises_stale_calibration_error_on_classifier_model_revision_mismatch(tmp_path):
    # Reproduces a real incident (a retrained classifier silently drifting from a calibration fit
    # against the previous one) at the pipeline level, not just the unit level -- check_model_revision
    # is deliberately NOT mocked here, so this exercises the real call site in pipeline.run() rather
    # than just the function in isolation (see tests/test_calibrate.py).
    fake_logits = torch.zeros(1, len(FAKE_LABELS))

    class _FakeConfig:
        _commit_hash = "some-other-revision"  # doesn't match FAKE_CALIBRATION["classifier"]["model_revision"]

    class _FakeModel:
        config = _FakeConfig()

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=(_FakeModel(), "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", return_value=[]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION):
        pipeline._models.clear()
        with pytest.raises(StaleCalibrationError, match="some-other-revision"):
            pipeline.run(_make_test_image(tmp_path))


def test_run_raises_stale_calibration_error_on_extraction_model_revision_mismatch(tmp_path):
    # Same failure mode as the classifier test above, but for the extractor's own, independently
    # checked revision -- proves the two checks are genuinely separate, not one shared check that
    # happens to mention two names.
    fake_logits = torch.zeros(1, len(FAKE_LABELS))

    class _FakeConfig:
        _commit_hash = "wrong-extractor-revision"

    class _FakeModel:
        config = _FakeConfig()

    class _MatchingClassifierConfig:
        _commit_hash = FAKE_CALIBRATION["classifier"]["model_revision"]

    class _MatchingClassifierModel:
        config = _MatchingClassifierConfig()

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=(_MatchingClassifierModel(), "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=(_FakeModel(), "processor")), \
         patch("src.pipeline.extract_fields", return_value=[]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION):
        pipeline._models.clear()
        with pytest.raises(StaleCalibrationError, match="wrong-extractor-revision"):
            pipeline.run(_make_test_image(tmp_path))


def test_run_wraps_ocr_failure_in_pipeline_stage_error(tmp_path):
    # An OCR failure (e.g. an unreadable image) must surface as a human-readable, stage-tagged
    # error, not a raw, unlabeled traceback.
    ocr_failure = ValueError("unreadable image")

    with patch("src.pipeline.extract_words_boxes", side_effect=ocr_failure), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION), \
         patch("src.pipeline.check_model_revision"):
        pipeline._models.clear()
        with pytest.raises(PipelineStageError, match="ocr stage failed") as exc_info:
            pipeline.run(_make_test_image(tmp_path))

    assert exc_info.value.stage == "ocr"
    assert exc_info.value.original is ocr_failure
    assert exc_info.value.__cause__ is ocr_failure  # `raise ... from e` chains the real traceback


def test_run_wraps_extract_failure_in_pipeline_stage_error(tmp_path):
    # Same mechanism, a different stage -- proves _run_stage tags whichever stage actually
    # failed, not just the first one wired up.
    fake_logits = torch.zeros(1, len(FAKE_LABELS))
    extract_failure = RuntimeError("model produced malformed output")

    with patch("src.pipeline.extract_words_boxes", return_value=(["word"], [[0, 0, 1, 1]])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", side_effect=extract_failure), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION), \
         patch("src.pipeline.check_model_revision"):
        pipeline._models.clear()
        with pytest.raises(PipelineStageError, match="extract stage failed") as exc_info:
            pipeline.run(_make_test_image(tmp_path))

    assert exc_info.value.stage == "extract"


def test_run_wraps_malformed_calibration_config_in_pipeline_stage_error(tmp_path):
    # A calibration.yaml missing a per-head block (e.g. a hand-edit that drops `extraction:`)
    # must raise a stage-tagged PipelineStageError, not a raw, unlabeled KeyError that would
    # propagate past both `except PipelineStageError` and `except StaleCalibrationError` in
    # src/pipeline.py's __main__ block. Every CLI error must name its failing stage.
    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.yaml.safe_load", return_value={"classifier": FAKE_CALIBRATION["classifier"]}):
        pipeline._models.clear()
        with pytest.raises(PipelineStageError, match="calibrate stage failed") as exc_info:
            pipeline.run(_make_test_image(tmp_path))

    assert exc_info.value.stage == "calibrate"
    assert isinstance(exc_info.value.original, KeyError)


def test_run_wraps_flag_failure_in_pipeline_stage_error(tmp_path):
    # should_flag() is the single canonical flagging rule (src/calibrate/flagging.py); any
    # failure there (e.g. a corrupted threshold) must surface as a stage-tagged
    # PipelineStageError, not an unwrapped exception.
    fake_logits = torch.zeros(1, len(FAKE_LABELS))
    flag_failure = TypeError("'<' not supported between instances of 'float' and 'NoneType'")

    with patch("src.pipeline.extract_words_boxes", return_value=([], [])), \
         patch("src.pipeline.load_classifier", return_value=("model", "processor")), \
         patch("src.pipeline.label_names", return_value=FAKE_LABELS), \
         patch("src.pipeline.classify", return_value=fake_logits), \
         patch("src.pipeline.load_extractor", return_value=("model", "processor")), \
         patch("src.pipeline.extract_fields", return_value=[]), \
         patch("src.pipeline.yaml.safe_load", return_value=FAKE_CALIBRATION), \
         patch("src.pipeline.check_model_revision"), \
         patch("src.pipeline.should_flag", side_effect=flag_failure):
        pipeline._models.clear()
        with pytest.raises(PipelineStageError, match="flag stage failed") as exc_info:
            pipeline.run(_make_test_image(tmp_path))

    assert exc_info.value.stage == "flag"
    assert exc_info.value.original is flag_failure
