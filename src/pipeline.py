"""End-to-end OCR -> classify -> calibrate/flag -> extract pipeline."""
from pathlib import Path

import torch.nn.functional as F
import yaml
from PIL import Image

from src.calibrate.flagging import should_flag
from src.calibrate.temperature_scaling import StaleCalibrationError, check_model_revision
from src.classify.predict import classify, label_names, load_classifier
from src.extract.predict import extract_fields, load_extractor
from src.ocr.tesseract import extract_words_boxes

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "calibration.yaml"

_models = {}  # module-level cache so repeated run() calls don't re-download from the Hub,
              # re-read/re-parse configs/calibration.yaml, or rebuild the label list every time


class PipelineStageError(Exception):
    """Raised when a per-document processing stage (ingest/ocr/classify/calibrate/extract/flag)
    fails, tagging which stage so the caller knows *where* it failed -- a flatter alternative to
    a full typed exception hierarchy (one class per stage), appropriate for a single-document
    pipeline that only needs failure *location*, not failure *type*."""

    def __init__(self, stage: str, original: Exception):
        super().__init__(f"{stage} stage failed ({type(original).__name__}): {original}")
        self.stage = stage
        self.original = original


def _run_stage(stage: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        raise PipelineStageError(stage, e) from e


def _load_models():
    if "classifier" not in _models:
        _models["classifier"] = load_classifier()
    if "extractor" not in _models:
        _models["extractor"] = load_extractor()
    return _models["classifier"], _models["extractor"]


def _load_calibration():
    if "calibration" not in _models:
        with open(CONFIG_PATH) as f:
            _models["calibration"] = yaml.safe_load(f)
    return _models["calibration"]


def run(document_path: str) -> dict:
    def _load_calibration_heads():
        calibration = _load_calibration()
        return calibration["classifier"], calibration["extraction"]

    # Config loading/shape is wrapped the same as any other stage: a malformed calibration.yaml,
    # e.g. a missing `extraction:` block, should raise a stage-tagged PipelineStageError instead
    # of a raw KeyError.
    classifier_cal, extraction_cal = _run_stage("calibrate", _load_calibration_heads)

    (classifier_model, classifier_processor), (extractor_model, extractor_processor) = _load_models()
    check_model_revision(classifier_model, classifier_cal["model_revision"], "vxa8502/trustdoc-classifier")
    check_model_revision(extractor_model, extraction_cal["model_revision"], "vxa8502/trustdoc-extractor")
    if "labels" not in _models:
        _models["labels"] = label_names(classifier_model)
    labels = _models["labels"]

    image = _run_stage("ingest", lambda: Image.open(document_path).convert("RGB"))
    words, boxes = _run_stage("ocr", extract_words_boxes, image)

    logits = _run_stage("classify", classify, image, words, boxes, classifier_model, classifier_processor)
    # one reduction instead of separate argmax()+max()
    confidence, predicted_idx = _run_stage(
        "calibrate", lambda: F.softmax(logits / classifier_cal["temperature"], dim=-1)[0].max(dim=-1)
    )

    # Each extracted field carries its own calibrated confidence and its own flag decision --
    # independent of the document-level classification confidence. Previously the extractor had
    # no confidence signal at all.
    fields = _run_stage(
        "extract", extract_fields,
        image, words, boxes, extractor_model, extractor_processor, temperature=extraction_cal["temperature"],
    )

    def _flag_fields():
        for field in fields:
            field["flagged_for_review"] = should_flag(field["confidence"], extraction_cal["confidence_threshold"])

    _run_stage("flag", _flag_fields)
    document_flagged = _run_stage("flag", should_flag, float(confidence), classifier_cal["confidence_threshold"])

    return {
        "document_type": labels[int(predicted_idx)],
        "confidence": float(confidence),
        "flagged_for_review": document_flagged,
        "extracted_fields": fields,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m src.pipeline <document_image_path>", file=sys.stderr)
        sys.exit(1)
    try:
        print(json.dumps(run(sys.argv[1]), indent=2))
    except PipelineStageError as e:
        # CLI errors must be human-readable and name the failing stage.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except StaleCalibrationError as e:
        print(f"error: calibration is out of date -- {e}", file=sys.stderr)
        sys.exit(1)
