"""End-to-end OCR -> classify -> calibrate/flag -> extract pipeline."""
from pathlib import Path

import torch.nn.functional as F
import yaml
from PIL import Image

from src.calibrate.flagging import should_flag
from src.classify.predict import classify, label_names, load_classifier
from src.extract.predict import extract_fields, load_extractor
from src.ocr.tesseract import extract_words_boxes

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "calibration.yaml"

_models = {}  # module-level cache so repeated run() calls don't re-download from the Hub,
              # re-read/re-parse configs/calibration.yaml, or rebuild the label list every time


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
    calibration = _load_calibration()
    temperature = calibration["temperature"]
    threshold = calibration["confidence_threshold"]

    (classifier_model, classifier_processor), (extractor_model, extractor_processor) = _load_models()
    if "labels" not in _models:
        _models["labels"] = label_names(classifier_model)
    labels = _models["labels"]

    image = Image.open(document_path).convert("RGB")
    words, boxes = extract_words_boxes(image)

    logits = classify(image, words, boxes, classifier_model, classifier_processor)
    calibrated_probs = F.softmax(logits / temperature, dim=-1)[0]
    confidence, predicted_idx = calibrated_probs.max(dim=-1)  # one reduction instead of separate argmax()+max()

    fields = extract_fields(image, words, boxes, extractor_model, extractor_processor)

    return {
        "document_type": labels[int(predicted_idx)],
        "confidence": float(confidence),
        "flagged_for_review": should_flag(float(confidence), threshold),
        "extracted_fields": fields,
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m src.pipeline <document_image_path>", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(run(sys.argv[1]), indent=2))
