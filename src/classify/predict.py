"""Document type classification via the published TrustDoc classifier (HF Hub)."""
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor

MODEL_ID = "vxa8502/trustdoc-classifier"


def load_classifier():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model.eval()
    return model, processor


def label_names(model) -> list[str]:
    """Label names straight from the model's own config -- id2label is baked in at training time
    (see notebooks/01_train_classifier_rvl_cdip_mini.ipynb), so this can never drift out of sync
    with the model the way a hand-maintained list could."""
    return [model.config.id2label[i] for i in range(model.config.num_labels)]


@torch.no_grad()
def classify(image: Image.Image, words: list[str], boxes: list[list[int]], model, processor) -> torch.Tensor:
    """Raw (uncalibrated) logits, shape (1, num_labels). Apply temperature scaling before use."""
    encoding = processor(
        image, words, boxes=boxes, truncation=True, padding="max_length", return_tensors="pt"
    )
    return model(**encoding).logits
