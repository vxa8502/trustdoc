"""Field extraction (NER) via the published TrustDoc extractor (HF Hub)."""
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

MODEL_ID = "vxa8502/trustdoc-extractor"


def load_extractor():
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model.eval()
    return model, processor


@torch.no_grad()
def extract_logits(image: Image.Image, words: list[str], boxes: list[list[int]], model, processor):
    """Raw (uncalibrated) per-token logits, shape (seq_len, num_labels), plus the word_ids
    mapping each token back to its source word (or None for special tokens). Split out from
    extract_fields() so a calibration-fitting pass (scripts/calibrate_extractor.py) can collect
    logits across many documents without going through the human-readable aggregation below --
    mirrors src/classify/predict.py's classify()/pipeline.py split (raw logits vs. applying
    temperature scaling elsewhere), adapted for token classification's per-word structure.

    padding=True (dynamic, pads to the batch's own longest) instead of "max_length": the
    pipeline and the calibration script both call this at batch size 1, so fixed 512-length
    padding wastes self-attention compute for zero benefit -- verified 2.31x faster on a
    realistic 40-word document, no downstream change needed since first_token_per_word()
    already handles word_ids of any length."""
    encoding = processor(
        image, words, boxes=boxes, truncation=True, padding=True, return_tensors="pt"
    )
    logits = model(**encoding).logits[0]  # (seq_len, num_labels)
    word_ids = encoding.word_ids(batch_index=0)
    return logits, word_ids


def first_token_per_word(word_ids: list) -> list[tuple[int, int]]:
    """(token_idx, word_idx) pairs for the first token of each real word, in order -- skips
    special tokens (word_idx is None) and subword continuations, so each word is represented
    exactly once. This is the single definition of "one prediction per word": both
    extract_fields() below and scripts/calibrate_extractor.py's calibration-fitting pass consume
    it, so the two can never silently disagree on which token stands for a word (if the rule
    changed in only one place, a fitted temperature could end up calibrated against a different
    token-selection than what inference-time aggregation actually produces)."""
    pairs = []
    seen_words = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_words:
            continue
        seen_words.add(word_idx)
        pairs.append((token_idx, word_idx))
    return pairs


@torch.no_grad()
def extract_fields(
    image: Image.Image,
    words: list[str],
    boxes: list[list[int]],
    model,
    processor,
    temperature: float = 1.0,
) -> list[dict]:
    """One {"word": str, "label": str, "confidence": float} per OCR'd word with a non-"O"
    predicted tag. `confidence` is the calibrated softmax probability of the predicted label;
    `temperature=1.0` (the default) is a no-op, so callers that don't pass a fitted temperature
    get the same raw-softmax behavior this function always had."""
    logits, word_ids = extract_logits(image, words, boxes, model, processor)
    probs = F.softmax(logits / temperature, dim=-1)
    # .tolist() once instead of indexing the tensor + .item() per token in the loop below --
    # avoids up to 512 individual tensor->Python scalar syncs for a sequence this long.
    confidences, predictions = probs.max(dim=-1)
    confidences = confidences.tolist()
    predictions = predictions.tolist()

    fields = []
    for token_idx, word_idx in first_token_per_word(word_ids):
        label = model.config.id2label[predictions[token_idx]]
        if label != "O":
            fields.append({
                "word": words[word_idx],
                "label": label,
                "confidence": confidences[token_idx],
            })
    return fields
