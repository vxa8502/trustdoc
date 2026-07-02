"""Field extraction (NER) via the published TrustDoc extractor (HF Hub)."""
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

MODEL_ID = "vxa8502/trustdoc-extractor"


def load_extractor():
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID, apply_ocr=False)
    model.eval()
    return model, processor


@torch.no_grad()
def extract_fields(
    image: Image.Image, words: list[str], boxes: list[list[int]], model, processor
) -> list[dict]:
    """One {"word": str, "label": str} per OCR'd word with a non-"O" predicted tag."""
    encoding = processor(
        image, words, boxes=boxes, truncation=True, padding="max_length", return_tensors="pt"
    )
    # .tolist() once instead of indexing the tensor + .item() per token in the loop below --
    # avoids up to 512 individual tensor->Python scalar syncs for a sequence this long.
    predictions = model(**encoding).logits.argmax(dim=-1)[0].tolist()
    word_ids = encoding.word_ids(batch_index=0)  # maps each token back to its source word index

    fields = []
    seen_words = set()
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen_words:
            continue  # skip special tokens and subword continuations -- one label per word
        seen_words.add(word_idx)
        label = model.config.id2label[predictions[token_idx]]
        if label != "O":
            fields.append({"word": words[word_idx], "label": label})
    return fields
