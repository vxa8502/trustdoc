"""Tesseract OCR wrapper: raw document image -> words + LayoutLMv3-ready normalized boxes."""
from PIL import Image
import pytesseract


def extract_words_boxes(image: Image.Image) -> tuple[list[str], list[list[int]]]:
    """Run Tesseract OCR and return (words, boxes), boxes normalized to a 0-1000 scale.

    LayoutLMv3 requires box coordinates in [0, 1000] relative to image size -- an unclipped
    out-of-range box crashes training with a CUDA scatter/gather index-out-of-bounds error,
    so every box is clamped here regardless of whether it looks like it needs it.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")  # LayoutLMv3 requires RGB, not grayscale

    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    words: list[str] = []
    boxes: list[list[int]] = []
    for i, text in enumerate(data["text"]):
        word = text.strip()
        if not word:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [
            max(0, min(1000, int(1000 * x / width))),
            max(0, min(1000, int(1000 * y / height))),
            max(0, min(1000, int(1000 * (x + w) / width))),
            max(0, min(1000, int(1000 * (y + h) / height))),
        ]
        words.append(word)
        boxes.append(box)

    return words, boxes
