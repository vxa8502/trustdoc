from unittest.mock import patch

from PIL import Image, ImageDraw

from src.ocr.tesseract import extract_words_boxes


def _synthetic_document():
    img = Image.new("L", (800, 300), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), "INVOICE", fill=0)
    draw.text((50, 100), "Total Due: $1,250.00", fill=0)
    return img


def test_extract_words_boxes_finds_expected_text():
    words, boxes = extract_words_boxes(_synthetic_document())
    assert "INVOICE" in words
    assert len(words) == len(boxes)


def test_boxes_are_normalized_to_0_1000():
    words, boxes = extract_words_boxes(_synthetic_document())
    assert all(0 <= c <= 1000 for box in boxes for c in box)


def test_out_of_range_box_is_clipped_to_0_1000():
    # A tiny synthetic image's real Tesseract output never naturally goes out of range, so the
    # test above can't actually exercise the clip -- it would still pass even if the
    # max(0, min(1000, ...)) clamps were deleted entirely. This reproduces the real failure mode
    # the clip exists for (an unclipped out-of-range word box crashing training with a CUDA
    # index-out-of-bounds error), by faking a Tesseract box that extends past the image bounds.
    fake_tesseract_output = {
        "text": ["OVERFLOW"],
        "left": [0],
        "top": [0],
        "width": [850],  # wider than the 800px-wide image below
        "height": [50],
    }
    img = Image.new("RGB", (800, 300), color="white")

    with patch("src.ocr.tesseract.pytesseract.image_to_data", return_value=fake_tesseract_output):
        words, boxes = extract_words_boxes(img)

    assert words == ["OVERFLOW"]
    # Unclipped this would be int(1000 * 850 / 800) = 1062 -- outside [0, 1000].
    assert boxes[0] == [0, 0, 1000, 166]


def test_grayscale_input_is_handled():
    # LayoutLMv3 requires RGB; the wrapper must not crash or mis-scale on grayscale input.
    img = Image.new("L", (400, 200), color=255)
    ImageDraw.Draw(img).text((20, 20), "MEMO", fill=0)
    words, _ = extract_words_boxes(img)
    assert "MEMO" in words
