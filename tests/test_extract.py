"""extract_fields()'s token->word aggregation (subword dedup, special-token skipping, "O"
filtering) is pure Python with no dependency on real model weights -- tested here with a fake
model/processor instead of the real network. Only the surrounding processor(...)/model(**encoding)
calls themselves are thin pass-throughs to the real HF API, verified manually instead (see
tests/test_pipeline.py's module docstring)."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.extract.predict import extract_fields, first_token_per_word

# softmax([0, 0, 10]) for a 3-class problem -- the exact confidence _fake_model's dominant
# logit (10.0) produces, computed once here instead of a magic-looking literal in each test.
_DOMINANT_LOGIT_CONFIDENCE = torch.softmax(torch.tensor([0.0, 0.0, 10.0]), dim=-1).max().item()

ID2LABEL = {0: "O", 1: "B-QUESTION", 2: "B-ANSWER"}


class _FakeEncoding(dict):
    """Mimics a BatchEncoding just enough for extract_fields: unpacks as empty kwargs via
    **encoding, and reports which source word each token belongs to via word_ids()."""

    def __init__(self, word_ids):
        super().__init__()
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


def _fake_processor(word_ids):
    return lambda *args, **kwargs: _FakeEncoding(word_ids)


class _FakeModel:
    # SimpleNamespace can't be made callable via an instance attribute -- Python looks up
    # __call__ on the type, not the instance -- so this needs a real (tiny) class.
    def __init__(self, id2label, logits):
        self.config = SimpleNamespace(id2label=id2label)
        self._logits = logits

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=self._logits)


def _fake_model(predicted_label_ids):
    """predicted_label_ids: one label id per token; argmax(logits) must recover exactly these."""
    n_tokens, n_labels = len(predicted_label_ids), len(ID2LABEL)
    logits = torch.zeros(1, n_tokens, n_labels)
    for token_idx, label_id in enumerate(predicted_label_ids):
        logits[0, token_idx, label_id] = 10.0  # dominates softmax/argmax for that token
    return _FakeModel(ID2LABEL, logits)


def test_first_token_per_word_dedups_subwords_and_skips_specials():
    # The shared rule extract_fields() and scripts/calibrate_extractor.py both consume -- tested
    # directly here as a pure function, independent of any model/processor.
    word_ids = [None, 0, 1, 1, 1, 1, 2, None]
    #             ^CLS  Date  --------8/4/92-------- Random ^SEP
    assert first_token_per_word(word_ids) == [(1, 0), (2, 1), (6, 2)]


def test_first_token_per_word_empty_when_all_special():
    assert first_token_per_word([None, None]) == []


def test_extract_fields_dedups_subwords_skips_specials_and_filters_o():
    # 3 words: "Date" (1 token), "8/4/92" (4 subword tokens), "Random" (1 token) -- with a
    # leading/trailing special token (word_id None) like a real [CLS]/[SEP] would produce.
    words = ["Date", "8/4/92", "Random"]
    word_ids = [None, 0, 1, 1, 1, 1, 2, None]
    #             ^CLS  Date  --------8/4/92-------- Random ^SEP

    # Only the FIRST subword's prediction should be kept for "8/4/92" (B-ANSWER); the later
    # subword tokens are deliberately predicted differently (O, O, B-ANSWER) to prove that a
    # broken dedup (e.g. keeping the *last* subword, or overwriting on every subword) would be
    # caught here instead of silently passing.
    predicted_label_ids = [0, 1, 2, 0, 0, 2, 0, 0]
    #                    CLS  Date  8  /  4  /92 Random SEP

    model = _fake_model(predicted_label_ids)
    processor = _fake_processor(word_ids)
    image = Image.new("RGB", (10, 10))

    fields = extract_fields(image, words, boxes=[[0, 0, 1, 1]] * len(words), model=model, processor=processor)

    # "Random" is excluded: its first (only) token predicts "O".
    assert fields == [
        {"word": "Date", "label": "B-QUESTION", "confidence": pytest.approx(_DOMINANT_LOGIT_CONFIDENCE)},
        {"word": "8/4/92", "label": "B-ANSWER", "confidence": pytest.approx(_DOMINANT_LOGIT_CONFIDENCE)},
    ]


def test_extract_fields_returns_empty_list_when_everything_is_o_or_special():
    words = ["Random"]
    word_ids = [None, 0, None]
    predicted_label_ids = [0, 0, 0]  # CLS, "Random" -> O, SEP

    model = _fake_model(predicted_label_ids)
    processor = _fake_processor(word_ids)
    image = Image.new("RGB", (10, 10))

    fields = extract_fields(image, words, boxes=[[0, 0, 1, 1]], model=model, processor=processor)
    assert fields == []


def test_extract_fields_temperature_lowers_confidence_without_changing_labels():
    # Same property src/calibrate/temperature_scaling.py's classifier test asserts, applied to
    # the extractor's per-field confidence: cooling (T > 1) must lower confidence but must never
    # change which label wins (dividing every logit by the same positive constant can't flip an
    # argmax).
    words = ["Date"]
    word_ids = [None, 0, None]
    predicted_label_ids = [0, 1, 0]

    model = _fake_model(predicted_label_ids)
    processor = _fake_processor(word_ids)
    image = Image.new("RGB", (10, 10))

    fields_uncalibrated = extract_fields(image, words, boxes=[[0, 0, 1, 1]], model=model, processor=processor)
    fields_cooled = extract_fields(
        image, words, boxes=[[0, 0, 1, 1]], model=model, processor=processor, temperature=5.0
    )

    assert fields_uncalibrated[0]["label"] == fields_cooled[0]["label"] == "B-QUESTION"
    assert fields_cooled[0]["confidence"] < fields_uncalibrated[0]["confidence"]
