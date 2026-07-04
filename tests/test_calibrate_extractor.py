"""scripts/calibrate_extractor.py's collect_logits_and_labels() -- the one piece of real
aggregation logic in that script (everything else is thin calls into src/calibrate/* and
src/extract/predict.py, already covered by their own test files). Tested here with a fake
model/processor and a tiny in-memory dataset, the same style tests/test_extract.py uses for
extract_fields() -- no real FUNSD download or model weights needed.
"""
from types import SimpleNamespace

import torch
from PIL import Image

from scripts.calibrate_extractor import collect_logits_and_labels

ID2LABEL = {0: "O", 1: "B-QUESTION", 2: "B-ANSWER"}


class _FakeEncoding(dict):
    def __init__(self, word_ids):
        super().__init__()
        self._word_ids = word_ids

    def word_ids(self, batch_index=0):
        return self._word_ids


class _FakeModel:
    def __init__(self, id2label, logits_by_call):
        self.config = SimpleNamespace(id2label=id2label)
        self._logits_by_call = iter(logits_by_call)

    def __call__(self, **kwargs):
        return SimpleNamespace(logits=next(self._logits_by_call))


def _fake_processor(word_ids_by_call):
    calls = iter(word_ids_by_call)
    return lambda *args, **kwargs: _FakeEncoding(next(calls))


def test_collect_logits_and_labels_one_row_per_word_across_multiple_documents():
    # Two tiny documents: doc 1 has one word split into 2 subword tokens (only the first should
    # be collected); doc 2 has two single-token words. Proves the function collects exactly one
    # (logits, label) pair per real word, aggregated correctly across document boundaries.
    doc1_word_ids = [None, 0, 0, None]  # CLS, "Date"(2 subwords), SEP
    doc2_word_ids = [None, 0, 1, None]  # CLS, "Name", "Total", SEP

    doc1_logits = torch.zeros(1, len(doc1_word_ids), len(ID2LABEL))
    doc1_logits[0, 1, 1] = 10.0  # first subword of "Date" -> B-QUESTION
    doc1_logits[0, 2, 2] = 10.0  # second subword -- must NOT be collected

    doc2_logits = torch.zeros(1, len(doc2_word_ids), len(ID2LABEL))
    doc2_logits[0, 1, 2] = 10.0  # "Name" -> B-ANSWER
    doc2_logits[0, 2, 1] = 10.0  # "Total" -> B-QUESTION

    model = _FakeModel(ID2LABEL, [doc1_logits, doc2_logits])
    processor = _fake_processor([doc1_word_ids, doc2_word_ids])

    dataset = [
        {
            "image": Image.new("RGB", (10, 10)),
            "tokens": ["Date"],
            "bboxes": [[0, 0, 1, 1]],
            "ner_tags": [3],  # real FUNSD tag id for this word (arbitrary here, just distinct)
        },
        {
            "image": Image.new("RGB", (10, 10)),
            "tokens": ["Name", "Total"],
            "bboxes": [[0, 0, 1, 1], [0, 0, 1, 1]],
            "ner_tags": [5, 3],
        },
    ]

    logits, labels = collect_logits_and_labels(dataset, model, processor)

    # 3 real words total (1 + 2), not 4 tokens -- the duplicate subword must be dropped.
    assert logits.shape == (3, len(ID2LABEL))
    assert labels.tolist() == [3, 5, 3]
    # The collected logits are exactly the first-subword-token's row for each word.
    assert torch.equal(logits[0], doc1_logits[0, 1])
    assert torch.equal(logits[1], doc2_logits[0, 1])
    assert torch.equal(logits[2], doc2_logits[0, 2])
