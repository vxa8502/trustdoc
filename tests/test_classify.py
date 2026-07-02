"""label_names() is pure logic derived from a loaded model's config -- no real model/network
needed to test it, unlike classify() itself (a thin pass-through to the real HF processor/model,
verified manually instead per tests/test_pipeline.py's module docstring)."""
from types import SimpleNamespace

from src.classify.predict import label_names


def _fake_model(id2label: dict):
    return SimpleNamespace(config=SimpleNamespace(id2label=id2label, num_labels=len(id2label)))


def test_label_names_reads_from_model_config_in_id_order():
    model = _fake_model({2: "email", 0: "letter", 1: "form"})  # deliberately out of order
    assert label_names(model) == ["letter", "form", "email"]
