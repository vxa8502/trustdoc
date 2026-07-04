"""Fit temperature scaling for the extractor (NER head) on FUNSD's held-out test split.

Unlike the classifier's calibration (notebooks/03_calibrate_classifier.ipynb), this doesn't need
a GPU or Kaggle -- extractor inference is CPU-only and FUNSD's test split is 50 documents, so it
runs locally in under a minute. Closes a gap where the trust layer previously only covered the
classifier: each head is calibrated separately, and every extracted field now gets its own
confidence score, neither of which existed for the extractor before this script produced real
fitted values for them.

Reuses src/calibrate/* (fit_temperature, compute_ece/mce, threshold_report) instead of
reimplementing the math -- the same functions the classifier's calibration already trusts.

Usage: .venv/bin/python scripts/calibrate_extractor.py
Writes: results/extractor_calibration_summary.json
"""
import json
from pathlib import Path

import torch
from datasets import load_dataset

from src.calibrate.flagging import threshold_report
from src.calibrate.metrics import compute_ece, compute_mce, confidence_and_correctness
from src.calibrate.temperature_scaling import fit_temperature
from src.extract.predict import extract_logits, first_token_per_word, load_extractor

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "extractor_calibration_summary.json"


def collect_logits_and_labels(dataset, model, processor):
    """One row per real word across the whole split -- logits (N, num_labels) and true label
    ids (N,). Uses the same first_token_per_word() rule src/extract/predict.py's extract_fields()
    uses at inference time, imported rather than reimplemented, so a fitted temperature can never
    silently end up calibrated against a different word-selection than what inference actually
    aggregates."""
    all_logits, all_labels = [], []
    for example in dataset:
        image = example["image"].convert("RGB")
        words = example["tokens"]
        boxes = example["bboxes"]
        ner_tags = example["ner_tags"]

        logits, word_ids = extract_logits(image, words, boxes, model, processor)
        for token_idx, word_idx in first_token_per_word(word_ids):
            all_logits.append(logits[token_idx])
            all_labels.append(ner_tags[word_idx])
    return torch.stack(all_logits), torch.tensor(all_labels)


def main():
    print("Loading FUNSD test split...")
    test_ds = load_dataset("nielsr/funsd-layoutlmv3", split="test")

    print("Loading vxa8502/trustdoc-extractor...")
    model, processor = load_extractor()
    model_revision = model.config._commit_hash
    print(f"Model revision: {model_revision}")

    print(f"Running inference on {len(test_ds)} test documents...")
    logits, labels = collect_logits_and_labels(test_ds, model, processor)
    print(f"Collected {logits.shape[0]} word-level predictions across {len(test_ds)} documents.")

    probs_before = torch.softmax(logits, dim=-1).numpy()
    conf_before, correct_before = confidence_and_correctness(probs_before, labels.numpy())
    test_accuracy = float(correct_before.mean())
    ece_before = compute_ece(conf_before, correct_before)
    mce_before = compute_mce(conf_before, correct_before)

    print("Fitting temperature...")
    T = fit_temperature(logits, labels)
    print(f"Fitted temperature: {T:.4f}")

    probs_after = torch.softmax(logits / T, dim=-1).numpy()
    conf_after, correct_after = confidence_and_correctness(probs_after, labels.numpy())
    # Temperature scaling must never change argmax -- assert it here, the same invariant
    # tests/test_calibrate.py checks for the classifier, verified against this real run too.
    assert (probs_before.argmax(axis=1) == probs_after.argmax(axis=1)).all()
    ece_after = compute_ece(conf_after, correct_after)
    mce_after = compute_mce(conf_after, correct_after)

    rows = threshold_report(conf_after, correct_after)

    summary = {
        "model_revision": model_revision,
        "n_words": int(logits.shape[0]),
        "temperature": T,
        "test_accuracy": test_accuracy,
        "ece_before": ece_before,
        "mce_before": mce_before,
        "ece_after": ece_after,
        "mce_after": mce_after,
        "threshold_report": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {RESULTS_PATH}")
    print(f"test_accuracy={test_accuracy:.4f} ece_before={ece_before:.4f} ece_after={ece_after:.4f} "
          f"mce_before={mce_before:.4f} mce_after={mce_after:.4f}")


if __name__ == "__main__":
    main()
