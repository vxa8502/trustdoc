"""Confidence-threshold flagging: the human-in-the-loop trust-layer trade-off.

Business framing: at a given confidence threshold, what fraction of predictions get
auto-approved, and how precise are those auto-approved predictions? Raising the threshold
routes more predictions to human review but raises precision among what's auto-approved.
"""
import numpy as np


def should_flag(confidence, threshold: float):
    """True if confidence is below the threshold -> route to human review instead of
    auto-approving. The single canonical definition of the flagging rule: `<` broadcasts fine
    over both a scalar confidence (src/pipeline.py, one prediction) and a numpy array of
    confidences (threshold_report below, many predictions at once), so both call sites share
    this exact comparison instead of maintaining it in two places.
    """
    return confidence < threshold


def threshold_report(confidences: np.ndarray, correct: np.ndarray, thresholds=None) -> list[dict]:
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.99, 50)
    rows = []
    for t in thresholds:
        mask = ~should_flag(confidences, t)  # auto-approved = not flagged
        n_auto = int(mask.sum())
        rows.append({
            "threshold": float(t),
            "auto_approve_rate": float(mask.mean()),
            "precision": float(correct[mask].mean()) if n_auto > 0 else None,
            "n_auto_approved": n_auto,
        })
    return rows
