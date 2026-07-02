"""Calibration metrics: top-label ECE and MCE (Guo et al. 2017)."""
import numpy as np


def confidence_and_correctness(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """From a (N, C) probability matrix and (N,) true labels, get top-label confidence + correctness."""
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    correct = (predictions == labels).astype(float)
    return confidences, correct


def binned_stats(confidences: np.ndarray, correct: np.ndarray, n_bins: int):
    """Per-bin (lo, hi, proportion, accuracy, mean_confidence, count). Shared by ECE/MCE and the reliability diagram.

    `count` is an exact per-bin sample count (not derived from the rounded `proportion`) --
    reliability diagrams can otherwise show a misleadingly "perfect" or "terrible" bar for a bin
    with only a handful of examples, which is exactly the pattern that surfaced a suspicious
    100%-accuracy low-confidence bar on the real classifier's diagram.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    stats = []
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        count_in_bin = int(in_bin.sum())
        prop_in_bin = in_bin.mean()
        if count_in_bin > 0:
            acc_in_bin = correct[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
        else:
            acc_in_bin = conf_in_bin = 0.0
        stats.append((lo, hi, prop_in_bin, acc_in_bin, conf_in_bin, count_in_bin))
    return stats


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15, stats=None) -> float:
    """Expected Calibration Error: bin-size-weighted average of |accuracy - confidence|.

    Pass `stats` (this module's own `binned_stats(...)` output) if the caller is also computing
    MCE and/or a reliability diagram from the same (confidences, correct, n_bins) -- all three
    consume identical per-bin stats, so computing them once and sharing avoids redundant scans
    over the same data (see notebooks/03_calibrate_classifier.ipynb, which does exactly this).
    """
    if stats is None:
        stats = binned_stats(confidences, correct, n_bins)
    ece = 0.0
    for _, _, prop_in_bin, acc_in_bin, conf_in_bin, _ in stats:
        ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return ece


def compute_mce(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15, stats=None) -> float:
    """Maximum Calibration Error: worst-bin |accuracy - confidence| (unweighted, unlike ECE).

    See `compute_ece`'s docstring re: the optional `stats` parameter.
    """
    if stats is None:
        stats = binned_stats(confidences, correct, n_bins)
    gaps = [
        abs(acc_in_bin - conf_in_bin)
        for _, _, prop_in_bin, acc_in_bin, conf_in_bin, _ in stats
        if prop_in_bin > 0
    ]
    return max(gaps) if gaps else 0.0
