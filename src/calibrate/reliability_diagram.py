"""Reliability diagrams: binned accuracy vs. confidence, against the perfect-calibration diagonal.

Includes a per-bin sample-count panel by default (Guo et al.'s own presentation does this too) --
without it, a sparsely populated bin can show a misleadingly "perfect" or "terrible" accuracy bar
that's really just noise from a handful of examples.
"""
import numpy as np
import matplotlib.pyplot as plt

from src.calibrate.metrics import binned_stats


def plot_reliability_diagram(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15,
                              title: str = "Reliability Diagram", ax=None, count_ax=None, stats=None):
    """Pass `stats` (this module's `binned_stats(...)` output) to reuse a computation already
    done for ECE/MCE on the same data instead of scanning the confidence array a third time --
    see src/calibrate/metrics.py:compute_ece's docstring."""
    if stats is None:
        stats = binned_stats(confidences, correct, n_bins)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    accs = [s[3] for s in stats]
    counts = [s[5] for s in stats]
    width = (1.0 / n_bins) * 0.9

    if ax is None:
        ax = plt.subplots(figsize=(5, 5))[1]

    ax.bar(bin_centers, accs, width=width, edgecolor="black", alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()

    if count_ax is not None:
        count_ax.bar(bin_centers, counts, width=width, color="gray", alpha=0.7)
        count_ax.set_xlabel("Confidence")
        count_ax.set_ylabel("Count")
        count_ax.set_xlim(0, 1)
    else:
        ax.set_xlabel("Confidence")

    return ax
