import numpy as np
import torch

from src.calibrate.flagging import should_flag, threshold_report
from src.calibrate.metrics import binned_stats, compute_ece, compute_mce, confidence_and_correctness
from src.calibrate.reliability_diagram import plot_reliability_diagram
from src.calibrate.temperature_scaling import fit_temperature


def test_ece_zero_for_perfect_calibration():
    # In each of 10 confidence buckets, accuracy exactly equals the bucket's confidence level.
    rng = np.random.default_rng(0)
    confidences, correct = [], []
    for level in np.arange(0.05, 1.0, 0.1):
        n = 200
        confidences.extend([level] * n)
        correct.extend((rng.random(n) < level).astype(float))
    ece = compute_ece(np.array(confidences), np.array(correct), n_bins=10)
    assert ece < 0.03  # small sampling noise only


def test_ece_and_mce_detect_overconfidence():
    # Always 99% confident, but only correct 60% of the time -- badly overconfident.
    n = 1000
    confidences = np.full(n, 0.99)
    correct = (np.arange(n) % 5 < 3).astype(float)  # 60% correct
    ece = compute_ece(confidences, correct, n_bins=15)
    mce = compute_mce(confidences, correct, n_bins=15)
    assert ece > 0.35
    assert mce > 0.35


def test_confidence_and_correctness_extraction():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels = np.array([0, 1])
    confidences, correct = confidence_and_correctness(probs, labels)
    np.testing.assert_allclose(confidences, [0.7, 0.8])
    np.testing.assert_array_equal(correct, [1.0, 0.0])  # second example predicted class 2, true is 1


def test_temperature_scaling_preserves_accuracy_and_reduces_ece():
    torch.manual_seed(0)
    n, c = 1000, 4
    labels = torch.randint(0, c, (n,))
    correct_mask = torch.rand(n) < 0.7  # a model that's right 70% of the time
    base_logits = torch.randn(n, c) * 0.5
    for i in range(n):
        target = labels[i] if correct_mask[i] else (labels[i] + 1) % c
        base_logits[i, target] += 2.0

    overconfident_logits = base_logits * 5.0  # scaling up creates overconfidence without touching argmax

    preds_before = overconfident_logits.argmax(dim=1)
    accuracy_before = (preds_before == labels).float().mean().item()

    T = fit_temperature(overconfident_logits, labels)
    assert T > 1.0  # should learn to cool down the overconfident logits

    scaled_logits = overconfident_logits / T
    preds_after = scaled_logits.argmax(dim=1)
    accuracy_after = (preds_after == labels).float().mean().item()

    # The core correctness property of temperature scaling: predictions/accuracy must be unchanged.
    assert torch.equal(preds_before, preds_after)
    assert accuracy_before == accuracy_after

    probs_before = torch.softmax(overconfident_logits, dim=1).numpy()
    probs_after = torch.softmax(scaled_logits, dim=1).numpy()
    conf_before, corr_before = confidence_and_correctness(probs_before, labels.numpy())
    conf_after, corr_after = confidence_and_correctness(probs_after, labels.numpy())

    ece_before = compute_ece(conf_before, corr_before)
    ece_after = compute_ece(conf_after, corr_after)
    assert ece_after < ece_before


def test_threshold_report_monotonic_auto_approve_rate():
    rng = np.random.default_rng(1)
    confidences = rng.uniform(0.5, 1.0, 500)
    correct = (rng.random(500) < confidences).astype(float)
    rows = threshold_report(confidences, correct, thresholds=[0.5, 0.7, 0.9, 0.99])
    rates = [r["auto_approve_rate"] for r in rows]
    assert rates == sorted(rates, reverse=True)  # higher threshold -> fewer auto-approvals
    assert all(0.0 <= r["auto_approve_rate"] <= 1.0 for r in rows)


def test_should_flag_scalar():
    # Same canonical rule src/pipeline.py applies to one live prediction at a time.
    assert should_flag(0.5, threshold=0.9) is True
    assert should_flag(0.95, threshold=0.9) is False
    assert should_flag(0.9, threshold=0.9) is False  # exactly at threshold -> not flagged


def test_should_flag_array_matches_threshold_report_auto_approve_mask():
    # `<` broadcasts over a numpy array the same way it does over a scalar -- this is the
    # property that lets threshold_report reuse should_flag instead of re-deriving the rule.
    confidences = np.array([0.5, 0.7, 0.9, 0.95, 0.99])
    flagged = should_flag(confidences, threshold=0.9)
    np.testing.assert_array_equal(flagged, [True, True, False, False, False])


def test_plot_reliability_diagram_bars_match_binned_stats_accuracy():
    # `ax is not None` alone can't fail (plot_reliability_diagram always returns an Axes) -- this
    # cross-checks the actual plotted bar heights against binned_stats' own accuracy values, so a
    # bug that decouples the plot from the underlying stats would be caught.
    rng = np.random.default_rng(2)
    confidences = rng.uniform(0, 1, 200)
    correct = (rng.random(200) < confidences).astype(float)
    ax = plot_reliability_diagram(confidences, correct, n_bins=15)

    expected_accs = [s[3] for s in binned_stats(confidences, correct, n_bins=15)]
    plotted_accs = [patch.get_height() for patch in ax.patches]
    np.testing.assert_allclose(plotted_accs, expected_accs)


def test_plot_reliability_diagram_count_panel_matches_binned_stats_counts():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    confidences = rng.uniform(0, 1, 200)
    correct = (rng.random(200) < confidences).astype(float)
    _, (ax, count_ax) = plt.subplots(2, 1)
    plot_reliability_diagram(confidences, correct, n_bins=15, ax=ax, count_ax=count_ax)

    expected_counts = [s[5] for s in binned_stats(confidences, correct, n_bins=15)]
    plotted_counts = [int(round(patch.get_height())) for patch in count_ax.patches]
    assert plotted_counts == expected_counts
    plt.close("all")


def test_binned_stats_counts_sum_to_total():
    rng = np.random.default_rng(4)
    confidences = rng.uniform(0, 1, 500)
    correct = (rng.random(500) < confidences).astype(float)
    stats = binned_stats(confidences, correct, n_bins=15)
    total_count = sum(s[5] for s in stats)
    assert total_count == 500


def test_compute_ece_mce_and_reliability_diagram_accept_precomputed_stats():
    # compute_ece/compute_mce/plot_reliability_diagram all consume the exact same per-bin stats
    # -- a caller computing all three (e.g. notebooks/03_calibrate_classifier.ipynb) should be
    # able to call binned_stats once and share it, instead of each function silently recomputing
    # it. Passing precomputed stats must give bit-identical results to letting each function
    # compute its own.
    rng = np.random.default_rng(5)
    confidences = rng.uniform(0, 1, 300)
    correct = (rng.random(300) < confidences).astype(float)
    stats = binned_stats(confidences, correct, n_bins=15)

    assert compute_ece(confidences, correct, n_bins=15, stats=stats) == compute_ece(confidences, correct, n_bins=15)
    assert compute_mce(confidences, correct, n_bins=15, stats=stats) == compute_mce(confidences, correct, n_bins=15)

    ax = plot_reliability_diagram(confidences, correct, n_bins=15, stats=stats)
    assert ax is not None
