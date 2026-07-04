"""Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural Networks")."""
import torch
import torch.nn.functional as F


class StaleCalibrationError(Exception):
    """Raised when a fitted temperature is applied to a different model revision than the one
    it was fit against. Retraining silently invalidates a previously-fit T (confirmed in
    practice: the same test image's confidence drifted from 0.948 to 0.978 when a stale T=1.074
    was applied to a retrained classifier) -- this makes that drift loud instead of silent."""


def check_model_revision(model, expected_revision: str, model_name: str) -> None:
    """Raise StaleCalibrationError if `model`'s loaded Hub commit doesn't match the revision a
    calibration config was fit against. `_commit_hash` is set by `from_pretrained` on any model
    loaded from the Hub; comparing it here is what lets a config drift from its model loudly
    (a re-run failure) instead of silently (a miscalibrated confidence nobody notices)."""
    actual_revision = getattr(model.config, "_commit_hash", None)
    if actual_revision != expected_revision:
        raise StaleCalibrationError(
            f"{model_name} is at revision {actual_revision!r}, but its calibration config was "
            f"fit against revision {expected_revision!r}. Re-run the calibration notebook "
            f"against the current model and update the config's pinned revision before trusting "
            f"this temperature/threshold again."
        )


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50, lr: float = 0.01) -> float:
    """Fit a scalar temperature T minimizing NLL of softmax(logits/T) on held-out data.

    Dividing every logit by the same positive constant T never changes the argmax, so accuracy
    is provably unchanged -- only confidence is reshaped. T is optimized in log-space so it stays
    positive throughout (a plain unconstrained T can wander to <=0 mid-optimization, which breaks
    softmax); the returned value is exp(log_T).
    """
    logits = logits.detach()
    labels = labels.detach()
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_T.exp().item()
