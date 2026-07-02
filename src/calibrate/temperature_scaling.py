"""Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural Networks")."""
import torch
import torch.nn.functional as F


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
