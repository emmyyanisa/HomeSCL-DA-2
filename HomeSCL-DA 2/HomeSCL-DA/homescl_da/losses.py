"""Loss functions."""
import numpy as np
import torch
import torch.nn.functional as F


def class_weights(segments, classes, cap):
    """Square-root inverse-frequency weights, dominant class = 1, capped."""
    counts = np.zeros(len(classes), dtype=np.float32)
    for s in segments:
        counts[s["label"]] += 1
    print("  train counts: " + ", ".join(f"{c}={int(n)}" for c, n in zip(classes, counts)))
    counts = np.clip(counts, 1, None)
    w = np.sqrt(counts.max() / counts)
    w = np.clip(w / w[int(np.argmax(counts))], None, cap)
    return torch.tensor(w, dtype=torch.float32)


def weighted_ce(logits, labels, weight):
    """Per-sample weighted CE averaged over samples (not over weights)."""
    return F.cross_entropy(logits, labels, weight=weight, reduction="none").mean()


def cross_home_supcon(z, labels, home_ids, temperature):
    """Supervised contrastive loss where each anchor only sees other-home samples.

    Positives: same class, different home. Negatives: different class,
    different home. Anchors without a positive are skipped.
    Returns (loss or None, number of valid anchors).
    """
    n = z.shape[0]
    if n < 2:
        return None, 0
    z = F.normalize(z, dim=-1)
    sim = z @ z.T / temperature
    allowed = home_ids[:, None].ne(home_ids[None, :])
    allowed &= ~torch.eye(n, dtype=torch.bool, device=z.device)
    positive = allowed & labels[:, None].eq(labels[None, :])
    valid = positive.any(dim=1)
    if not valid.any():
        return None, 0
    log_prob = sim - torch.logsumexp(sim.masked_fill(~allowed, -1e9), dim=1, keepdim=True)
    pos = positive.to(log_prob.dtype)
    mean_log_pos = (log_prob * pos).sum(dim=1) / pos.sum(dim=1).clamp_min(1)
    return -mean_log_pos[valid].mean(), int(valid.sum())
