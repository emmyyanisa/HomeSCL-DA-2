"""HomeSCL-DA networks.

MLPBiGRU: CLAP [T, D] -> MLP -> BiGRU -> masked mean -> activity logits.
The pooled BiGRU feature is shared by the activity head, the domain
discriminator (through a gradient reversal layer) and the SupCon projector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MLPBiGRU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(cfg.clap_dim, cfg.mlp_hidden1),
            nn.LayerNorm(cfg.mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden1, cfg.mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.bigru = nn.GRU(
            input_size=cfg.mlp_hidden2,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(cfg.feat_dim, cfg.n_classes)
        # Built only when SupCon is used, so models without SupCon have exactly
        # the parameters (and random initialisation) of the original scripts.
        self.projector = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.feat_dim),
            nn.ReLU(),
            nn.Linear(cfg.feat_dim, cfg.supcon_proj_dim),
        ) if cfg.use_supcon else None

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        out, _ = self.bigru(self.backbone(x))
        feat = masked_mean(out, mask)
        return self.classifier(feat), feat

    def project(self, feat):
        return F.normalize(self.projector(feat), dim=-1)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x, lambd):
    return _GradReverse.apply(x, lambd)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, n_domains, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x):
        return self.net(x)


def build_discriminator(cfg, n_domains):
    if cfg.ablation == "baseline":
        return None
    in_dim = cfg.feat_dim if cfg.ablation == "dann" else cfg.feat_dim * cfg.n_classes
    return DomainDiscriminator(in_dim, n_domains, cfg.domain_hidden, cfg.domain_dropout)


def domain_input(feat, probs, ablation):
    """DANN: pooled feature. CDAN / CDAN-E: flattened outer product feat x probs."""
    if ablation == "dann":
        return feat
    return torch.bmm(feat.unsqueeze(2), probs.unsqueeze(1)).flatten(1)


def entropy_weight(probs, eps=1e-8):
    """CDAN-E weight 1 + exp(-H(p)), normalised to mean 1."""
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
    w = 1.0 + torch.exp(-entropy)
    return w / (w.mean() + eps)
