"""MLP-only baseline: the MLP-BiGRU baseline with the BiGRU removed.

    CLAP [T, 1024] -> MLP (1024 -> 256 -> 64) -> masked mean -> linear -> activity

Everything else is identical to `main_ablation.py --ablation baseline`:
same segments, dedup, ADASYN, class-weighted CE, Adam, StepLR, seed and
final-epoch evaluation. The only difference is the temporal model, so the gap
between this and MLP-BiGRU measures what the BiGRU contributes.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .adasyn import adasyn_segments
from .dataset import SegmentDataset, collate_segments
from .losses import class_weights
from .model import masked_mean
from .reporting import save_fold_outputs
from .trainer import evaluate, train_one_epoch
from .utils import seed_worker, set_seed


class MLPOnly(nn.Module):
    """Per-step MLP, then mean over the valid steps of the segment."""

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
        self.classifier = nn.Linear(cfg.mlp_hidden2, cfg.n_classes)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        feat = masked_mean(self.backbone(x), mask)
        return self.classifier(feat), feat


def run_fold(cfg, heldout, segments, _target_data, store, out_root):
    source_names = [h["name"] for h in cfg.homes if h["name"] != heldout]
    out_dir = Path(out_root) / f"heldout_{heldout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.torch_device
    print(f"\n{'=' * 70}\nMLP-only LOHO fold  held-out: {heldout}\n"
          f"source homes: {source_names}\n{'=' * 70}")
    set_seed(cfg.seed)

    source = [s for s in segments if s["home"] != heldout]
    target = [s for s in segments if s["home"] == heldout]
    real, synthetic = adasyn_segments(source, store, cfg)
    train_set = real + synthetic

    cw = class_weights(train_set, cfg.classes, cfg.weight_cap).to(device)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, collate_fn=collate_segments,
                  num_workers=cfg.num_workers, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(SegmentDataset(train_set, store), shuffle=True, **common)
    heldout_loader = DataLoader(SegmentDataset(target, store), shuffle=False, **common)

    model = MLPOnly(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.sched_step, cfg.sched_gamma)
    print(f"  model params={sum(p.numel() for p in model.parameters()):,} device={device}")
    print(f"  adasyn=+{len(synthetic)} train={len(train_set)} held-out={len(target)}")

    history = {k: [] for k in ("train_loss", "heldout_loss", "heldout_f1")}
    print(f"\n{'ep':>4} {'train':>8} {'held':>8} {'F1':>7}")
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, None, train_loader, optimizer, cw, cfg)
        if cfg.track_heldout:
            held_loss, held_f1, _, _ = evaluate(model, heldout_loader, cw, cfg)
        else:
            held_loss, held_f1 = float("nan"), float("nan")
        scheduler.step()
        for key, val in (("train_loss", tr["total"]), ("heldout_loss", held_loss),
                         ("heldout_f1", held_f1)):
            history[key].append(val)
        print(f"{epoch:>4} {tr['total']:>8.4f} {held_loss:>8.4f} {held_f1:>7.4f}")

    test_loss, test_f1, pred, true = evaluate(model, heldout_loader, cw, cfg)
    return save_fold_outputs(cfg, out_dir, heldout, model, None, history,
                             test_loss, test_f1, pred, true)
