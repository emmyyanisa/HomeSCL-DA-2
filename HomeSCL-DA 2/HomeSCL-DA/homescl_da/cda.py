"""CDA: contrastive-adversarial domain adaptation on the MLP-BiGRU encoder.

Port of run_train_test_5_4_CDA_2stage_balance_domain.py
(Yadav et al., arXiv:2301.03826, adapted to CLAP episodes).

  e <  E'      : CE + SupCL                                (Stage I)
  E' <= e < E'': CE + SupCL + lambda * Adv                 (Stage I)
  e >= E''     : CE + lambda * Adv + beta * CrossCL        (Stage II, SupCL off)

* Adv uses the home-balanced real-audio domain stream; the GRL inside the
  discriminator has a fixed coefficient 1.0 and the loss is scaled by lambda.
* CrossCL pairs every source sample with target windows pseudo-labelled by
  argmax (no confidence filter). Target windows come from unlabeled_tsv only;
  held-out labels never enter training.
* The order of every random draw matches the original script, so the same
  seed gives the same run.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .adasyn import adasyn_segments
from .config import Config
from .dataset import SegmentDataset, _pad, collate_segments, cycle
from .losses import class_weights
from .model import DomainDiscriminator, MLPBiGRU, grad_reverse
from .reporting import save_fold_outputs
from .trainer import build_domain_stream, evaluate
from .utils import seed_worker, set_seed


@dataclass
class CDAConfig(Config):
    # Defaults reproduce run_train_test_5_4_CDA_2stage_balance_domain.py
    epochs: int = 90
    sched_step: int = 10
    sched_gamma: float = 0.8
    seed: int = 189
    domain_dropout: float = 0.5
    use_supcon: bool = False             # CDA has no HomeSCL projector / selection

    cda_tau: float = 0.5
    cda_e_prime: int = 50                # adversarial ramp starts
    cda_e_double_prime: int = 75         # Stage II: SupCL off, CrossCL on
    cda_lambda_gamma: float = 5.0
    cda_beta_alpha: Optional[float] = None   # None = E'' / (E - E'')
    cda_class_weights: bool = False      # paper uses plain CE
    cda_heldout_audio_as_target: bool = False  # transductive switch (labels discarded)

    @property
    def beta_alpha(self):
        if self.cda_beta_alpha is not None:
            return self.cda_beta_alpha
        return self.cda_e_double_prime / max(self.epochs - self.cda_e_double_prime, 1)

    def validate(self):
        super().validate()
        if self.use_supcon:
            raise ValueError("CDA has its own contrastive losses; set use_supcon=false.")
        if not 0 < self.cda_e_prime <= self.cda_e_double_prime <= self.epochs:
            raise ValueError("Need 0 < cda_e_prime <= cda_e_double_prime <= epochs.")
        return self


# ---------------------------------------------------------------- schedules
def cda_lambda(epoch, cfg):
    if epoch < cfg.cda_e_prime:
        return 0.0
    p = np.clip((epoch - cfg.cda_e_prime) / max(cfg.cda_e_double_prime - cfg.cda_e_prime, 1),
                0.0, 1.0)
    return float(2.0 / (1.0 + np.exp(-cfg.cda_lambda_gamma * p)) - 1.0)


def cda_beta(epoch, cfg):
    if epoch <= cfg.cda_e_double_prime:
        return 0.0
    return float(min(1.0, cfg.beta_alpha * (epoch - cfg.cda_e_double_prime) / cfg.cda_e_double_prime))


def cda_stage(epoch, cfg):
    if epoch < cfg.cda_e_prime:
        return "I: CE+SupCL"
    if epoch < cfg.cda_e_double_prime:
        return "I: +lambda*Adv"
    return "II: Adv+CrossCL"


# ------------------------------------------------------------------- losses
def supcl_loss(z, labels, tau):
    """Stage-I supervised contrastive loss over the whole source batch."""
    z = F.normalize(z, p=2, dim=1)
    n = z.size(0)
    if n < 2:
        return z.sum() * 0.0, 0
    sim = z @ z.T / tau
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = labels[:, None].eq(labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1, keepdim=True)
    pos_count = pos.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return z.sum() * 0.0, 0
    per_anchor = -(log_prob.masked_fill(~pos, 0.0).sum(dim=1) / pos_count.clamp(min=1))
    return per_anchor[valid].mean(), int(valid.sum().item())


def crosscl_loss(z_src, y_src, z_tgt, y_tgt, tau):
    """Stage-II CrossCL (Eq. 5): source anchor vs target positives / negatives."""
    z_src = F.normalize(z_src, p=2, dim=1)
    z_tgt = F.normalize(z_tgt, p=2, dim=1)
    if z_src.numel() == 0 or z_tgt.numel() == 0:
        return (z_src.sum() + z_tgt.sum()) * 0.0, 0
    sim = z_src @ z_tgt.T / tau
    pos_mask = y_src[:, None].eq(y_tgt[None, :])
    losses, pairs = [], 0
    for i in range(sim.size(0)):
        pos_scores = sim[i][pos_mask[i]]
        neg_scores = sim[i][~pos_mask[i]]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        neg_lse = torch.logsumexp(neg_scores, dim=0)
        denom = torch.logaddexp(pos_scores, neg_lse.expand_as(pos_scores))
        losses.append((denom - pos_scores).mean())
        pairs += int(pos_scores.numel())
    if not losses:
        return (z_src.sum() + z_tgt.sum()) * 0.0, 0
    return torch.stack(losses).mean(), pairs


# -------------------------------------------------------------------- data
class TargetDataset(Dataset):
    """Unlabeled target windows -> features [T, D]."""

    def __init__(self, items, store):
        self.items = items
        self.store = store

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.tensor(self.store.features(self.items[i]), dtype=torch.float32)


def collate_target(batch):
    return _pad(batch)


# ------------------------------------------------------------------ training
def train_one_epoch(model, disc, loader, optimizer, ce_fn, cfg, epoch,
                    target_iter, domain_iter):
    device = cfg.torch_device
    model.train()
    disc.train()
    lam, beta = cda_lambda(epoch, cfg), cda_beta(epoch, cfg)
    use_supcl = epoch < cfg.cda_e_double_prime
    use_crosscl = epoch >= cfg.cda_e_double_prime
    domain_active = epoch >= cfg.cda_e_prime and domain_iter is not None

    totals = dict(total=0.0, ce=0.0, supcl=0.0, adv=0.0, crosscl=0.0)
    steps = d_correct = d_count = anchors = pairs = 0
    pseudo_hist = torch.zeros(cfg.n_classes, dtype=torch.long)

    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits, z_src = model(x, mask)
        ce = ce_fn(logits, y)
        total = ce

        supcl = z_src.sum() * 0.0
        if use_supcl:
            supcl, n_valid = supcl_loss(z_src, y, cfg.cda_tau)
            anchors += n_valid
            total = total + supcl

        adv = z_src.sum() * 0.0
        crosscl = z_src.sum() * 0.0

        t_logits = z_tgt = None
        if use_crosscl:
            tx, tmask = next(target_iter)
            t_logits, z_tgt = model(tx.to(device), tmask.to(device))

        if domain_active:
            dx, dmask, dy = (t.to(device) for t in next(domain_iter))
            _, z_dom = model(dx, dmask)
            d_logits = disc(grad_reverse(z_dom, 1.0))
            adv = F.cross_entropy(d_logits, dy)
            with torch.no_grad():
                d_correct += (d_logits.argmax(1) == dy).sum().item()
                d_count += dy.numel()
            total = total + lam * adv

        if use_crosscl:
            with torch.no_grad():
                pseudo = t_logits.argmax(dim=1)
                pseudo_hist += torch.bincount(pseudo.cpu(), minlength=cfg.n_classes)
            if beta > 0.0:
                crosscl, n_pairs = crosscl_loss(z_src, y, z_tgt, pseudo, cfg.cda_tau)
                pairs += n_pairs
                total = total + beta * crosscl

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip)
        optimizer.step()

        totals["total"] += total.item()
        totals["ce"] += ce.item()
        totals["supcl"] += supcl.item()
        totals["adv"] += adv.item()
        totals["crosscl"] += crosscl.item()
        steps += 1

    out = {k: v / max(steps, 1) for k, v in totals.items()}
    out.update(lam=lam, beta=beta, anchors=anchors, pairs=pairs, pseudo_hist=pseudo_hist.numpy(),
               domain_acc=d_correct / d_count if d_count else float("nan"))
    return out


def run_fold(cfg, heldout, segments, unlabeled, store, out_root):
    source_names = [h["name"] for h in cfg.homes if h["name"] != heldout]
    out_dir = Path(out_root) / f"heldout_{heldout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.torch_device
    print(f"\n{'=' * 70}\nCDA LOHO fold  held-out: {heldout}\nsource homes: {source_names}\n{'=' * 70}")
    set_seed(cfg.seed)

    source = [s for s in segments if s["home"] != heldout]
    target = [s for s in segments if s["home"] == heldout]

    real, synthetic = adasyn_segments(source, store, cfg)
    train_set = real + synthetic
    cw = class_weights(train_set, cfg.classes, cfg.weight_cap).to(device)
    weight = cw if cfg.cda_class_weights else None

    def ce_fn(logits, labels):
        return F.cross_entropy(logits, labels, weight=weight)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, collate_fn=collate_segments,
                  num_workers=cfg.num_workers, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(SegmentDataset(train_set, store), shuffle=True, **common)
    heldout_loader = DataLoader(SegmentDataset(target, store), shuffle=False, **common)

    # Target pool for CrossCL and the adversarial branch: unlabeled audio only.
    target_items = list(unlabeled.get(heldout, []))
    if cfg.cda_heldout_audio_as_target:
        target_items += [{"home": s["home"], "batch_no": s["batch_no"], "indices": s["indices"]}
                         for s in target]
        print(f"  [target] +{len(target)} held-out labeled-audio windows (labels discarded)")
    print(f"  [target] unlabeled windows used by CDA: {len(target_items)}")
    if not target_items:
        raise RuntimeError(f"No target windows for {heldout}: CDA needs the unlabeled_tsv "
                           "(or cda_heldout_audio_as_target=true).")
    target_loader = DataLoader(TargetDataset(target_items, store), batch_size=cfg.batch_size,
                               shuffle=True, collate_fn=collate_target,
                               num_workers=cfg.num_workers, worker_init_fn=seed_worker)
    target_iter = cycle(target_loader)
    domain_iter = build_domain_stream(cfg, source, source_names, target_items, store)

    model = MLPBiGRU(cfg).to(device)
    n_domains = len(source_names) + (cfg.domain_mode == "multi_source_target")
    disc = DomainDiscriminator(cfg.feat_dim, n_domains, cfg.domain_hidden,
                               cfg.domain_dropout).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.sched_step, cfg.sched_gamma)

    print(f"  model params={sum(p.numel() for p in model.parameters()):,} "
          f"discriminator params={sum(p.numel() for p in disc.parameters()):,} device={device}")
    print(f"  domain_mode={cfg.domain_mode} adasyn=+{len(synthetic)} train={len(train_set)} "
          f"held-out={len(target)}  domain chance accuracy ~ {1.0 / n_domains:.3f}")
    print(f"  E={cfg.epochs} E'={cfg.cda_e_prime} E''={cfg.cda_e_double_prime} tau={cfg.cda_tau} "
          f"gamma={cfg.cda_lambda_gamma} alpha={cfg.beta_alpha:.3f} "
          f"class_weights={'on' if cfg.cda_class_weights else 'off'}")

    history = {k: [] for k in ("train_loss", "ce", "supcl", "adv", "crosscl", "lambda", "beta",
                               "domain_acc", "heldout_loss", "heldout_f1")}
    print(f"\n{'ep':>4} {'stage':<16} {'train':>8} {'held':>8} {'F1':>7}  details")
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, disc, train_loader, optimizer, ce_fn, cfg, epoch,
                             target_iter, domain_iter)
        if cfg.track_heldout:
            held_loss, held_f1, _, _ = evaluate(model, heldout_loader, None, cfg, loss_fn=ce_fn)
        else:
            held_loss, held_f1 = float("nan"), float("nan")
        scheduler.step()

        for key, val in (("train_loss", tr["total"]), ("ce", tr["ce"]), ("supcl", tr["supcl"]),
                         ("adv", tr["adv"]), ("crosscl", tr["crosscl"]), ("lambda", tr["lam"]),
                         ("beta", tr["beta"]), ("domain_acc", tr["domain_acc"]),
                         ("heldout_loss", held_loss), ("heldout_f1", held_f1)):
            history[key].append(val)

        print(f"{epoch:>4} {cda_stage(epoch, cfg):<16} {tr['total']:>8.4f} {held_loss:>8.4f} "
              f"{held_f1:>7.4f}  ce={tr['ce']:.4f} supcl={tr['supcl']:.4f} adv={tr['adv']:.4f} "
              f"cross={tr['crosscl']:.4f} lam={tr['lam']:.3f} beta={tr['beta']:.3f} "
              f"acc={tr['domain_acc']:.3f}")
        e2 = cfg.cda_e_double_prime
        if epoch >= e2 and (epoch in (e2, e2 + 1) or epoch % 10 == 0):
            hist = ", ".join(f"{c}={int(n)}" for c, n in zip(cfg.classes, tr["pseudo_hist"]))
            print(f"       pseudo(all target, no filter): {hist}; cross-positive-pairs={tr['pairs']}")

    test_loss, test_f1, pred, true = evaluate(model, heldout_loader, None, cfg, loss_fn=ce_fn)
    return save_fold_outputs(cfg, out_dir, heldout, model, disc, history,
                             test_loss, test_f1, pred, true)
