"""Training, evaluation and one leave-one-home-out fold.

Loss = CE(activity)
     + adversarial domain CE via GRL          (ablation != "baseline")
     + lambda_s * cross-home SupCon on selected real source segments
The held-out home's labels are never used for training or model selection;
the final-epoch model is the one tested.
"""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .adasyn import adasyn_segments
from .dataset import (BalancedDomainBatchSampler, CrossHomeClassBatchSampler, DomainDataset,
                      SegmentDataset, collate_domain, collate_segments, cycle)
from .losses import class_weights, cross_home_supcon, weighted_ce
from .model import MLPBiGRU, build_discriminator, domain_input, entropy_weight, grad_reverse
from .selection import select_segments
from .utils import plot_confusion, plot_curves, seed_worker, set_seed

TARGET_GROUP = "__target__"


def domain_lambda(epoch, cfg):
    if epoch <= cfg.domain_warmup_epochs:
        return 0.0
    p = (epoch - cfg.domain_warmup_epochs) / max(cfg.epochs - cfg.domain_warmup_epochs, 1)
    return min(cfg.domain_lambda_max * (2.0 / (1.0 + np.exp(-5.0 * p)) - 1.0),
               cfg.domain_lambda_cap)


def build_domain_stream(cfg, source_segments, source_names, target_items, store):
    """Home-balanced real-audio batches: one domain id per source home (+ target)."""
    local_id = {name: i for i, name in enumerate(source_names)}
    items = [{"home": s["home"], "batch_no": s["batch_no"], "indices": s["indices"],
              "domain_id": local_id[s["home"]], "group": s["home"]} for s in source_segments]
    quotas = OrderedDict((name, cfg.domain_samples_per_home) for name in source_names)

    if cfg.domain_mode == "multi_source_target":
        if not target_items:
            print("  [domain] no target windows; domain loss disabled for this fold")
            return None
        items += [{**t, "domain_id": len(source_names), "group": TARGET_GROUP} for t in target_items]
        quotas[TARGET_GROUP] = cfg.domain_samples_per_home

    sampler = BalancedDomainBatchSampler([it["group"] for it in items], quotas,
                                         cfg.domain_min_batches, cfg.seed)
    loader = DataLoader(DomainDataset(items, store), batch_sampler=sampler,
                        collate_fn=collate_domain, num_workers=cfg.num_workers,
                        worker_init_fn=seed_worker)
    print(f"  [domain] {cfg.domain_mode}: {len(items)} windows, batch="
          + ", ".join(f"{g}:{q}" for g, q in quotas.items()))
    return cycle(loader)


def build_supcon_stream(cfg, selected, store):
    viable = any(len({s["home_id"] for s in selected if s["label"] == c}) >= 2
                 for c in range(cfg.n_classes))
    if len(selected) < 2 or not viable:
        print("  [supcon] no cross-home positive pairs; SupCon disabled for this fold")
        return None
    sampler = CrossHomeClassBatchSampler(selected, cfg.n_classes, cfg.supcon_homes_per_class,
                                         cfg.supcon_samples_per_home, cfg.supcon_min_batches,
                                         cfg.seed)
    loader = DataLoader(SegmentDataset(selected, store), batch_sampler=sampler,
                        collate_fn=collate_segments, num_workers=cfg.num_workers,
                        worker_init_fn=seed_worker)
    print(f"  [supcon] {len(selected)} selected segments, {len(sampler)} batches/cycle")
    return cycle(loader)


def train_one_epoch(model, disc, loader, optimizer, cw, cfg,
                    lam_d=0.0, domain_iter=None, lam_s=0.0, supcon_iter=None):
    device = cfg.torch_device
    model.train()
    if disc is not None:
        disc.train()
    totals = dict(total=0.0, ce=0.0, domain=0.0, supcon=0.0)
    d_correct = d_count = anchors = steps = 0

    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        logits, _ = model(x, mask)
        loss = weighted_ce(logits, y, cw)
        if not torch.isfinite(loss):
            continue
        ce_val, d_val, s_val = loss.item(), 0.0, 0.0

        if disc is not None and lam_d > 0 and domain_iter is not None:
            dx, dmask, dy = (t.to(device) for t in next(domain_iter))
            d_logits_act, d_feat = model(dx, dmask)
            d_probs = F.softmax(d_logits_act, dim=-1)
            d_logits = disc(grad_reverse(domain_input(d_feat, d_probs, cfg.ablation), lam_d))
            per_sample = F.cross_entropy(d_logits, dy, reduction="none")
            if cfg.ablation == "cdan_e":
                per_sample = entropy_weight(d_probs) * per_sample
            d_loss = per_sample.mean()
            loss = loss + d_loss
            d_val = d_loss.item()
            d_correct += (d_logits.argmax(1) == dy).sum().item()
            d_count += dy.numel()

        if lam_s > 0 and supcon_iter is not None:
            sx, sy, smask, sh = next(supcon_iter)
            _, s_feat = model(sx.to(device), smask.to(device))
            s_loss, n_anchor = cross_home_supcon(model.project(s_feat), sy.to(device),
                                                 sh.to(device), cfg.supcon_temperature)
            anchors += n_anchor
            if s_loss is not None and torch.isfinite(s_loss):
                loss = loss + lam_s * s_loss
                s_val = s_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if disc is not None:
            torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip)
        optimizer.step()

        totals["total"] += loss.item()
        totals["ce"] += ce_val
        totals["domain"] += d_val
        totals["supcon"] += s_val
        steps += 1

    out = {k: v / max(steps, 1) for k, v in totals.items()}
    out["domain_acc"] = d_correct / d_count if d_count else float("nan")
    out["anchors"] = anchors
    return out


@torch.no_grad()
def evaluate(model, loader, cw, cfg, loss_fn=None):
    """loss_fn(logits, labels) overrides the default per-sample weighted CE."""
    device = cfg.torch_device
    model.eval()
    preds, labels, losses = [], [], []
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits, _ = model(x, mask)
        loss = loss_fn(logits, y) if loss_fn is not None else weighted_ce(logits, y, cw)
        losses.append(loss.item())
        preds.append(logits.argmax(-1).cpu())
        labels.append(y.cpu())
    if not labels:
        return float("nan"), 0.0, np.array([]), np.array([])
    pred, true = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    f1 = f1_score(true, pred, average="macro", zero_division=0, labels=list(range(cfg.n_classes)))
    return float(np.nanmean(losses)), f1, pred, true


def run_fold(cfg, heldout, segments, unlabeled, store, out_root=None):
    source_names = [h["name"] for h in cfg.homes if h["name"] != heldout]
    out_root = Path(out_root) if out_root is not None else Path(cfg.output_dir) / cfg.ablation
    out_dir = out_root / f"heldout_{heldout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.torch_device
    print(f"\n{'=' * 70}\nLOHO fold  held-out: {heldout}\nsource homes: {source_names}\n{'=' * 70}")
    set_seed(cfg.seed)

    source = [s for s in segments if s["home"] != heldout]
    target = [s for s in segments if s["home"] == heldout]

    real, synthetic = adasyn_segments(source, store, cfg)
    train_set = real + synthetic

    domain_iter = None
    n_domains = len(source_names) + (cfg.domain_mode == "multi_source_target")
    if cfg.ablation != "baseline":
        target_items = []
        if cfg.domain_mode == "multi_source_target":
            target_items = [{"home": s["home"], "batch_no": s["batch_no"], "indices": s["indices"]}
                            for s in target]
            if cfg.use_unlabeled_target:
                target_items += unlabeled.get(heldout, [])
            print(f"  [domain] target windows: {len(target)} labeled-boundary + "
                  f"{len(target_items) - len(target)} unlabeled")
        domain_iter = build_domain_stream(cfg, source, source_names, target_items, store)

    cw = class_weights(train_set, cfg.classes, cfg.weight_cap).to(device)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, collate_fn=collate_segments,
                  num_workers=cfg.num_workers, worker_init_fn=seed_worker, generator=g)
    # The held-out loader shares the generator with the train loader (original protocol).
    train_loader = DataLoader(SegmentDataset(train_set, store), shuffle=True, **common)
    heldout_loader = DataLoader(SegmentDataset(target, store), shuffle=False, **common)

    model = MLPBiGRU(cfg).to(device)
    disc = build_discriminator(cfg, n_domains)
    params = list(model.parameters())
    if disc is not None:
        disc = disc.to(device)
        params += list(disc.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.sched_step, cfg.sched_gamma)

    n_disc = sum(p.numel() for p in disc.parameters()) if disc is not None else 0
    print(f"  model params={sum(p.numel() for p in model.parameters()):,} "
          f"discriminator params={n_disc:,} device={device}")
    print(f"  ablation={cfg.ablation} domain_mode={cfg.domain_mode} adasyn=+{len(synthetic)} "
          f"train={len(train_set)} held-out={len(target)}")
    if disc is not None:
        print(f"  domain chance accuracy ~ {1.0 / n_domains:.3f}")

    history = {k: [] for k in ("train_loss", "ce", "domain", "supcon", "domain_acc",
                               "heldout_loss", "heldout_f1")}
    supcon_iter = None
    print(f"\n{'ep':>4} {'train':>8} {'held':>8} {'F1':>7}  details")
    for epoch in range(1, cfg.epochs + 1):
        if cfg.use_supcon and epoch == cfg.supcon_start_epoch and supcon_iter is None:
            print(f"  [select] scoring source segments with the epoch-{epoch - 1} model")
            selected, audit = select_segments(model, source, store, cfg)
            audit.to_csv(out_dir / "selection_audit.csv", index=False)
            supcon_iter = build_supcon_stream(cfg, selected, store)

        lam_d = domain_lambda(epoch, cfg) if domain_iter is not None else 0.0
        lam_s = cfg.supcon_weight if supcon_iter is not None else 0.0
        tr = train_one_epoch(model, disc, train_loader, optimizer, cw, cfg,
                             lam_d, domain_iter, lam_s, supcon_iter)

        if cfg.track_heldout:
            held_loss, held_f1, _, _ = evaluate(model, heldout_loader, cw, cfg)
        else:
            held_loss, held_f1 = float("nan"), float("nan")
        scheduler.step()

        for key, val in (("train_loss", tr["total"]), ("ce", tr["ce"]), ("domain", tr["domain"]),
                         ("supcon", tr["supcon"]), ("domain_acc", tr["domain_acc"]),
                         ("heldout_loss", held_loss), ("heldout_f1", held_f1)):
            history[key].append(val)

        details = f"ce={tr['ce']:.4f}"
        if disc is not None:
            details += f" dom={tr['domain']:.4f} lam_d={lam_d:.3f} acc={tr['domain_acc']:.3f}"
        details += f" supcon={tr['supcon']:.4f} lam_s={lam_s:.2f} anchors={tr['anchors']}"
        print(f"{epoch:>4} {tr['total']:>8.4f} {held_loss:>8.4f} {held_f1:>7.4f}  {details}")

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    if disc is not None:
        torch.save(disc.state_dict(), out_dir / "discriminator_final.pt")
    pd.DataFrame(history).rename_axis("epoch").rename(lambda i: i + 1).to_csv(out_dir / "history.csv")
    if cfg.track_heldout and cfg.save_plots:
        plot_curves(history, out_dir / "curves.png")

    test_loss, test_f1, pred, true = evaluate(model, heldout_loader, cw, cfg)
    labels = list(range(cfg.n_classes))
    report = classification_report(true, pred, labels=labels, target_names=cfg.classes,
                                   digits=4, zero_division=0)
    cm = confusion_matrix(true, pred, labels=labels)
    print(f"\nFINAL TEST ({heldout}, epoch {cfg.epochs})  loss={test_loss:.4f}  macro-F1={test_f1:.4f}")
    print(report)
    (out_dir / "classification_report.txt").write_text(report)
    pd.DataFrame(cm, index=cfg.classes, columns=cfg.classes).to_csv(out_dir / "confusion_matrix.csv")
    if cfg.save_plots:
        plot_confusion(cm, cfg.classes, f"held-out {heldout}", out_dir / "confusion_matrix.png")

    result = {"heldout": heldout, "macro_f1": round(float(test_f1), 4),
              "test_loss": round(float(test_loss), 4), "epochs": cfg.epochs}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
