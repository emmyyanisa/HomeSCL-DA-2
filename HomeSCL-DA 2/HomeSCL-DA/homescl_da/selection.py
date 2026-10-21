"""Classifier-guided selection of reliable real source segments for SupCon.

Inside every (class, home) cell a segment is kept when
  (a) the current activity classifier predicts its annotated class,
  (b) it is in the top fraction by P(annotation | x), and
  (c) it is in the top fraction by cosine similarity to the same-class
      prototype built from OTHER source homes.
A small per-cell floor is filled only from prediction-matched segments.
Held-out labels and synthetic samples are never used.
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import SegmentDataset, collate_segments
from .utils import seed_worker


@torch.no_grad()
def score_segments(model, segments, store, cfg):
    loader = DataLoader(SegmentDataset(segments, store), batch_size=cfg.batch_size,
                        shuffle=False, collate_fn=collate_segments,
                        num_workers=cfg.num_workers, worker_init_fn=seed_worker)
    device = cfg.torch_device
    was_training = model.training
    model.eval()
    conf, pred, feat = [], [], []
    for x, y, mask, _ in loader:
        logits, f = model(x.to(device), mask.to(device))
        probs = F.softmax(logits, dim=-1)
        conf.append(probs.gather(1, y.to(device)[:, None]).squeeze(1).cpu().numpy())
        pred.append(probs.argmax(dim=1).cpu().numpy())
        feat.append(f.cpu().numpy())
    model.train(was_training)
    return np.concatenate(conf), np.concatenate(pred), np.concatenate(feat)


def cross_home_prototype_similarity(feat, labels, homes):
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    sim = np.full(len(labels), -1.0, dtype=np.float32)
    for i in range(len(labels)):
        ids = np.flatnonzero((labels == labels[i]) & (homes != homes[i]))
        if len(ids) == 0:
            ids = np.flatnonzero((labels == labels[i]) & (np.arange(len(labels)) != i))
        if len(ids) == 0:
            continue
        proto = feat[ids].mean(axis=0)
        proto /= max(float(np.linalg.norm(proto)), 1e-12)
        sim[i] = float(feat[i] @ proto)
    return sim


def _fill_floor(chosen, matched, conf, proto, min_keep):
    conf_rank = {int(i): r for r, i in enumerate(matched[np.argsort(conf[matched], kind="stable")])}
    proto_rank = {int(i): r for r, i in enumerate(matched[np.argsort(proto[matched], kind="stable")])}
    denom = max(len(matched) - 1, 1)
    ranked = sorted(matched.tolist(),
                    key=lambda i: conf_rank[i] / denom + proto_rank[i] / denom, reverse=True)
    out = set(int(i) for i in chosen)
    for i in ranked:
        out.add(int(i))
        if len(out) >= min_keep:
            break
    return np.asarray(sorted(out), dtype=np.int64)


def select_segments(model, segments, store, cfg):
    """Return (selected_segments, audit DataFrame)."""
    if not segments:
        return [], pd.DataFrame()
    labels = np.array([s["label"] for s in segments], dtype=np.int64)
    homes = np.array([s["home_id"] for s in segments], dtype=np.int64)
    conf, pred, feat = score_segments(model, segments, store, cfg)
    proto = cross_home_prototype_similarity(feat, labels, homes)
    match = pred == labels

    keep = np.zeros(len(segments), dtype=bool)
    home_order = list(dict.fromkeys(homes.tolist()))
    table = {}
    for c in range(cfg.n_classes):
        for h in home_order:
            ids = np.flatnonzero((labels == c) & (homes == h))
            if len(ids) == 0:
                continue
            matched = ids[match[ids]] if cfg.select_require_match else ids
            k = max(1, math.ceil(cfg.select_top_frac * len(ids)))
            top_conf = ids[np.argsort(conf[ids], kind="stable")[-k:]]
            top_proto = ids[np.argsort(proto[ids], kind="stable")[-k:]]
            strict = np.intersect1d(top_conf, top_proto)
            if cfg.select_require_match:
                strict = strict[match[strict]]
            chosen = strict
            min_keep = min(cfg.select_min_per_cell, len(matched))
            if len(chosen) < min_keep:
                chosen = _fill_floor(chosen, matched, conf, proto, min_keep)
            keep[chosen] = True
            table[(c, h)] = (len(chosen), len(matched), len(ids))

    selected = [s for s, k in zip(segments, keep) if k]
    print(f"  [select] classifier agreement={match.mean():.1%}  "
          f"selected {len(selected)}/{len(segments)}  top_frac={cfg.select_top_frac}")
    names = {s["home_id"]: s["home"] for s in segments}
    for c, cname in enumerate(cfg.classes):
        cells = [f"{names[h]}={'/'.join(map(str, table.get((c, h), (0, 0, 0))))}" for h in home_order]
        print(f"    {cname:<16}(selected/matched/total) " + "  ".join(cells))

    audit = pd.DataFrame({
        "sid": [s["sid"] for s in segments],
        "home": [s["home"] for s in segments],
        "class": [cfg.classes[c] for c in labels],
        "pred_class": [cfg.classes[p] for p in pred],
        "label_confidence": conf,
        "pred_matches_label": match,
        "cross_home_proto_cos": proto,
        "selected": keep,
    })
    return selected, audit
