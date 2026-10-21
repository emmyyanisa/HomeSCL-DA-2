"""CAT: contrastive adversarial training on the MLP-BiGRU encoder.

Port of run_train_test_5_4_CAT_episode_clean.py
(Chen et al., arXiv:2407.12782, adapted to CLAP episodes).

  Loss = mlp_ce_weight * CE(MLP head) + gru_ce_weight * CE(BiGRU head)
       + (S+1)-way domain CE via GRL         (source homes + held-out target)
       + cat_weight * L_CAT                  (K closest - K farthest target-bank features)

* Target windows are built from batch_index_tsv only (fixed-length, split at
  index gaps), so no target label or label-derived boundary is used.
* The domain branch draws its target batch from that stream and uses the
  same source minibatch as CE (real source only; ADASYN samples excluded).
* CAT is computed inside the domain branch, so it is active only once the
  domain lambda is > 0 (after domain_warmup_epochs) -- as in the original.
* The order of every random draw matches the original script, so the same
  seed gives the same run.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from .adasyn import adasyn_segments
from .config import Config
from .dataset import SegmentDataset, _pad, collate_segments, cycle
from .losses import class_weights, weighted_ce
from .model import DomainDiscriminator, grad_reverse, masked_mean
from .reporting import save_fold_outputs
from .trainer import domain_lambda
from .utils import seed_worker, set_seed


@dataclass
class CATConfig(Config):
    # Defaults reproduce run_train_test_5_4_CAT_episode_clean.py
    seed: int = 189
    sched_step: int = 30
    use_supcon: bool = False
    ablation: str = "dann"
    domain_mode: str = "multi_source_target"

    cat_k: int = 5
    cat_weight: float = 1.0
    cat_start_epoch: int = 0
    cat_bank_momentum: float = 0.0
    cat_normalize_l2: bool = True
    cat_refresh_bank_at_start: bool = True
    cat_window_len: int = 50
    cat_window_min_len: int = 5
    domain_batch_size: int = 64
    mlp_ce_weight: float = 0.0           # ALPHA in the original script
    gru_ce_weight: float = 1.0           # BETA in the original script

    def validate(self):
        super().validate()
        if self.ablation != "dann":
            raise ValueError("CAT is attached to DANN; set ablation='dann'.")
        if self.domain_mode != "multi_source_target":
            raise ValueError("CAT needs target audio; set domain_mode='multi_source_target'.")
        if self.use_supcon:
            raise ValueError("CAT replaces SupCon; set use_supcon=false.")
        for name in self.heldout or [h["name"] for h in self.homes]:
            h = next(x for x in self.homes if x["name"] == name)
            if not h.get("batch_index_tsv"):
                raise ValueError(f"CAT builds target windows from batch_index_tsv; "
                                 f"home {name!r} has none.")
        return self


# -------------------------------------------------------------------- model
class CATBiGRU(nn.Module):
    """MLP-BiGRU with the auxiliary MLP head of the original CAT script.

    Layers are created in the original order (backbone, MLP head, BiGRU,
    BiGRU head) so the random initialisation is identical.
    """

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
        self.mlp_head = nn.Linear(cfg.mlp_hidden2, cfg.n_classes)
        self.bigru = nn.GRU(
            input_size=cfg.mlp_hidden2,
            hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(cfg.feat_dim, cfg.n_classes)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        feat = self.backbone(x)
        out, _ = self.bigru(feat)
        mlp_logits = self.mlp_head(masked_mean(feat, mask))
        pooled = masked_mean(out, mask)
        return mlp_logits, self.classifier(pooled), pooled


# --------------------------------------------------------------- target data
def load_label_free_windows(home_cfg, home_id, cfg):
    """Fixed windows from batch_index_tsv, split only at index gaps."""
    raw = pd.read_csv(home_cfg["batch_index_tsv"], sep="\t")
    missing = {"batch_no", "index"} - set(raw.columns)
    if missing:
        raise ValueError(f"{home_cfg['batch_index_tsv']} is missing columns {sorted(missing)}")
    raw["batch_no"] = raw["batch_no"].astype(str)
    items = []
    for bno, grp in raw.groupby("batch_no", sort=False):
        idx = np.asarray(sorted(grp["index"].astype(int).unique()), dtype=np.int64)
        if idx.size == 0:
            continue
        for run in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            for start in range(0, len(run), cfg.cat_window_len):
                chunk = run[start:start + cfg.cat_window_len]
                if len(chunk) >= cfg.cat_window_min_len:
                    items.append({"home": home_cfg["name"], "home_id": home_id,
                                  "batch_no": bno, "indices": chunk.astype(np.int64)})
    print(f"  [CAT] {home_cfg['name']}: {len(items)} label-free target windows "
          f"(window={cfg.cat_window_len}, min={cfg.cat_window_min_len})")
    return items


class BankDataset(Dataset):
    """Target windows plus a stable feature-bank index. No label is exposed."""

    def __init__(self, items, store):
        self.items = items
        self.store = store

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return torch.tensor(self.store.features(self.items[i]), dtype=torch.float32), i


def collate_bank(batch):
    xs, idx = zip(*batch)
    x, mask = _pad(xs)
    return x, mask, torch.tensor(idx, dtype=torch.long)


class FeatureBank:
    """Detached features for every target window, refreshed as windows are seen."""

    def __init__(self, n_items, dim, device, momentum=0.0):
        self.momentum = float(momentum)
        self.device = device
        self.features = torch.zeros(n_items, dim, dtype=torch.float32, device=device)
        self.valid = torch.zeros(n_items, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, idx, feats):
        idx = idx.to(self.device, dtype=torch.long)
        z = feats.detach().to(self.device, dtype=torch.float32)
        seen = self.valid[idx]
        if self.momentum > 0 and seen.any():
            old = idx[seen]
            self.features[old] = self.momentum * self.features[old] + (1 - self.momentum) * z[seen]
        self.features[idx[~seen]] = z[~seen]
        if self.momentum == 0 and seen.any():
            self.features[idx[seen]] = z[seen]
        self.valid[idx] = True

    @property
    def n_valid(self):
        return int(self.valid.sum().item())


@torch.no_grad()
def fill_bank(model, loader, bank, device):
    was_training = model.training
    model.eval()
    for x, mask, idx in loader:
        _, _, feat = model(x.to(device), mask.to(device))
        bank.update(idx, feat)
    model.train(was_training)
    print(f"  [CAT] target bank filled: {bank.n_valid}/{len(bank.valid)} entries")


def cat_loss(source_feat, bank, cfg):
    """Pull the K closest target-bank features, push the K farthest (L2)."""
    bank_feat = bank.features[bank.valid]
    if source_feat.numel() == 0 or bank_feat.shape[0] < 2:
        return None, 0, float("nan"), float("nan")
    k = min(cfg.cat_k, max(1, bank_feat.shape[0] // 2))
    sim = F.normalize(source_feat, dim=-1) @ F.normalize(bank_feat.detach(), dim=-1).T
    close_idx = torch.topk(sim, k=k, dim=1, largest=True).indices
    far_idx = torch.topk(sim, k=k, dim=1, largest=False).indices
    src, close, far = source_feat, bank_feat.detach()[close_idx], bank_feat.detach()[far_idx]
    if cfg.cat_normalize_l2:
        src, close, far = (F.normalize(t, dim=-1) for t in (src, close, far))
    d_close = torch.linalg.vector_norm(src.unsqueeze(1) - close, dim=-1)
    d_far = torch.linalg.vector_norm(src.unsqueeze(1) - far, dim=-1)
    loss = (d_close.sum(dim=1) - d_far.sum(dim=1)).mean()
    with torch.no_grad():
        close_sim = sim.gather(1, close_idx).mean().item()
        far_sim = sim.gather(1, far_idx).mean().item()
    return loss, k, close_sim, far_sim


# ------------------------------------------------------------------ training
def _ce_pair(mlp_logits, gru_logits, y, cw, cfg):
    lm = weighted_ce(mlp_logits, y, cw)
    lb = weighted_ce(gru_logits, y, cw)
    if torch.isnan(lm) or torch.isnan(lb):
        return None, lm, lb
    return cfg.mlp_ce_weight * lm + cfg.gru_ce_weight * lb, lm, lb


def train_one_epoch(model, disc, loader, optimizer, cw, cfg, lam_d, lam_cat,
                    target_iter, bank, local_of_global, target_domain):
    device = cfg.torch_device
    model.train()
    disc.train()
    totals = dict(total=0.0, mlp_ce=0.0, gru_ce=0.0, domain=0.0, cat=0.0)
    steps = d_correct = d_count = cat_batches = k_eff = 0
    close_sum = far_sum = 0.0

    for x, y, mask, home_ids in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        optimizer.zero_grad()
        mlp_o, gru_o, feat_src = model(x, mask)
        loss, lm, lb = _ce_pair(mlp_o, gru_o, y, cw, cfg)
        if loss is None:
            continue

        d_val = c_val = 0.0
        if lam_d > 0 and target_iter is not None:
            local = local_of_global[home_ids.clamp(min=0)].to(device)
            local[home_ids.to(device) < 0] = -1          # ADASYN samples have no home
            valid = local >= 0

            tx, tmask, tidx = next(target_iter)
            _, _, feat_tgt = model(tx.to(device), tmask.to(device))

            losses = []
            if valid.any():
                logit_src = disc(grad_reverse(feat_src[valid], lam_d))
                y_src = local[valid]
                losses.append(F.cross_entropy(logit_src, y_src))
                d_correct += (logit_src.argmax(1) == y_src).sum().item()
                d_count += y_src.numel()
            logit_tgt = disc(grad_reverse(feat_tgt, lam_d))
            y_tgt = torch.full((feat_tgt.shape[0],), target_domain,
                               dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logit_tgt, y_tgt))
            d_correct += (logit_tgt.argmax(1) == y_tgt).sum().item()
            d_count += y_tgt.numel()
            d_loss = torch.stack(losses).mean()
            loss = loss + d_loss
            d_val = d_loss.item()

            bank.update(tidx, feat_tgt)
            if lam_cat > 0 and valid.any():
                c_loss, k_eff, close_sim, far_sim = cat_loss(feat_src[valid], bank, cfg)
                if c_loss is not None and torch.isfinite(c_loss):
                    loss = loss + lam_cat * c_loss
                    c_val = c_loss.item()
                    close_sum += close_sim
                    far_sum += far_sim
                    cat_batches += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip)
        optimizer.step()

        totals["total"] += loss.item()
        totals["mlp_ce"] += lm.item()
        totals["gru_ce"] += lb.item()
        totals["domain"] += d_val
        totals["cat"] += c_val
        steps += 1

    out = {k: v / max(steps, 1) for k, v in totals.items()}
    out.update(domain_acc=d_correct / d_count if d_count else float("nan"), k=k_eff,
               close=close_sum / cat_batches if cat_batches else float("nan"),
               far=far_sum / cat_batches if cat_batches else float("nan"),
               bank=bank.n_valid if bank is not None else 0)
    return out


@torch.no_grad()
def evaluate(model, loader, cw, cfg):
    """Returns (loss, BiGRU macro-F1, BiGRU preds, labels, MLP macro-F1)."""
    device = cfg.torch_device
    model.eval()
    losses, pm, pg, ys = [], [], [], []
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        mlp_o, gru_o, _ = model(x, mask)
        loss, _, _ = _ce_pair(mlp_o, gru_o, y, cw, cfg)
        if loss is not None:
            losses.append(loss.item())
        pm.append(mlp_o.argmax(-1).cpu())
        pg.append(gru_o.argmax(-1).cpu())
        ys.append(y.cpu())
    if not ys:
        return float("nan"), 0.0, np.array([]), np.array([]), 0.0
    pm, pg, ys = torch.cat(pm).numpy(), torch.cat(pg).numpy(), torch.cat(ys).numpy()
    labels = list(range(cfg.n_classes))
    f1_gru = f1_score(ys, pg, average="macro", zero_division=0, labels=labels)
    f1_mlp = f1_score(ys, pm, average="macro", zero_division=0, labels=labels)
    loss = float(np.mean(losses)) if losses else float("nan")
    return loss, f1_gru, pg, ys, f1_mlp


def run_fold(cfg, heldout, segments, target_windows, store, out_root):
    source_names = [h["name"] for h in cfg.homes if h["name"] != heldout]
    out_dir = Path(out_root) / f"heldout_{heldout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.torch_device
    print(f"\n{'=' * 70}\nCAT LOHO fold  held-out: {heldout}\nsource homes: {source_names}\n{'=' * 70}")
    set_seed(cfg.seed)

    source = [s for s in segments if s["home"] != heldout]
    target = [s for s in segments if s["home"] == heldout]

    real, synthetic = adasyn_segments(source, store, cfg)
    train_set = real + synthetic

    # global home id -> source-domain id (0..S-1); held-out home maps to -1.
    names = [h["name"] for h in cfg.homes]
    local_of_global = torch.full((len(names),), -1, dtype=torch.long)
    for i, name in enumerate(source_names):
        local_of_global[names.index(name)] = i

    items = target_windows.get(heldout, [])
    target_iter = bank = bank_loader = None
    if items:
        bank_ds = BankDataset(items, store)
        target_iter = cycle(DataLoader(bank_ds, batch_size=cfg.domain_batch_size, shuffle=True,
                                       collate_fn=collate_bank, num_workers=cfg.num_workers,
                                       worker_init_fn=seed_worker))
    else:
        print(f"  [CAT] no target windows for {heldout}; domain and CAT losses disabled")

    cw = class_weights(train_set, cfg.classes, cfg.weight_cap).to(device)
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    common = dict(batch_size=cfg.batch_size, collate_fn=collate_segments,
                  num_workers=cfg.num_workers, worker_init_fn=seed_worker, generator=g)
    train_loader = DataLoader(SegmentDataset(train_set, store), shuffle=True, **common)
    heldout_loader = DataLoader(SegmentDataset(target, store), shuffle=False, **common)

    model = CATBiGRU(cfg).to(device)
    if items:
        bank = FeatureBank(len(items), cfg.feat_dim, device, cfg.cat_bank_momentum)
        bank_loader = DataLoader(bank_ds, batch_size=cfg.domain_batch_size, shuffle=False,
                                 collate_fn=collate_bank, num_workers=cfg.num_workers,
                                 worker_init_fn=seed_worker)
    n_domains = len(source_names) + 1
    disc = DomainDiscriminator(cfg.feat_dim, n_domains, cfg.domain_hidden,
                               cfg.domain_dropout).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(disc.parameters()),
                                 lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.sched_step, cfg.sched_gamma)

    print(f"  model params={sum(p.numel() for p in model.parameters()):,} "
          f"discriminator params={sum(p.numel() for p in disc.parameters()):,} device={device}")
    print(f"  adasyn=+{len(synthetic)} train={len(train_set)} held-out={len(target)} "
          f"target windows={len(items)}  domain chance accuracy ~ {1.0 / n_domains:.3f}")
    print(f"  CAT: K={cfg.cat_k} weight={cfg.cat_weight} start_ep={cfg.cat_start_epoch} "
          f"momentum={cfg.cat_bank_momentum}  CE = {cfg.mlp_ce_weight}*MLP + {cfg.gru_ce_weight}*BiGRU")

    history = {k: [] for k in ("train_loss", "mlp_ce", "gru_ce", "domain", "cat", "domain_acc",
                               "cat_close_sim", "cat_far_sim", "heldout_loss", "heldout_f1",
                               "heldout_mlp_f1")}
    print(f"\n{'ep':>4} {'train':>8} {'held':>8} {'F1':>7}  details")
    for epoch in range(1, cfg.epochs + 1):
        if (bank is not None and cfg.cat_refresh_bank_at_start
                and epoch == cfg.cat_start_epoch):
            fill_bank(model, bank_loader, bank, device)

        lam_d = domain_lambda(epoch, cfg)
        lam_cat = cfg.cat_weight if bank is not None and epoch >= cfg.cat_start_epoch else 0.0
        tr = train_one_epoch(model, disc, train_loader, optimizer, cw, cfg, lam_d, lam_cat,
                             target_iter, bank, local_of_global, len(source_names))
        if cfg.track_heldout:
            held_loss, held_f1, _, _, held_mlp = evaluate(model, heldout_loader, cw, cfg)
        else:
            held_loss = held_f1 = held_mlp = float("nan")
        scheduler.step()

        for key, val in (("train_loss", tr["total"]), ("mlp_ce", tr["mlp_ce"]),
                         ("gru_ce", tr["gru_ce"]), ("domain", tr["domain"]), ("cat", tr["cat"]),
                         ("domain_acc", tr["domain_acc"]), ("cat_close_sim", tr["close"]),
                         ("cat_far_sim", tr["far"]), ("heldout_loss", held_loss),
                         ("heldout_f1", held_f1), ("heldout_mlp_f1", held_mlp)):
            history[key].append(val)

        print(f"{epoch:>4} {tr['total']:>8.4f} {held_loss:>8.4f} {held_f1:>7.4f}  "
              f"ce={tr['gru_ce']:.4f} dom={tr['domain']:.4f} lam_d={lam_d:.3f} "
              f"acc={tr['domain_acc']:.3f} cat={tr['cat']:.4f} lam_cat={lam_cat:.2f} "
              f"K={tr['k']} close={tr['close']:.3f} far={tr['far']:.3f} bank={tr['bank']}")

    test_loss, test_f1, pred, true, test_mlp = evaluate(model, heldout_loader, cw, cfg)
    print(f"  MLP-head macro-F1 (auxiliary, not trained when mlp_ce_weight=0): {test_mlp:.4f}")
    return save_fold_outputs(cfg, out_dir, heldout, model, disc, history,
                             test_loss, test_f1, pred, true,
                             extra={"mlp_f1": round(float(test_mlp), 4)})
