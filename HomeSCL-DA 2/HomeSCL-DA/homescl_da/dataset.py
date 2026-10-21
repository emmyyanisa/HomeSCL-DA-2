"""Data loading, activity-segment construction, datasets and batch samplers.

Expected files per home (see README):
  labeled_tsv      columns: batch_no, index, event_label, onset
  unlabeled_tsv    columns: batch_no, index                       (optional)
  batch_index_tsv  columns: batch_no, index                       (optional)
  feature_dir/<batch_no>  torch file: {"tensor": [N, clap_dim]}, row = index
"""
import collections
import math
import random
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

LABELED_COLUMNS = ["batch_no", "index", "event_label", "onset"]


# ------------------------------------------------------------------ loading
def load_labeled_frames(cfg):
    """Read every home's labeled TSV; keep the later onset for multi-label steps."""
    class_to_idx = {c: i for i, c in enumerate(cfg.classes)}
    frames = []
    for home_id, h in enumerate(cfg.homes):
        df = pd.read_csv(h["labeled_tsv"], sep="\t")
        missing = set(LABELED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{h['labeled_tsv']} is missing columns {sorted(missing)}")
        df = df[LABELED_COLUMNS].copy()
        df["batch_no"] = df["batch_no"].astype(str)
        df["home"] = h["name"]
        df["home_id"] = home_id
        frames.append(df)
        print(f"  [{h['name']}] rows={len(df):,} sessions={df['batch_no'].nunique()}")

    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.sort_values(["home", "batch_no", "index", "onset"], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=["home", "batch_no", "index"], keep="first")
    df = df.sort_values(["home", "batch_no", "index"]).reset_index(drop=True)
    if before != len(df):
        print(f"  removed {before - len(df)} multi-label rows")
    df["label"] = df["event_label"].map(class_to_idx).fillna(-1).astype(int)
    return df


def build_segments(df, cfg):
    """One sample = one contiguous run of steps with the same label."""
    df = df.sort_values(["home", "batch_no", "index"]).reset_index(drop=True)
    grp = df.groupby(["home", "batch_no"], sort=False)
    new_run = (df["label"] != grp["label"].shift()) | (df["index"] - grp["index"].shift() != 1)
    df["run"] = new_run.cumsum()

    segments = []
    for (home, bno, _), sub in df.groupby(["home", "batch_no", "run"], sort=False):
        label = int(sub["label"].iloc[0])
        if label < 0:
            continue
        idx = sub["index"].to_numpy(dtype=np.int64)
        segments.append({
            "home": home,
            "home_id": int(sub["home_id"].iloc[0]),
            "batch_no": bno,
            "label": label,
            "indices": idx,
            "sid": f"{home}|{bno}|{label}|{idx[0]}-{idx[-1]}",
        })

    lengths = [len(s["indices"]) for s in segments]
    if lengths:
        print(f"  built {len(segments)} labeled segments "
              f"(len min/median/max = {min(lengths)}/{int(np.median(lengths))}/{max(lengths)})")
    if cfg.dedup_chains:
        segments = dedup_session_chains(segments, cfg)
    return segments


def dedup_session_chains(segments, cfg):
    """Keep one segment per chain of consecutive session files (same home and label).

    An activity crossing a session boundary appears once per file; these are
    repeated views of one event rather than independent samples.
    """
    def session_number(bno):
        m = re.search(r"(\d+)", str(bno))
        return int(m.group(1)) if m else -1

    active = (set(range(cfg.n_classes)) if cfg.dedup_classes is None
              else {cfg.classes.index(c) for c in cfg.dedup_classes})
    groups, kept = defaultdict(list), []
    for s in segments:
        if s["label"] in active:
            groups[(s["home"], s["label"])].append(s)
        else:
            kept.append(s)

    for items in groups.values():
        items = sorted(items, key=lambda s: session_number(s["batch_no"]))
        chains, chain = [], [items[0]]
        for a, b in zip(items, items[1:]):
            if session_number(b["batch_no"]) == session_number(a["batch_no"]) + 1:
                chain.append(b)
            else:
                chains.append(chain)
                chain = [b]
        chains.append(chain)
        for c in chains:
            if len(c) == 1 or cfg.dedup_keep == "first":
                kept.append(c[0])
            else:
                kept.append(max(c, key=lambda s: len(s["indices"])))

    before = collections.Counter(s["label"] for s in segments)
    after = collections.Counter(s["label"] for s in kept)
    print(f"  dedup ({cfg.dedup_keep}): {len(segments)} -> {len(kept)} segments")
    for c, name in enumerate(cfg.classes):
        b, a = before.get(c, 0), after.get(c, 0)
        print(f"    {name:<16}{b:>6} -> {a:<6}" + (f"({b / a:.1f}x)" if a else ""))
    return kept


def load_unlabeled_windows(home_cfg, home_id, cfg):
    """Fixed-length windows of unlabeled audio for the domain branch only."""
    path = home_cfg.get("unlabeled_tsv")
    if not path:
        return []
    if not Path(path).exists():
        print(f"  [unlabeled] {home_cfg['name']}: '{path}' not found, skipped")
        return []
    df = pd.read_csv(path, sep="\t")
    if not {"batch_no", "index"} <= set(df.columns):
        print(f"  [unlabeled] {home_cfg['name']}: needs columns batch_no, index; skipped")
        return []
    df = df[["batch_no", "index"]].astype({"batch_no": str})

    ref_path = home_cfg.get("batch_index_tsv")
    if ref_path:
        ref = pd.read_csv(ref_path, sep="\t", usecols=["batch_no", "index"])
        ref = ref.astype({"batch_no": str}).drop_duplicates()
        df = df.merge(ref, on=["batch_no", "index"], how="inner")

    df = df.sort_values(["batch_no", "index"]).reset_index(drop=True)
    gap = df["index"] - df.groupby("batch_no", sort=False)["index"].shift() != 1
    df["run"] = gap.cumsum()

    windows = []
    for (bno, _), sub in df.groupby(["batch_no", "run"], sort=False):
        idx = sub["index"].to_numpy(dtype=np.int64)
        for start in range(0, len(idx), cfg.domain_window_len):
            chunk = idx[start:start + cfg.domain_window_len]
            if len(chunk) >= cfg.domain_window_min_len:
                windows.append({"home": home_cfg["name"], "home_id": home_id,
                                "batch_no": bno, "indices": chunk})
    print(f"  [unlabeled] {home_cfg['name']}: {len(df):,} rows -> {len(windows)} windows")
    return windows


# ------------------------------------------------------------ feature store
class FeatureStore:
    """Small LRU cache of per-session CLAP matrices (one file per session)."""

    def __init__(self, feature_dirs, cache_size=4):
        self.feature_dirs = dict(feature_dirs)
        self.cache_size = cache_size
        self._cache = OrderedDict()

    def session(self, home, batch_no):
        key = (home, batch_no)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = Path(self.feature_dirs[home]) / str(batch_no)
        if not path.exists():
            raise FileNotFoundError(path)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        arr = obj["tensor"] if isinstance(obj, dict) else obj
        # Keep the stored dtype: datasets cast to float32 per item, and ADASYN
        # averages in the native dtype first, exactly like the original scripts.
        arr = arr.numpy() if torch.is_tensor(arr) else np.asarray(arr)
        self._cache[key] = arr
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return arr

    def features(self, item):
        return self.session(item["home"], item["batch_no"])[item["indices"]]


# ----------------------------------------------------------------- datasets
class SegmentDataset(Dataset):
    """Returns (features [T, D], label, home_id). Synthetic samples have T = 1."""

    def __init__(self, segments, store):
        self.segments = segments
        self.store = store

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, i):
        s = self.segments[i]
        x = s["vec"][None] if s.get("synthetic") else self.store.features(s)
        return torch.tensor(x, dtype=torch.float32), s["label"], s["home_id"]


class DomainDataset(Dataset):
    """Returns (features [T, D], domain_id). No activity label is exposed."""

    def __init__(self, items, store):
        self.items = items
        self.store = store

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        return torch.tensor(self.store.features(item), dtype=torch.float32), item["domain_id"]


def _pad(seqs):
    lengths = torch.tensor([len(s) for s in seqs])
    x = pad_sequence(list(seqs), batch_first=True)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return x, mask


def collate_segments(batch):
    xs, labels, homes = zip(*batch)
    x, mask = _pad(xs)
    return x, torch.tensor(labels, dtype=torch.long), mask, torch.tensor(homes, dtype=torch.long)


def collate_domain(batch):
    xs, domains = zip(*batch)
    x, mask = _pad(xs)
    return x, mask, torch.tensor(domains, dtype=torch.long)


def cycle(loader):
    while True:
        yield from loader


# ----------------------------------------------------------------- samplers
class BalancedDomainBatchSampler:
    """Every batch holds a fixed quota from each domain group (source homes, target)."""

    def __init__(self, groups, quotas, min_batches=8, seed=0):
        self.quotas = OrderedDict(quotas)
        groups = np.asarray(groups, dtype=object)
        self.pools = OrderedDict()
        for g in self.quotas:
            pool = np.flatnonzero(groups == g)
            if len(pool) == 0:
                raise ValueError(f"No domain windows for group {g!r}")
            self.pools[g] = pool
        coverage = [math.ceil(len(self.pools[g]) / max(q, 1)) for g, q in self.quotas.items()]
        self.n_batches = max(int(min_batches), max(coverage))
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for g, q in self.quotas.items():
                pool = self.pools[g]
                batch.extend(int(i) for i in self.rng.choice(pool, size=q, replace=len(pool) < q))
            self.rng.shuffle(batch)
            yield batch


class CrossHomeClassBatchSampler:
    """Class x home balanced batches so every class has positives from other homes."""

    def __init__(self, segments, n_classes, homes_per_class=3, samples_per_home=2,
                 min_batches=8, seed=0):
        self.n_classes = n_classes
        self.homes_per_class = homes_per_class
        self.samples_per_home = samples_per_home
        self.seed = seed
        self.epoch = 0
        self.buckets = defaultdict(lambda: defaultdict(list))
        for i, s in enumerate(segments):
            self.buckets[s["label"]][s["home_id"]].append(i)
        per_batch = sum(min(homes_per_class, len(self.buckets[c])) * samples_per_home
                        for c in range(n_classes)
                        if min(homes_per_class, len(self.buckets[c])) >= 2)
        self.n_batches = max(int(min_batches), math.ceil(len(segments) / max(per_batch, 1)))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.n_batches):
            batch = []
            for c in range(self.n_classes):
                homes = [h for h, ids in self.buckets[c].items() if ids]
                if len(homes) < 2:
                    continue
                for h in rng.sample(homes, min(self.homes_per_class, len(homes))):
                    batch.extend(rng.choice(self.buckets[c][h]) for _ in range(self.samples_per_home))
            rng.shuffle(batch)
            if len(batch) >= 2:
                yield batch
