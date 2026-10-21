"""Shared command line and LOHO loop for every entry script (main_*.py).

Every method writes the same layout:

    <out_root>/config_used.json
    <out_root>/summary.csv
    <out_root>/heldout_<home>/...      (see reporting.py)
"""
import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from .config import ABLATIONS, DOMAIN_MODES
from .dataset import FeatureStore, build_segments, load_labeled_frames


def build_parser(description, default_config, with_ablation=True, with_unlabeled=True):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config)
    p.add_argument("--heldout", nargs="+", help="held-out home name(s); default: config / all")
    if with_ablation:
        p.add_argument("--ablation", choices=ABLATIONS)
    p.add_argument("--domain-mode", dest="domain_mode", choices=DOMAIN_MODES)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:0 ...")
    if with_unlabeled:
        p.add_argument("--no-unlabeled", dest="use_unlabeled_target",
                       action="store_const", const=False)
    p.add_argument("--no-plots", dest="save_plots", action="store_const", const=False)
    return p


def run_loho(cfg, fold_fn, out_root, title, target_loader=None):
    """Load data once, run every fold and write summary.csv.

    fold_fn(cfg, heldout, segments, target_data, store, out_root) -> result dict
    target_loader(home_cfg, home_id, cfg) -> list of target windows (optional)
    """
    warnings.filterwarnings("ignore")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print(f"device={cfg.torch_device} classes={cfg.n_classes} homes={len(cfg.homes)}")
    print("Loading labeled data...")
    segments = build_segments(load_labeled_frames(cfg), cfg)
    store = FeatureStore({h["name"]: h["feature_dir"] for h in cfg.homes}, cfg.cache_size)

    folds = cfg.heldout or [h["name"] for h in cfg.homes]
    target_data = {}
    if target_loader is not None:
        for home_id, h in enumerate(cfg.homes):
            if h["name"] in folds:
                target_data[h["name"]] = target_loader(h, home_id, cfg)

    results = [fold_fn(cfg, name, segments, target_data, store, out_root) for name in folds]

    summary = pd.DataFrame(results)
    summary.to_csv(out_root / "summary.csv", index=False)
    print(f"\n{'=' * 50}\nLOHO summary  {title}\n{'=' * 50}")
    for r in results:
        print(f"{r['heldout']:<32}{r['macro_f1']:>9.4f}")
    print(f"{'MEAN':<32}{np.mean(summary['macro_f1']):>9.4f}")
    print(f"results -> {out_root}")
    return results
