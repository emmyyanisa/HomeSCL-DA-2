"""Leave-one-home-out training and evaluation for HomeSCL-DA.

Example:
    python main.py --config configs/default.json
    python main.py --config configs/default.json --heldout home_6 --ablation cdan_e
    python main.py --no-da        # HomeSCL without domain adaptation (SupCon only)
"""
import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from homescl_da.config import ABLATIONS, DOMAIN_MODES, load_config
from homescl_da.dataset import (FeatureStore, build_segments, load_labeled_frames,
                                load_unlabeled_windows)
from homescl_da.trainer import run_fold


def parse_args():
    p = argparse.ArgumentParser(description="HomeSCL-DA leave-one-home-out experiment")
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--heldout", nargs="+", help="held-out home name(s); default: config / all")
    p.add_argument("--ablation", choices=ABLATIONS)
    p.add_argument("--domain-mode", dest="domain_mode", choices=DOMAIN_MODES)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--num-workers", dest="num_workers", type=int)
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--no-unlabeled", dest="use_unlabeled_target", action="store_const", const=False)
    p.add_argument("--no-plots", dest="save_plots", action="store_const", const=False)
    p.add_argument("--no-da", dest="no_da", action="store_true",
                   help="HomeSCL without domain adaptation: SupCon kept, domain branch off")
    return p.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = vars(parse_args())
    no_da = args.pop("no_da")
    if no_da and args.get("ablation") not in (None, "baseline"):
        raise SystemExit("--no-da turns the domain branch off; do not combine it with --ablation.")
    cfg = load_config(args.pop("config"), args)
    if no_da:
        cfg.ablation = "baseline"       # no discriminator, no target audio; SupCon unchanged
    name = "homescl_no_da" if no_da else cfg.ablation
    out_root = Path(cfg.output_dir) / name
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config_used.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    print(f"device={cfg.torch_device} classes={cfg.n_classes} homes={len(cfg.homes)}")
    print("Loading labeled data...")
    segments = build_segments(load_labeled_frames(cfg), cfg)
    store = FeatureStore({h["name"]: h["feature_dir"] for h in cfg.homes}, cfg.cache_size)

    folds = cfg.heldout or [h["name"] for h in cfg.homes]
    unlabeled = {}
    if cfg.ablation != "baseline" and cfg.domain_mode == "multi_source_target" \
            and cfg.use_unlabeled_target:
        for home_id, h in enumerate(cfg.homes):
            if h["name"] in folds:
                unlabeled[h["name"]] = load_unlabeled_windows(h, home_id, cfg)

    results = [run_fold(cfg, home, segments, unlabeled, store, out_root) for home in folds]

    summary = pd.DataFrame(results)
    summary.to_csv(out_root / "summary.csv", index=False)
    title = "HomeSCL without DA" if no_da else f"ablation={cfg.ablation}"
    print(f"\n{'=' * 50}\nLOHO summary  {title}\n{'=' * 50}")
    for r in results:
        print(f"{r['heldout']:<32}{r['macro_f1']:>9.4f}")
    print(f"{'MEAN':<32}{np.mean(summary['macro_f1']):>9.4f}")
    print(f"results -> {out_root}")


if __name__ == "__main__":
    main()
