"""Generate a small random dataset in the expected format and a matching config.

    python scripts/make_toy_data.py --out toy_data
    python main.py --config toy_data/config.json --epochs 3 --heldout home_3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homescl_da.config import DEFAULT_CLASSES, Config  # noqa: E402


CLASS_PROBS = [0.15, 0.2, 0.3, 0.2, 0.05, 0.1]


def make_home(root, home, rng, class_means, n_sessions, steps, dim):
    feat_dir = root / home / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    offset = rng.normal(0, 0.5, dim)
    labeled, unlabeled, index_rows = [], [], []
    for s in range(n_sessions):
        bno = f"batch-{s}"
        x = rng.normal(0, 1, (steps, dim)).astype(np.float32) + offset
        t = 0
        while t < steps:
            length = int(rng.integers(5, 25))
            if rng.random() < 0.25:
                unlabeled += [(bno, i) for i in range(t, min(t + length, steps))]
            else:
                c = int(rng.choice(len(DEFAULT_CLASSES), p=CLASS_PROBS))
                for i in range(t, min(t + length, steps)):
                    x[i] += class_means[c]
                    labeled.append((bno, i, DEFAULT_CLASSES[c], round(i * 0.1, 1)))
            t += length
        index_rows += [(bno, i) for i in range(steps)]
        torch.save({"tensor": x}, feat_dir / bno)

    def write(name, header, rows):
        with open(root / home / name, "w") as f:
            f.write("\t".join(header) + "\n")
            f.writelines("\t".join(map(str, r)) + "\n" for r in rows)

    write("labeled.tsv", ["batch_no", "index", "event_label", "onset"], labeled)
    write("unlabeled.tsv", ["batch_no", "index"], unlabeled)
    write("batch_index.tsv", ["batch_no", "index"], index_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="toy_data")
    p.add_argument("--homes", type=int, default=4)
    p.add_argument("--sessions", type=int, default=12)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dim = Config().clap_dim
    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    class_means = rng.normal(0, 0.15, (len(DEFAULT_CLASSES), dim)).astype(np.float32)
    homes = []
    for i in range(1, args.homes + 1):
        name = f"home_{i}"
        make_home(root, name, rng, class_means, args.sessions, args.steps, dim)
        homes.append({
            "name": name,
            "labeled_tsv": str(root / name / "labeled.tsv"),
            "unlabeled_tsv": str(root / name / "unlabeled.tsv"),
            "batch_index_tsv": str(root / name / "batch_index.tsv"),
            "feature_dir": str(root / name / "features"),
        })

    cfg = Config(homes=homes, output_dir=str(root / "outputs"), num_workers=0,
                 supcon_start_epoch=2, domain_warmup_epochs=0)
    (root / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"toy dataset -> {root}/  config -> {root / 'config.json'}")


if __name__ == "__main__":
    main()
