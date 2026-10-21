"""MLP-only baseline (no BiGRU, no domain adaptation), leave-one-home-out.

Example:
    python main_mlp.py --config configs/mlp.json
"""
from pathlib import Path

from homescl_da.config import load_config
from homescl_da.mlp import run_fold
from homescl_da.runner import build_parser, run_loho


def main():
    args = vars(build_parser("MLP-only baseline, leave-one-home-out", "configs/mlp.json",
                             with_ablation=False, with_unlabeled=False).parse_args())
    cfg = load_config(args.pop("config"), args)
    cfg.ablation, cfg.use_supcon = "baseline", False     # no domain branch, no SupCon
    run_loho(cfg, run_fold, Path(cfg.output_dir) / "mlp", "MLP only")


if __name__ == "__main__":
    main()
