"""Baseline / DANN / CDAN / CDAN-E ablation without SupCon.

Replaces run_train_test_5_4_ablation_dann_balance_domain.py.

Example:
    python main_ablation.py --config configs/ablation.json --ablation cdan
"""
from pathlib import Path

from homescl_da.config import load_config
from homescl_da.dataset import load_unlabeled_windows
from homescl_da.runner import build_parser, run_loho
from homescl_da.trainer import run_fold


def main():
    args = vars(build_parser("Domain-adaptation ablation (no SupCon), leave-one-home-out",
                             "configs/ablation.json").parse_args())
    cfg = load_config(args.pop("config"), args)
    cfg.use_supcon = False                       # MLP-BiGRU + domain branch only
    cfg.output_dir = str(Path(cfg.output_dir) / "ablation")
    needs_target = (cfg.ablation != "baseline" and cfg.domain_mode == "multi_source_target"
                    and cfg.use_unlabeled_target)
    run_loho(cfg,
             lambda c, name, seg, tgt, store, root: run_fold(c, name, seg, tgt, store),
             Path(cfg.output_dir) / cfg.ablation,
             f"ablation={cfg.ablation} (no SupCon)",
             load_unlabeled_windows if needs_target else None)


if __name__ == "__main__":
    main()
