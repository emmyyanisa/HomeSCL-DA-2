"""CDA (contrastive-adversarial domain adaptation), leave-one-home-out.

Replaces run_train_test_5_4_CDA_2stage_balance_domain.py.

Example:
    python main_cda.py --config configs/cda.json
"""
from pathlib import Path

from homescl_da.cda import CDAConfig, run_fold
from homescl_da.config import load_config
from homescl_da.dataset import load_unlabeled_windows
from homescl_da.runner import build_parser, run_loho


def main():
    args = vars(build_parser("CDA leave-one-home-out", "configs/cda.json",
                             with_ablation=False, with_unlabeled=False).parse_args())
    cfg = load_config(args.pop("config"), args, cls=CDAConfig)
    run_loho(cfg, run_fold, Path(cfg.output_dir) / "cda" / cfg.domain_mode,
             f"CDA domain_mode={cfg.domain_mode}", load_unlabeled_windows)


if __name__ == "__main__":
    main()
