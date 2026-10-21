"""CAT (contrastive adversarial training) on DANN, leave-one-home-out.

Replaces run_train_test_5_4_CAT_episode_clean.py.

Example:
    python main_cat.py --config configs/cat.json
"""
from pathlib import Path

from homescl_da.cat import CATConfig, load_label_free_windows, run_fold
from homescl_da.config import load_config
from homescl_da.runner import build_parser, run_loho


def main():
    args = vars(build_parser("CAT (DANN + target-bank contrast) leave-one-home-out",
                             "configs/cat.json", with_ablation=False,
                             with_unlabeled=False).parse_args())
    cfg = load_config(args.pop("config"), args, cls=CATConfig)
    run_loho(cfg, run_fold, Path(cfg.output_dir) / "cat", "CAT (DANN + CAT)",
             load_label_free_windows)


if __name__ == "__main__":
    main()
