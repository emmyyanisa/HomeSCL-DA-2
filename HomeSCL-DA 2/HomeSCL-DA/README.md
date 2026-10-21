# HomeSCL-DA

**Cross-home acoustic activity recognition with selective cross-home supervised contrastive learning and domain-adversarial training.**

HomeSCL-DA recognises daily activities from pre-extracted CLAP audio embeddings. It is evaluated with leave-one-home-out (LOHO): the model trains on all homes but one and is tested on the unseen home.

The training loss has three terms.

- Activity loss.
- Domain loss.
- Cross-home SupCon loss.

The held-out home's labels are never used for training or for model selection. The final-epoch model is the one tested.

## Repository structure

```
HomeSCL-DA/
├── main.py                  # HomeSCL-DA (SupCon + domain-adversarial)
├── main_mlp.py              # MLP only (no BiGRU, no DA)
├── main_ablation.py         # baseline / DANN / CDAN / CDAN-E, no SupCon
├── main_cda.py              # CDA (contrastive-adversarial DA) comparison
├── main_cat.py              # CAT (DANN + target-bank contrast) comparison
├── configs/
│   ├── default.json         # HomeSCL-DA
│   ├── mlp.json             # MLP only, same settings as ablation.json
│   ├── ablation.json        # = run_train_test_5_4_ablation_dann_balance_domain.py
│   ├── cda.json             # = run_train_test_5_4_CDA_2stage_balance_domain.py
│   └── cat.json             # = run_train_test_5_4_CAT_episode_clean.py
├── homescl_da/
│   ├── config.py            # Config dataclass, JSON loader, validation
│   ├── dataset.py           # TSV loading, segments, feature cache, datasets, samplers
│   ├── adasyn.py            # ADASYN oversampling
│   ├── model.py             # MLP-BiGRU, gradient reversal, domain discriminator
│   ├── losses.py            # class weights, weighted CE, cross-home SupCon
│   ├── selection.py         # classifier-guided selection for SupCon
│   ├── trainer.py           # training loop, evaluation, one LOHO fold
│   ├── mlp.py               # MLP-only model and fold
│   ├── cda.py               # CDA config, losses, schedules, fold
│   ├── cat.py               # CAT config, model, feature bank, fold
│   ├── runner.py            # shared CLI + LOHO loop for main_*.py
│   ├── reporting.py         # shared per-fold output files
│   └── utils.py             # seeding, plots
├── scripts/make_toy_data.py # random dataset for a quick sanity check
└── requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required. `matplotlib` is optional; without it, plots are skipped. `imbalanced-learn` provides ADASYN.

## Data format

One folder per home. Paths in the config are relative to the directory you run `main.py` from.

```
data/home_1/
├── labeled.tsv        # required
├── unlabeled.tsv      # optional, domain branch only
├── batch_index.tsv    # optional
└── features/
    ├── batch-0        # torch.save({"tensor": array [N, 1024]})
    ├── batch-1
    └── ...
```
**The dataset will be made publicly available upon publication.**

## How to run

**Quick sanity check.** This uses random data and takes about a minute on CPU.

```bash
python scripts/make_toy_data.py --out toy_data
python main.py --config toy_data/config.json --epochs 3 --heldout home_2
```

**Full LOHO run.** Edit `configs/default.json` to point at your data, then run:

```bash
python main.py --config configs/default.json
```

**Common variations:**

```bash
# single fold
python main.py --heldout home_6

# domain-adaptation ablations (the activity architecture is identical in all four)
python main.py --ablation baseline
python main.py --ablation dann
python main.py --ablation cdan
python main.py --ablation cdan_e

# domain definition
python main.py --domain-mode multi_source_target   # source homes + unlabeled held-out home
python main.py --domain-mode multi_home            # source homes only (no target audio)
python main.py --no-unlabeled                      # ignore unlabeled_tsv files

# HomeSCL without domain adaptation (SupCon kept, discriminator off)
python main.py --no-da

# runtime
python main.py --epochs 50 --seed 42 --device cuda:0 --num-workers 8 --output-dir runs --no-plots
```

Command-line flags override the matching keys in the JSON config. Every other hyper-parameter is set in the config file.

**Comparison methods.** Each has its own entry script and config, and writes the same output files as `main.py`.

```bash
python main_mlp.py                                 # MLP only: no BiGRU, no DA
python main_ablation.py --ablation cdan            # no SupCon; any of the four ablations
python main_cda.py                                 # CDA, 90 epochs, E'=50, E''=75
python main_cat.py                                 # CAT on DANN, label-free target windows
```

With the same seed and data, `main_ablation.py`, `main_cda.py` and `main_cat.py` reproduce their original single-file scripts exactly. `main_mlp.py` is new and uses the MLP-BiGRU baseline's settings.

| Method | Domain branch | Extra loss | Target audio |
|---|---|---|---|
| `main_mlp.py` | none (MLP + mean pooling, no BiGRU) | none | none |
| `main.py --no-da` | none | selective cross-home SupCon (from epoch 11) | none |
| `main.py` | DANN/CDAN/CDAN-E, balanced stream | selective cross-home SupCon (from epoch 11) | held-out audio + unlabeled_tsv |
| `main_ablation.py` | DANN/CDAN/CDAN-E, balanced stream | none | held-out audio + unlabeled_tsv |
| `main_cda.py` | DANN-style, balanced stream, AdamW | SupCL (Stage I), CrossCL with argmax pseudo-labels (Stage II) | unlabeled_tsv only |
| `main_cat.py` | DANN on the CE minibatch + target batch | K-closest / K-farthest target-bank contrast | label-free windows from batch_index_tsv |

## Outputs

```
outputs/<ablation>/                    # main.py
outputs/homescl_no_da/                 # main.py --no-da
outputs/mlp/                           # main_mlp.py
outputs/ablation/<ablation>/           # main_ablation.py
outputs/cda/<domain_mode>/             # main_cda.py
outputs/cat/                           # main_cat.py
├── config_used.json
├── summary.csv                       # macro-F1 per held-out home
└── heldout_<home>/
    ├── model_final.pt
    ├── discriminator_final.pt        # not written for baseline
    ├── history.csv                   # per-epoch losses, domain accuracy, held-out F1
    ├── selection_audit.csv           # main.py only: selection scores and decisions
    ├── classification_report.txt
    ├── confusion_matrix.csv / .png
    ├── curves.png
    └── result.json                   # main_cat.py also stores the auxiliary mlp_f1
```

The held-out F1 printed each epoch (`track_heldout`) is a diagnostic only. It never affects training or which model is reported.

## Reproducibility

Each fold re-seeds Python, NumPy and PyTorch and enables deterministic algorithms. On GPU, `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set automatically.

The training and held-out loaders share one random generator. As a result, setting `track_heldout` to false changes the shuffling order.

## License

Released for anonymous review. License information will be added upon publication.
