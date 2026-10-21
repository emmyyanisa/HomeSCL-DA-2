import json
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

import torch

ABLATIONS = ("baseline", "dann", "cdan", "cdan_e")
DOMAIN_MODES = ("multi_source_target", "multi_home")
DEFAULT_CLASSES = ["cooking", "eating", "going_out", "go_to_toilet", "taking_bath", "washing"]


@dataclass
class Config:
    # Data. Each home: {"name", "labeled_tsv", "feature_dir",
    #                   optional "unlabeled_tsv", optional "batch_index_tsv"}
    homes: List[dict] = field(default_factory=list)
    classes: List[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    heldout: Optional[List[str]] = None          # None = run every LOHO fold
    output_dir: str = "outputs"

    # Segment construction
    dedup_chains: bool = True                    # collapse consecutive-session repeats
    dedup_keep: str = "longest"                  # "longest" | "first"
    dedup_classes: Optional[List[str]] = None    # None = all classes

    # Model
    clap_dim: int = 1024
    mlp_hidden1: int = 256
    mlp_hidden2: int = 64
    gru_hidden: int = 128
    gru_layers: int = 2
    dropout: float = 0.5

    # Optimisation
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    sched_step: int = 50
    sched_gamma: float = 0.5
    grad_clip: float = 1.0
    weight_cap: float = 5.0

    # ADASYN class balancing
    adasyn_k: int = 5
    adasyn_multiplier: float = 1.0

    # Domain-adversarial branch
    ablation: str = "dann"
    domain_mode: str = "multi_source_target"
    use_unlabeled_target: bool = True
    domain_hidden: int = 64
    domain_dropout: float = 0.3                  # discriminator dropout (CDA script used 0.5)
    domain_lambda_max: float = 1.0
    domain_lambda_cap: float = 0.2
    domain_warmup_epochs: int = 5
    domain_samples_per_home: int = 10
    domain_min_batches: int = 8
    domain_window_len: int = 50
    domain_window_min_len: int = 5

    # Selective cross-home supervised contrastive branch
    use_supcon: bool = True                      # False = no projector, no selection, no SupCon
    supcon_start_epoch: int = 11
    supcon_weight: float = 0.2
    supcon_temperature: float = 0.10
    supcon_proj_dim: int = 64
    select_top_frac: float = 0.10
    select_min_per_cell: int = 2
    select_require_match: bool = True
    supcon_homes_per_class: int = 3
    supcon_samples_per_home: int = 2
    supcon_min_batches: int = 8

    # Runtime
    seed: int = 42
    num_workers: int = 4
    cache_size: int = 4
    track_heldout: bool = True
    save_plots: bool = True
    device: str = "auto"

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def feat_dim(self):
        return 2 * self.gru_hidden

    @property
    def torch_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def validate(self):
        def check(value, allowed, name):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")

        check(self.ablation, ABLATIONS, "ablation")
        check(self.domain_mode, DOMAIN_MODES, "domain_mode")
        check(self.dedup_keep, ("longest", "first"), "dedup_keep")

        if len(self.homes) < 2:
            raise ValueError("At least two homes are required for leave-one-home-out.")
        names = [h.get("name") for h in self.homes]
        if len(set(names)) != len(names):
            raise ValueError(f"Home names must be unique: {names}")
        for h in self.homes:
            for key in ("name", "labeled_tsv", "feature_dir"):
                if not h.get(key):
                    raise ValueError(f"Home entry {h} is missing '{key}'.")
        if self.ablation != "baseline" and self.domain_mode == "multi_home" and len(self.homes) < 3:
            raise ValueError("domain_mode='multi_home' needs at least two source homes.")
        for name in self.heldout or []:
            if name not in names:
                raise ValueError(f"Unknown held-out home {name!r}; known: {names}")
        for c in self.dedup_classes or []:
            if c not in self.classes:
                raise ValueError(f"Unknown class in dedup_classes: {c!r}")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path, overrides=None, cls=Config):
    """Read a JSON config into ``cls`` (Config or a method-specific subclass)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    known = {f.name for f in fields(cls)}
    unknown = sorted(k for k in raw if k not in known and not k.startswith("_"))
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    cfg = cls(**{k: v for k, v in raw.items() if k in known})
    for key, value in (overrides or {}).items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg.validate()
