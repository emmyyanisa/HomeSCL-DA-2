"""Seeding and optional plotting helpers."""
import os
import random

import numpy as np
import torch


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


def seed_worker(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  matplotlib not installed; plots skipped")
        return None


def plot_curves(history, path):
    plt = _pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    if any(np.isfinite(history["heldout_loss"])):
        axes[0].plot(history["heldout_loss"], label="held-out")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["heldout_f1"])
    axes[1].set_title("Held-out macro-F1 (diagnostic)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, classes, title, path):
    plt = _pyplot()
    if plt is None:
        return
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, mat, fmt, name in ((axes[0], cm, "d", "counts"), (axes[1], norm, ".2f", "normalised")):
        ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title} ({name})")
        thresh = mat.max() / 2 if mat.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
