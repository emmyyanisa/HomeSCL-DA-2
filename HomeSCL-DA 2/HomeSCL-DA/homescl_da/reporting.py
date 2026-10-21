"""Per-fold output files, identical in layout for every method.

heldout_<home>/
    model_final.pt, discriminator_final.pt, history.csv, curves.png,
    classification_report.txt, confusion_matrix.csv / .png, result.json
"""
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .utils import plot_confusion, plot_curves


def save_fold_outputs(cfg, out_dir, heldout, model, disc, history,
                      test_loss, test_f1, pred, true, extra=None):
    """Write every fold artifact and return the result dict used in summary.csv."""
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    if disc is not None:
        torch.save(disc.state_dict(), out_dir / "discriminator_final.pt")
    pd.DataFrame(history).rename_axis("epoch").rename(lambda i: i + 1).to_csv(out_dir / "history.csv")
    if cfg.track_heldout and cfg.save_plots:
        plot_curves(history, out_dir / "curves.png")

    labels = list(range(cfg.n_classes))
    report = classification_report(true, pred, labels=labels, target_names=cfg.classes,
                                   digits=4, zero_division=0)
    cm = confusion_matrix(true, pred, labels=labels)
    print(f"\nFINAL TEST ({heldout}, epoch {cfg.epochs})  loss={test_loss:.4f}  macro-F1={test_f1:.4f}")
    print(report)
    (out_dir / "classification_report.txt").write_text(report)
    pd.DataFrame(cm, index=cfg.classes, columns=cfg.classes).to_csv(out_dir / "confusion_matrix.csv")
    if cfg.save_plots:
        plot_confusion(cm, cfg.classes, f"held-out {heldout}", out_dir / "confusion_matrix.png")

    result = {"heldout": heldout, "macro_f1": round(float(test_f1), 4),
              "test_loss": round(float(test_loss), 4), "epochs": cfg.epochs}
    result.update(extra or {})
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
