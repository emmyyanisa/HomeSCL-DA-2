"""ADASYN oversampling of source segments in CLAP feature space."""
import collections

import numpy as np
from imblearn.over_sampling import ADASYN


def adasyn_segments(segments, store, cfg):
    """Return (real_segments, synthetic_segments)."""
    y = np.array([s["label"] for s in segments], dtype=np.int64)
    counts = collections.Counter(y.tolist())
    target = int(round(max(counts.values()) * cfg.adasyn_multiplier))
    strategy = {c: target for c, n in counts.items() if 2 <= n < target}
    if not strategy:
        print("  [adasyn] classes already balanced; nothing generated")
        return list(segments), []
    k = max(1, min(cfg.adasyn_k, min(counts[c] for c in strategy) - 1))

    X = np.stack([store.features(s).mean(axis=0) for s in segments]).astype(np.float32)
    try:
        sampler = ADASYN(sampling_strategy=strategy, random_state=cfg.seed, n_neighbors=k)
        X_res, y_res = sampler.fit_resample(X, y)
    except (ValueError, RuntimeError) as e:
        print(f"  [adasyn] failed ({e}); using original segments")
        return list(segments), []

    if not np.allclose(X_res[:len(X)], X):
        print("  [adasyn] output order changed unexpectedly; ADASYN skipped")
        return list(segments), []

    synthetic = [{
        "home": None, "home_id": -1, "batch_no": None, "indices": None,
        "label": int(label), "sid": f"synthetic|adasyn|{int(label)}|{i}",
        "synthetic": True, "vec": np.asarray(vec, dtype=np.float32),
    } for i, (vec, label) in enumerate(zip(X_res[len(X):], y_res[len(X):]))]

    final = collections.Counter([s["label"] for s in segments] + [s["label"] for s in synthetic])
    print(f"  [adasyn] +{len(synthetic)} synthetic; "
          + ", ".join(f"{name}={counts.get(c, 0)}->{final.get(c, 0)}"
                      for c, name in enumerate(cfg.classes)))
    return list(segments), synthetic
