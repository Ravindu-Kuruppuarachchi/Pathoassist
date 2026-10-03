"""Train the lumen vs non-lumen classifier on BreCaHAD (Aksac et al. 2019; 162 images, 17 cases,
~0.318 µm/px calibrated from lymphocyte size; point labels for 493 lumens and 610 non-lumens).

Each annotated point is matched to the candidate space containing it (or within 3 µm). Case-grouped
5-fold CV reports AUC, accuracy and point-level lumen recall / non-lumen specificity; the final model
is fitted on all cases. Unmatched points (no candidate found) count as misses in the recall.

    python training/tubules/train_lumen.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.tubules.lumen import UM, candidates

D = ROOT / "training/data/brecahad"
SCALE = 0.318


def build():
    X, y, groups, n_pts = [], [], [], {"lumen": 0, "non_lumen": 0}
    missed = {"lumen": 0, "non_lumen": 0}
    for jf in sorted((D / "groundTruth").glob("*.json")):
        ann = json.loads(jf.read_text())
        im = cv2.cvtColor(cv2.imread(str(D / "jpg" / (jf.stem + ".jpg"))), cv2.COLOR_BGR2RGB)
        c = candidates(im, SCALE)
        h, w = c.labels.shape
        dist, (iy, ix) = __import__("scipy.ndimage", fromlist=["x"]).distance_transform_edt(c.labels == 0, return_indices=True)
        for cls, lab in (("lumen", 1), ("non_lumen", 0)):
            for p in ann.get(cls, []):
                n_pts[cls] += 1
                x, yy = min(int(p["x"] * w), w - 1), min(int(p["y"] * h), h - 1)
                k = c.labels[yy, x] if c.labels[yy, x] else (c.labels[iy[yy, x], ix[yy, x]] if dist[yy, x] <= 3 / UM else 0)
                if k == 0:
                    missed[cls] += 1
                    continue
                X.append(c.feats[k - 1])
                y.append(lab)
                groups.append(jf.stem.split("-")[0])
        names = c.names
    return np.array(X), np.array(y), np.array(groups), names, n_pts, missed


def main():
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g, names, n_pts, missed = build()
    print(f"points: {n_pts}; no candidate found for {missed}; matched {len(y)} ({y.sum()} lumen) from {len(set(g))} cases", flush=True)
    params = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, is_unbalance=True)
    aucs, P_all = [], np.zeros(len(y))
    for r in range(5):
        P = np.zeros(len(y))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, g):
            P[te] = lgb.train(params, lgb.Dataset(X[tr], y[tr]), 200).predict(X[te])
        aucs.append(roc_auc_score(y, P))
        P_all += P / 5
    thr = 0.5
    pred = P_all >= thr
    rec = (pred & (y == 1)).sum() / (n_pts["lumen"])                  # unmatched lumen points count as misses
    spec = (~pred & (y == 0)).sum() / (y == 0).sum()
    res = dict(n_points=n_pts, unmatched=missed, auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
               accuracy_matched=float((pred == y).mean()), lumen_recall_all_points=float(rec), non_lumen_specificity=float(spec),
               lumen_precision=float((pred & (y == 1)).sum() / max(pred.sum(), 1)))
    print(json.dumps(res, indent=1), flush=True)
    m = lgb.train(params, lgb.Dataset(X, y), 200)
    m.save_model(str(ROOT / "models/tubule_lumen_lgbm.txt"))
    imp = sorted(zip(names, m.feature_importance("gain")), key=lambda t: -t[1])[:10]
    (ROOT / "models/tubule_lumen_lgbm.json").write_text(json.dumps(dict(
        features=names, threshold=thr, working_um_per_px=UM, source="BreCaHAD (17 cases), case-grouped CV",
        cv=res, top_features=[k for k, _ in imp]), indent=1))
    print("top features:", [k for k, _ in imp])
    (ROOT / "eval/results/tubule_lumen.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
