"""Accuracy of an exported nuclei model on the held-out PanNuke test fold (fold 3).

Runs the real app engine (core.nuclei.NucleiDetector), so the numbers include export,
quantisation, tiling and post-processing effects.

Metrics: (1) segmentation-aware object matching at IoU >= 0.5 (stardist.matching) ->
precision, recall, F1, panoptic quality; (2) detection F1 with centroid matching within
3 µm (12 px), the PanNuke / HoVer-Net detection protocol. Reported overall, for breast only, per tissue,
and for a sweep of IoU thresholds.

    python eval/evaluate.py models/stardist_he_ft_int8.xml
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from stardist.matching import matching_dataset
from tqdm import tqdm

from core.nuclei import NucleiDetector
from training.data import load_split


def centroids(lab):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    if not len(ids):
        return np.zeros((0, 2))
    return np.array(ndi.center_of_mass(lab > 0, lab, ids))


def centroid_detection(Y, P, radius):
    """PanNuke/HoVer-Net style detection F1: 1:1 Hungarian matching of centres within `radius` px."""
    tp = n_true = n_pred = 0
    for y, p in zip(Y, P):
        a, b = centroids(y), centroids(p)
        n_true += len(a)
        n_pred += len(b)
        if len(a) and len(b):
            d = np.linalg.norm(a[:, None] - b[None], axis=-1)
            d[d > radius] = 1e6  # out-of-range pairs must not steal assignments
            r, c = linear_sum_assignment(d)
            tp += int((d[r, c] <= radius).sum())
    pr, rc = tp / max(n_pred, 1), tp / max(n_true, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9))


def scores(Y, P, thresh=0.5):
    m = matching_dataset(Y, P, thresh=thresh, show_progress=False)
    det = centroid_detection(Y, P, radius=12)  # 12 px = 3 µm at 0.25 µm/px (PanNuke protocol)
    return dict(n_true=m.n_true, n_pred=m.n_pred, tp=m.tp, fp=m.fp, fn=m.fn,
                precision=m.precision, recall=m.recall, f1=m.f1, pq=m.panoptic_quality,
                det_precision=det["precision"], det_recall=det["recall"], det_f1=det["f1"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prob-thresh", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()

    X, Y, T = load_split(args.split)
    if args.limit:
        X, Y, T = X[:args.limit], Y[:args.limit], T[:args.limit]
    det = NucleiDetector(args.model, tile=384, margin=64, prob_thresh=args.prob_thresh)

    P, t_net = [], []
    t0 = time.time()
    for x in tqdm(X, desc=Path(args.model).name):
        d = det.detect(x, um_per_px=None)
        P.append(d.labels)
        t_net.append(d.timings["total"])
    Y = [y.astype(np.int32) for y in Y]

    res = dict(model=args.model, split=args.split, n_images=len(X),
               prob_thresh=det.prob_thresh, nms_thresh=det.nms_thresh,
               sec_per_256px_patch=float(np.mean(t_net)), wall_sec=time.time() - t0)
    res["all"] = scores(Y, P)
    breast = np.where(T == "Breast")[0]
    res["breast"] = scores([Y[i] for i in breast], [P[i] for i in breast])
    res["iou_sweep_all"] = {f"{t:.1f}": scores(Y, P, t)["f1"] for t in (0.3, 0.5, 0.7, 0.9)}
    rows = []
    for tissue in sorted(set(T)):
        ii = np.where(T == tissue)[0]
        rows.append(dict(tissue=tissue, n=len(ii), **scores([Y[i] for i in ii], [P[i] for i in ii])))
    res["per_tissue"] = rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args.model).stem}_{args.split}" + (f"_p{args.prob_thresh}" if args.prob_thresh else "")
    (out / f"{stem}.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame(rows).to_csv(out / f"{stem}_per_tissue.csv", index=False)

    fmt = lambda s: (f"IoU0.5: P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} PQ={s['pq']:.3f} | "
                     f"centroid(3um): P={s['det_precision']:.3f} R={s['det_recall']:.3f} F1={s['det_f1']:.3f}")
    print(f"\n{args.model} on {args.split} ({len(X)} patches)")
    print("  all    :", fmt(res["all"]))
    print("  breast :", fmt(res["breast"]), f"(n={len(breast)})")
    print("  F1 vs IoU:", res["iou_sweep_all"])


if __name__ == "__main__":
    main()
