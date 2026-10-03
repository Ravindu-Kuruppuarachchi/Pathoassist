"""Calibrate and validate HER2 scoring on BCI (proxy for the local 100-case study).

1. extract : run the HER2 pipeline once per image (nuclei -> cells -> membrane sectors) and
             cache per-cell sector DAB ODs (training/data/bci/feats/*.npz)
2. fit     : grid-search the per-cell thresholds on BCI-train to maximise the quadratic-weighted
             kappa of the field score vs the label (the 10% rule cut-offs stay fixed)
3. evaluate: BCI-test - weighted kappa, accuracy, confusion; the 0 vs 1+ boundary separately.

BCI caveats: labels are the slide's HER2 score copied to every patch (many patches are fat
or stroma); images are ~1 µm/px (10x), so membranes are ~1 px wide. Treat as a proxy only.

    python training/her2/calibrate_bci.py --workers 8
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

D = ROOT / "training" / "data" / "bci"
UM_PER_PX = 1.0
LABELS = {"0": 0, "1+": 1, "2+": 2, "3+": 3}
_an = None


def _init():
    global _an
    import cv2
    cv2.setNumThreads(1)
    from core.her2.pipeline import HER2Analyzer
    from core.nuclei import NucleiDetector
    _an = HER2Analyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml", num_threads=4))


def _extract(rec):
    import cv2
    out = D / "feats" / (Path(rec["file"]).stem + ".npz")
    if out.exists():
        return rec
    im = cv2.cvtColor(cv2.imread(str(D / "ihc" / rec["file"])), cv2.COLOR_BGR2RGB)
    a = _an.analyze(im, UM_PER_PX)
    np.savez_compressed(out, sector_od=a.sector_od.astype(np.float16), measurable=a.measurable,
                        points=a.points.astype(np.float32), cyto=a.artefact["cytoplasm_dab"],
                        extra=a.artefact["extracellular_dab"])
    return rec


def cluster_mask(points, um_per_px=UM_PER_PX, radius_um=20.0, min_neighbours=4):
    """Stand-in tumour filter while the cell-type classifier is not usable on IHC: keep nuclei
    with >= min_neighbours others within radius_um (cells in epithelial sheets / nests)."""
    if len(points) < 2:
        return np.zeros(len(points), bool)
    from scipy.spatial import cKDTree
    t = cKDTree(points * um_per_px)
    return np.array([len(n) - 1 for n in t.query_ball_point(points * um_per_px, radius_um)]) >= min_neighbours


def refined_labels():
    """Expert-reviewed patch-level HER2 labels (Zenodo 17200604) - better than BCI's slide-level labels."""
    import csv
    f = D / "refined_labels.csv"
    if not f.exists():
        return {}
    norm = lambda v: LABELS[v.strip().rstrip("+") + ("+" if v.strip().rstrip("+") != "0" else "")]   # "1" / "0+" typos
    return {(r["pair_id"], r["partition"]): norm(r["her2_level_refined"]) for r in csv.DictReader(open(f))}


def load_feats(recs, filt, labels="original"):
    ref = refined_labels() if labels == "refined" else None
    out = []
    for r in recs:
        if ref is not None and (r["id"], r["split"]) not in ref:
            continue
        z = np.load(D / "feats" / (Path(r["file"]).stem + ".npz"))
        keep = z["measurable"].copy()
        if filt == "cluster":
            keep &= cluster_mask(z["points"])
        y = ref[(r["id"], r["split"])] if ref is not None else LABELS[r["label"]]
        out.append(dict(S=z["sector_od"][keep].astype(np.float32), y=y, file=r["file"],
                        cyto=float(z["cyto"]), extra=float(z["extra"])))
    return out


def score_fields(feats, th):
    from core.her2.scoring import classify_cells, her2_score
    preds, details = [], []
    for f in feats:
        cat, ii, *_ = classify_cells(f["S"], th, min_valid_sectors=0)
        r = her2_score(np.bincount(cat, minlength=4)[:4], int(ii.sum()), min_cells=0)
        preds.append(int(r.score[0]))
        details.append(r)
    return np.array(preds), details


def metrics(y, p):
    from core.ihc.scoring import weighted_kappa
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, p), 1)
    zo = (y <= 1)
    return dict(n=int(len(y)), weighted_kappa=float(weighted_kappa(y, p)), accuracy=float((y == p).mean()),
                within_one=float((np.abs(y - p) <= 1).mean()), confusion=cm.tolist(),
                zero_vs_1plus=dict(n=int(zo.sum()), accuracy=float(((p[zo] >= 1) == (y[zo] == 1)).mean()) if zo.any() else None,
                                   kappa=float(weighted_kappa((y[zo] == 1).astype(int), (p[zo] >= 1).astype(int), n=2)) if zo.any() else None),
                positive_vs_rest=dict(sens_3plus=float((p[y == 3] == 3).mean()) if (y == 3).any() else None,
                                      ish_triage=float(((p >= 2) == (y >= 2)).mean())))


def fit(feats):
    from core.her2.scoring import HER2Thresholds
    y = np.array([f["y"] for f in feats])
    best = (-np.inf, None)
    for so in [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.13]:
        for sm in [0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4]:
            for cm in [0.3, 0.4, 0.5, 0.6, 0.75]:
                for fm in [so, so * 1.5, so * 2]:
                    if fm >= sm:
                        continue
                    th = HER2Thresholds(so, sm, cm, round(fm, 3), "fitted on BCI-train field labels")
                    k = metrics(y, score_fields(feats, th)[0])["weighted_kappa"]
                    if k > best[0]:
                        best = (k, th)
    return best[1], best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=600, help="training images used to fit the thresholds")
    args = ap.parse_args()
    recs = json.loads((D / "index.json").read_text())
    recs = [r for r in recs if r["split"] == "test"] + [r for r in recs if r["split"] == "train"][:args.n_train]
    (D / "feats").mkdir(exist_ok=True)
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex:
        list(ex.map(_extract, recs, chunksize=4))
    train = [r for r in recs if r["split"] == "train"]
    test = [r for r in recs if r["split"] == "test"]
    from core.her2.scoring import HER2Thresholds
    results = {}
    lab = "refined" if refined_labels() else "original"
    for filt in ("all_cells",):          # the neighbour-count 'cluster' filter kept too few cells (kappa 0.09)
        ftr = load_feats(train, filt, lab)
        fte = load_feats(test, filt, lab)
        default = metrics(np.array([f["y"] for f in fte]), score_fields(fte, HER2Thresholds())[0])
        th, k_train = fit(ftr)
        p, det = score_fields(fte, th)
        y = np.array([f["y"] for f in fte])
        m = metrics(y, p)
        fo = load_feats(test, filt, "original")
        m_orig = metrics(np.array([f["y"] for f in fo]), score_fields(fo, th)[0])
        results[filt] = dict(labels=lab, n_train=len(ftr), thresholds=th.__dict__, train_kappa=k_train, test=m,
                             test_original_labels=m_orig, test_default_thresholds=default,
                             median_cells=float(np.median([len(f["S"]) for f in fte])))
        print(f"\n== {filt} (fitted on {len(ftr)} train images with {lab} labels): original-label test kappa "
              f"{m_orig['weighted_kappa']:.3f}, accuracy {m_orig['accuracy']:.3f} (n={m_orig['n']})")
        print(f"\n== {filt}: median {results[filt]['median_cells']:.0f} counted cells per image")
        print(f"   default thresholds : test kappa {default['weighted_kappa']:.3f}, accuracy {default['accuracy']:.3f}")
        print(f"   fitted on train    : {th.__dict__}")
        print(f"   test (n={m['n']})  : weighted kappa {m['weighted_kappa']:.3f}, accuracy {m['accuracy']:.3f}, "
              f"within one category {m['within_one']:.3f}")
        print(f"   0 vs 1+            : accuracy {m['zero_vs_1plus']['accuracy']:.3f}, kappa {m['zero_vs_1plus']['kappa']:.3f} "
              f"(n={m['zero_vs_1plus']['n']})")
        print(f"   ISH triage (2+/3+ vs 0/1+) agreement {m['positive_vs_rest']['ish_triage']:.3f}; "
              f"3+ recognised {m['positive_vs_rest']['sens_3plus']:.3f}")
        print("   confusion (rows = label 0,1+,2+,3+; cols = predicted):")
        for row in m["confusion"]:
            print("     ", row)
    best = max(results, key=lambda k: results[k]["test"]["weighted_kappa"])
    th = HER2Thresholds(**results[best]["thresholds"])
    th.save(ROOT / "models" / "her2_thresholds_bci.json")
    results["chosen"] = best
    (ROOT / "eval" / "results" / "her2_bci.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
