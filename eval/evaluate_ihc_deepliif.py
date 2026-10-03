"""Feature 2 proxy validation on real IHC: BC-DeepLIIF (breast Ki-67, every nucleus labelled
positive / negative).

Stand-in until archived ER/PR cases are available: it checks the parts of the pipeline
that do not depend on the marker - nuclei detection on IHC, the DAB positivity call per
nucleus, and the per-field % positive against the reference (ICC, category agreement).

  - DAB positivity threshold is calibrated on the BC-DeepLIIF *training* set, evaluated
    on the *validation* set (held out).
  - BC-DeepLIIF has only ~35 training and 6 validation fields; its other files are
    augmented copies. Headline numbers use the 6 original validation fields.
  - All detected nuclei are counted (the cell-type classifier is reported separately),
    because the DeepLIIF reference counts all nuclei.

    python eval/evaluate_ihc_deepliif.py
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from core.ihc.stain import StainMatrix, per_nucleus_stain
from core.ihc.validation import agreement
from core.nuclei import NucleiDetector

D = ROOT / "training" / "data" / "deepliif"


def load(zip_name, originals_only=False):
    """Yields (name, ihc_rgb, gt_labels, gt_positive[bool per label]).

    Files N_k are flipped / rotated / zoomed copies of field N (checked visually), not new
    tissue, so only N.png files are independent. originals_only=True yields just those;
    otherwise copies are included (exact duplicates skipped)."""
    import hashlib
    seen = set()
    with zipfile.ZipFile(D / zip_name) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".png"))
        if originals_only:
            names = [n for n in names if "_" not in Path(n).stem]
        for n in names:
            im = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            w = im.shape[0]
            h = hashlib.md5(im[:, :w].tobytes()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            ihc = cv2.cvtColor(im[:, :w], cv2.COLOR_BGR2RGB)
            seg = cv2.cvtColor(im[:, 5 * w:6 * w], cv2.COLOR_BGR2RGB).astype(int)
            R, G, B = seg[..., 0], seg[..., 1], seg[..., 2]
            pos = (R > 128) & (G < 100) & (B < 100)
            neg = (B > 128) & (R < 100) & (G < 100)
            lp, n_p = ndi.label(pos)
            ln, n_n = ndi.label(neg)
            lab = np.where(lp > 0, lp, np.where(ln > 0, ln + n_p, 0)).astype(np.int32)
            # drop specks (anti-aliasing at mask borders)
            area = np.bincount(lab.ravel(), minlength=n_p + n_n + 1)
            keep = area >= 20
            keep[0] = False
            remap = np.zeros(len(area), np.int32)
            remap[keep] = np.arange(1, keep.sum() + 1)
            positive = (np.arange(len(area)) <= n_p)[keep]
            yield Path(n).stem, ihc, remap[lab], positive


def centroids(lab):
    ids = np.arange(1, lab.max() + 1)
    return np.array(ndi.center_of_mass(lab > 0, lab, ids)).reshape(-1, 2) if len(ids) else np.zeros((0, 2))


def match(a, b, radius):
    if not len(a) or not len(b):
        return np.zeros((0, 2), int)
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    d[d > radius] = 1e6
    r, c = linear_sum_assignment(d)
    ok = d[r, c] <= radius
    return np.stack([r[ok], c[ok]], 1)


def run(det, stain, zip_name, radius, originals_only=False):
    rows = []
    for name, ihc, gt, gt_pos in load(zip_name, originals_only):
        d = det.detect(ihc)
        conc = stain.concentrations(ihc)
        _, dab = per_nucleus_stain(d.labels, conc, len(d))
        dab = np.nan_to_num(dab)
        m = match(centroids(gt), d.points, radius)
        rows.append(dict(name=name, n_gt=len(gt_pos), n_pred=len(d), tp=len(m), gt_pos=gt_pos,
                         pred_dab=dab, matched=m))
    return rows


def best_threshold(rows):
    """DAB OD cut maximising balanced accuracy of positive/negative on matched nuclei."""
    y = np.concatenate([r["gt_pos"][r["matched"][:, 0]] for r in rows])
    x = np.concatenate([r["pred_dab"][r["matched"][:, 1]] for r in rows])
    best = max(((((x >= t) & y).sum() / max(y.sum(), 1) + ((x < t) & ~y).sum() / max((~y).sum(), 1)) / 2, t)
               for t in np.linspace(0.02, 0.6, 117))
    return float(best[1]), float(best[0])


def summarize(rows, t):
    tp = sum(r["tp"] for r in rows)
    n_gt = sum(r["n_gt"] for r in rows)
    n_pr = sum(r["n_pred"] for r in rows)
    y = np.concatenate([r["gt_pos"][r["matched"][:, 0]] for r in rows])
    x = np.concatenate([r["pred_dab"][r["matched"][:, 1]] for r in rows]) >= t
    ref = [100 * r["gt_pos"].mean() if r["n_gt"] else 0 for r in rows]
    ass = [100 * (r["pred_dab"] >= t).mean() if r["n_pred"] else 0 for r in rows]
    det = dict(precision=tp / max(n_pr, 1), recall=tp / max(n_gt, 1), f1=2 * tp / max(n_pr + n_gt, 1))
    pos = dict(accuracy=float((x == y).mean()), sensitivity=float((x & y).sum() / max(y.sum(), 1)),
               specificity=float((~x & ~y).sum() / max((~y).sum(), 1)), n_matched=int(len(y)))
    return dict(n_images=len(rows), detection=det, positivity=pos, field_pct=agreement(ref, ass),
                reference_pct_range=[float(min(ref)), float(max(ref))])


def main():
    stain = StainMatrix()
    path = ROOT / "eval" / "results" / "ihc_deepliif.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    first = next(load("BC-DeepLIIF_Validation_Set.zip"))
    gt_area = np.median(np.bincount(first[2].ravel())[1:])
    print(f"median GT nucleus area {gt_area:.0f} px (PanNuke at 0.25 µm/px: ~300-450 px)")
    for model in sys.argv[1:] or ["models/stardist_he_pretrained_int8.xml", "models/stardist_he_ft_int8.xml",
                                  "models/stardist_he_ihc_ft_int8.xml"]:
        det = NucleiDetector(ROOT / model)
        radius = 12
        cal = run(det, stain, "BC-DeepLIIF_Training_Set.zip", radius)
        t, bacc = best_threshold(cal)
        test = run(det, stain, "BC-DeepLIIF_Validation_Set.zip", radius, originals_only=True)
        res = summarize(test, t)
        res["with_augmented_copies"] = summarize(run(det, stain, "BC-DeepLIIF_Validation_Set.zip", radius), t)
        res["dab_threshold_calibrated_on_train"] = t
        res["calibration_balanced_accuracy"] = bacc
        out[Path(model).stem] = res
        s = res
        print(f"\n{model}: {s['n_images']} independent held-out IHC fields "
              f"(+ copies: F1 {s['with_augmented_copies']['detection']['f1']:.3f}, "
              f"ICC {s['with_augmented_copies']['field_pct']['icc']:.3f})")
        print(f"  nuclei detection  P {s['detection']['precision']:.3f} R {s['detection']['recall']:.3f} "
              f"F1 {s['detection']['f1']:.3f}")
        print(f"  positivity (DAB OD >= {t:.3f}) acc {s['positivity']['accuracy']:.3f} "
              f"sens {s['positivity']['sensitivity']:.3f} spec {s['positivity']['specificity']:.3f}")
        f = s["field_pct"]
        print(f"  % positive per field: ICC {f['icc']:.3f} {f['icc_ci95']}  MAE {f['mae_pct']:.1f} pts  "
              f"category agreement {f['category_agreement']:.2f}")
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
