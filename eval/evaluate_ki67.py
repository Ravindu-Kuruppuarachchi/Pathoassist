"""Feature 3 proxy validation: Ki-67 index per field vs the DeepLIIF reference.

Datasets (every nucleus labelled Ki-67 positive / negative from co-registered multiplex IF):
  - BC-DeepLIIF validation (breast, 6 independent fields - the other files are flipped /
    zoomed copies). The nuclei model was fine-tuned on BC-DeepLIIF *training* fields and the
    positivity cut was fitted there, so these 6 are held out.
  - DeepLIIF testing set (lung + bladder Ki-67, never used for training or calibration).

The reference counts all nuclei, so the main comparison counts all detected nuclei
(cell-type filter off); the tumour-only index (classifier on) is reported alongside.
Target (plan): within +/-5 percentage points of manual counts.

    python eval/evaluate_ki67.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np

from core.ihc.celltype import CellTypeClassifier
from core.ihc.pipeline import TUMOUR
from core.ihc.scoring import weighted_kappa
from core.ihc.stain import StainMatrix
from core.ihc.validation import icc_2_1, bootstrap_ci
from core.ki67.scoring import PositivityThreshold, band, global_index, hotspot_index
from core.ki67.session import ki67_analyzer
from core.nuclei import NucleiDetector
from evaluate_ihc_deepliif import centroids, load, match

BANDS = ["Low", "Intermediate", "High"]


def stats(ref, ass):
    r, a = np.asarray(ref, float), np.asarray(ass, float)
    d = a - r
    rb = np.array([BANDS.index(band(v).split()[0]) for v in r])
    ab = np.array([BANDS.index(band(v).split()[0]) for v in a])
    return dict(n=len(r), within_5pts=float((np.abs(d) <= 5).mean()), mae=float(np.abs(d).mean()),
                bias=float(d.mean()), loa=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
                icc=icc_2_1(r, a), icc_ci95=bootstrap_ci(icc_2_1, r, a),
                band_agreement=float((rb == ab).mean()), band_kappa=float(weighted_kappa(rb, ab, n=3)),
                ref_range=[float(r.min()), float(r.max())])


def run(zip_name, originals_only, an_all, clf):
    rows = []
    for name, ihc, gt, gt_pos in load(zip_name, originals_only):
        a = an_all.analyze(ihc, um_per_px=0.25)
        pos = a.intensity > 0
        m = match(centroids(gt), a.points, 12)
        tumour = clf.predict(ihc, a.points, 0.25).argmax(1) == TUMOUR if (clf is not None and len(a)) else np.ones(len(a), bool)
        rows.append(dict(name=name, ref_n=len(gt_pos), ref_pos=int(gt_pos.sum()), n=len(a), pos=int(pos.sum()),
                         n_tumour=int(tumour.sum()), pos_tumour=int((pos & tumour).sum()),
                         matched_ref_pos=gt_pos[m[:, 0]].tolist() if len(m) else [],
                         matched_pred_pos=pos[m[:, 1]].tolist() if len(m) else []))
    return rows


def report(title, rows):
    ok = [r for r in rows if r["ref_n"] >= 20]
    ref = [100 * r["ref_pos"] / r["ref_n"] for r in ok]
    ass = [100 * r["pos"] / max(r["n"], 1) for r in ok]
    tum = [100 * r["pos_tumour"] / r["n_tumour"] if r["n_tumour"] else np.nan for r in ok]
    y = np.array(sum((r["matched_ref_pos"] for r in ok), []), bool)
    p = np.array(sum((r["matched_pred_pos"] for r in ok), []), bool)
    s = stats(ref, ass)
    s["nucleus_positivity"] = dict(n=int(len(y)), sensitivity=float((p & y).sum() / max(y.sum(), 1)),
                                   specificity=float((~p & ~y).sum() / max((~y).sum(), 1)))
    s["detection_recall"] = float(len(y) / max(sum(r["ref_n"] for r in ok), 1))
    tv = ~np.isnan(tum)
    s["tumour_only"] = dict(n_fields_with_tumour=int(tv.sum()),
                            median_tumour_fraction=float(np.median([r["n_tumour"] / max(r["n"], 1) for r in ok])),
                            within_5pts=float((np.abs(np.array(tum)[tv] - np.array(ref)[tv]) <= 5).mean()) if tv.any() else None)
    print(f"\n== {title}: {s['n']} fields (reference Ki-67 {s['ref_range'][0]:.1f}-{s['ref_range'][1]:.1f}%)")
    print(f"   within +/-5 pts: {100 * s['within_5pts']:.0f}%   MAE {s['mae']:.1f} pts   bias {s['bias']:+.1f}   "
          f"95% limits {s['loa'][0]:+.1f} to {s['loa'][1]:+.1f}")
    print(f"   ICC {s['icc']:.3f} (95% CI {s['icc_ci95'][0]:.3f}-{s['icc_ci95'][1]:.3f})   "
          f"IKWG band agreement {100 * s['band_agreement']:.0f}% (kappa {s['band_kappa']:.2f})")
    print(f"   per nucleus: detection recall {s['detection_recall']:.3f}, positivity sens "
          f"{s['nucleus_positivity']['sensitivity']:.3f} spec {s['nucleus_positivity']['specificity']:.3f}")
    t = s["tumour_only"]
    print(f"   tumour-only (classifier on): median {100 * t['median_tumour_fraction']:.0f}% of nuclei called tumour; "
          f"within +/-5 pts {100 * t['within_5pts']:.0f}% of {t['n_fields_with_tumour']} fields" if t["within_5pts"] is not None
          else "   tumour-only: no nuclei called tumour")
    return s, ok


def case_level(title, groups):
    """Global and hotspot index per case (fields grouped), reference computed the same way."""
    out = []
    for cid, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        rc = [[r["ref_n"] - r["ref_pos"], r["ref_pos"]] for r in rs]
        ac = [[r["n"] - r["pos"], r["pos"]] for r in rs]
        rg, ag = global_index(rc), global_index(ac)
        rh, ah = hotspot_index(rc), hotspot_index(ac)
        out.append(dict(case=cid, fields=len(rs), ref_global=rg.index_pct, global_=ag.index_pct,
                        ref_hotspot=rh.index_pct, hotspot=ah.index_pct,
                        same_hotspot_fields=sorted(rh.fields_used) == sorted(ah.fields_used)))
    if out:
        print(f"\n   {title}: {len(out)} cases (fields grouped by slide)")
        print(f"   {'case':>10} fields | global ref/assisted | hotspot ref/assisted | same hotspot fields")
        for o in out[:12]:
            print(f"   {o['case']:>10} {o['fields']:6d} | {o['ref_global']:6.1f} / {o['global_']:5.1f}     | "
                  f"{o['ref_hotspot']:6.1f} / {o['hotspot']:5.1f}      | {o['same_hotspot_fields']}")
        g = stats([o["ref_global"] for o in out], [o["global_"] for o in out])
        h = stats([o["ref_hotspot"] for o in out], [o["hotspot"] for o in out])
        print(f"   global : within +/-5 pts {100 * g['within_5pts']:.0f}%, MAE {g['mae']:.1f}")
        print(f"   hotspot: within +/-5 pts {100 * h['within_5pts']:.0f}%, MAE {h['mae']:.1f}, "
              f"same fields chosen {100 * np.mean([o['same_hotspot_fields'] for o in out]):.0f}%")
    return out


def main():
    an = ki67_analyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml"), None, StainMatrix(),
                       PositivityThreshold())
    clf = CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx")
    res = {}
    datasets = [("BC-DeepLIIF validation (breast)", "BC-DeepLIIF_Validation_Set.zip", True)]
    test_zip = ROOT / "training/data/deepliif/DeepLIIF_Testing_Set.zip"
    if test_zip.exists() and __import__("zipfile").is_zipfile(test_zip):
        datasets.append(("DeepLIIF test set (lung + bladder)", test_zip.name, False))
    for title, z, orig in datasets:
        rows = run(z, orig, an, clf)
        s, ok = report(title, rows)
        groups = defaultdict(list)
        for r in ok:
            groups[r["name"].rsplit("_", 2)[0] if not orig else "BC-all"].append(r)   # Tissue_x_y = one region
        s["cases"] = case_level("case-level global vs hotspot", groups)
        s["fields"] = [{k: v for k, v in r.items() if not k.startswith("matched")} for r in ok]
        res[title] = s
    (ROOT / "eval/results/ki67_validation.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
