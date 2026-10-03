"""Feature 8 check on Databiox: tubule % per patient from the 10x camera images vs overall grade.

Databiox labels are the overall Nottingham grade (tubule score itself is not given), so this checks
direction and separation (grade 1 tumours should form more tubules than grade 3), not exact
agreement. Large-camera 10x images only (0.964 µm/px; the small camera's 3 µm/px is too coarse for
lumens). Invasive tumour area comes from the Feature 5 region model.

    python eval/evaluate_tubules_databiox.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.stats import spearmanr

from core.ihc.scoring import weighted_kappa
from core.region.tissue_map import TissueRegionModel
from core.tubules.lumen import LumenClassifier
from core.tubules.score import TubuleSession, draw_tubule_overlay

D = ROOT / "training/data/databiox"
UM10 = 0.964


def main():
    recs = [r for r in json.loads((D / "index.json").read_text()) if r["mag"] == "10x" and r["width"] > 2000]
    clf = LumenClassifier(ROOT / "models/tubule_lumen_lgbm.txt")
    rm = None   # Databiox photos are tumour fields; the region model under-outlines tumour on these phone images
    by = {}
    for r in recs:
        by.setdefault((r["patient"], r["grade"]), []).append(r)
    rows, shown = [], 0
    for (pid, grade), rs in sorted(by.items()):
        s = TubuleSession(clf, rm, min_tumour_mm2=0.5)
        for r in rs:
            im = cv2.cvtColor(cv2.imread(str(D / r["file"])), cv2.COLOR_BGR2RGB)
            s.add_field(im, UM10)
            if shown < 3 and grade in (1, 3) and s.fields:
                f = s.fields[-1]
                cv2.imwrite(str(ROOT / f"demo_outputs/8_tubules_grade{grade}_{pid}.jpg"),
                            cv2.cvtColor(cv2.resize(draw_tubule_overlay(im, f), None, fx=0.35, fy=0.35), cv2.COLOR_RGB2BGR))
                shown += 1
        res = s.result()
        rows.append(dict(patient=pid, grade=grade, tubule_pct=res.tubule_pct, tubule_score=res.score, tumour_mm2=res.tumour_area_mm2,
                         lumens=res.n_lumens, fields=res.n_fields))
        print(f"patient {pid} grade {grade}: tubule {res.tubule_pct:5.1f}% -> score {res.score} ({res.tumour_area_mm2:.1f} mm² tumour, "
              f"{res.n_lumens} lumens, {res.n_fields} fields)", flush=True)
    ok = [r for r in rows if r["tumour_mm2"] > 0]
    g = np.array([r["grade"] for r in ok])
    pct = np.array([r["tubule_pct"] for r in ok])
    sc = np.array([r["tubule_score"] for r in ok])
    out = dict(n_patients=len(ok), spearman_tubule_pct_vs_grade=float(spearmanr(pct, g)[0]),
               spearman_tubule_score_vs_grade=float(spearmanr(sc, g)[0]),
               median_pct_by_grade={int(k): float(np.median(pct[g == k])) for k in (1, 2, 3)},
               score_distribution_by_grade={int(k): np.bincount(sc[g == k], minlength=4)[1:].tolist() for k in (1, 2, 3)},
               kappa_tubule_score_vs_grade=float(weighted_kappa(g - 1, sc - 1, n=3)), patients=rows)
    print(json.dumps({k: v for k, v in out.items() if k != "patients"}, indent=1))
    (ROOT / "eval/results/tubules_databiox.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
