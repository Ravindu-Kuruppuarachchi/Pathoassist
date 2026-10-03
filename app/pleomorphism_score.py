"""Nuclear pleomorphism score (1-3) from saved 40x tumour fields (until the live app exists).

    python app/pleomorphism_score.py fields/*.jpg --um-per-px 0.25

Fields overlapping an earlier field by > 50% are skipped. Prints the per-field and the overall
score; writes <out>/pleomorphism.json and an overlay per field (red = tumour nuclei measured).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.ihc.celltype import CellTypeClassifier
from core.nuclei import NucleiDetector
from core.pleomorphism.scorer import TUMOUR, PleomorphismScorer, PleomorphismSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--um-per-px", type=float, required=True)
    ap.add_argument("--out", default="cases/pleomorphism")
    args = ap.parse_args()
    from core.pleomorphism.embedding import TileEmbedder
    sc = PleomorphismScorer(ROOT / "models/pleomorphism_phikon.npz", NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml"),
                            CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx"),
                            embedder=TileEmbedder(ROOT / "models/phikon_v2_int8w.xml"))
    s = PleomorphismSession(sc)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in map(Path, args.images):
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"skip {p}: not an image")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = s.add_field(rgb, args.um_per_px)
        if not r["accepted"]:
            print(f"{p.name}: skipped - {r['reason']}")
            continue
        f = s.fields[-1]
        ov = rgb.copy()
        for (y, x), c in zip(f.points.astype(int), f.cell_type):
            cv2.circle(ov, (x, y), 6, (220, 0, 0) if c == TUMOUR else (150, 150, 150), 2)
        cv2.imwrite(str(out / f"field_{f.index + 1:02d}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        med = np.median(f.table["area"][f.cell_type == TUMOUR]) if f.n_tumour else float("nan")
        print(f"{p.name}: field {f.index + 1}, {f.n_tumour} tumour nuclei (median {med:.0f} µm²), "
              f"P(1/2/3) = {np.round(r['probs'], 2).tolist()}")
    res = s.result()
    print(f"\nNuclear pleomorphism score {res.score} (P = {res.probs}, expected {res.expected}) from "
          f"{res.n_fields} fields / {res.n_tumour} tumour nuclei")
    from core.pleomorphism.scorer import MEASUREMENTS
    for k, v in res.measurements.items():
        print(f"  {MEASUREMENTS[k]}: {v}")
    for w in res.warnings:
        print("note:", w)
    (out / "pleomorphism.json").write_text(json.dumps(res.__dict__, indent=2))


if __name__ == "__main__":
    main()
