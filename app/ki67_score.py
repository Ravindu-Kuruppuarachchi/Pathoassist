"""Score a Ki-67 case from saved frames (until the live capture app exists).

    python app/ki67_score.py frames/*.jpg --um-per-px 0.25 --case-id K-001 --out cases/K-001
    python app/ki67_score.py frames/*.jpg --um-per-px 0.25 --celltype   # count tumour nuclei only

Writes <out>/report_<case>.pdf, session.json (global + hotspot index, per-field counts) and
an overlay per counted field. Frames overlapping an earlier frame by > 50% are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from core.ihc.stain import StainMatrix
from core.ki67.report import draw_ki67_overlay, write_ki67_report
from core.ki67.scoring import PositivityThreshold
from core.ki67.session import Ki67Session, ki67_analyzer
from core.nuclei import NucleiDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--um-per-px", type=float, required=True)
    ap.add_argument("--case-id", default="KI67")
    ap.add_argument("--out", default=None)
    ap.add_argument("--celltype", action="store_true",
                    help="count only nuclei the cell-type classifier calls tumour (classifier not yet validated on IHC)")
    ap.add_argument("--positivity", default=None, help="JSON from calibrate_positivity (default: cut 0.52)")
    ap.add_argument("--stain-matrix", default=None, help="JSON from estimate_hdab_matrix (default: Ruifrok)")
    ap.add_argument("--primary", choices=["global", "hotspot"], default="global")
    ap.add_argument("--min-cells", type=int, default=500)
    args = ap.parse_args()

    clf = None
    if args.celltype:
        from core.ihc.celltype import CellTypeClassifier
        clf = CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx")
    an = ki67_analyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml"), clf,
                       StainMatrix.from_json(Path(args.stain_matrix)) if args.stain_matrix else StainMatrix(),
                       PositivityThreshold.load(args.positivity) if args.positivity else PositivityThreshold())
    out = Path(args.out or f"cases/{args.case_id}")
    s = Ki67Session(out, args.case_id, min_cells=args.min_cells, celltype_filter=args.celltype)
    for p in map(Path, args.images):
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"skip {p}: not an image")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = an.analyze(rgb, um_per_px=args.um_per_px)
        r = s.add_field(rgb, a)
        if not r.accepted:
            print(f"{p.name}: skipped, overlaps field {r.duplicate_of + 1} by {r.overlap:.0%}")
            continue
        c = s.field_counts()[-1]
        cv2.imwrite(str(out / f"fields/{r.field_index:03d}_overlay.jpg"),
                    cv2.cvtColor(draw_ki67_overlay(rgb, a), cv2.COLOR_RGB2BGR))
        print(f"{p.name}: field {r.field_index + 1}, {c[1]}/{c.sum()} positive ({100 * c[1] / max(c.sum(), 1):.1f}%), "
              f"{a.timings['total']:.2f}s")
    s.save()
    pdf = write_ki67_report(s, out / f"report_{args.case_id}.pdf", primary=args.primary)
    g, h = s.global_result(), s.hotspot_result()
    print(f"\nGlobal  Ki-67 {g.index_pct:.1f}% ({g.n_positive}/{g.n_counted}) {g.band}")
    print(f"Hotspot Ki-67 {h.index_pct:.1f}% ({h.n_positive}/{h.n_counted}, fields {[i + 1 for i in h.fields_used]}) {h.band}")
    for w in dict.fromkeys(g.warnings + h.warnings):
        print("warning:", w)
    print("report:", pdf)


if __name__ == "__main__":
    main()
