"""Score a HER2 case from saved frames (until the live capture app exists).

    python app/her2_score.py frames/*.jpg --um-per-px 0.5 --case-id H-001
    python app/her2_score.py frames/*.jpg --um-per-px 0.5 --thresholds models/her2_thresholds_bci.json

Writes <out>/report_<case>.pdf, session.json and an overlay per counted field. Frames
overlapping an earlier frame by > 50% are skipped.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from core.her2.pipeline import HER2Analyzer, field_counts
from core.her2.scoring import HER2Thresholds
from core.her2.session import HER2Session, draw_her2_overlay, write_her2_report
from core.ihc.stain import StainMatrix
from core.nuclei import NucleiDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--um-per-px", type=float, required=True)
    ap.add_argument("--case-id", default="HER2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresholds", default=None, help="JSON from calibration (default: built-in placeholders)")
    ap.add_argument("--stain-matrix", default=None)
    ap.add_argument("--cells", choices=["watershed", "voronoi"], default="watershed")
    ap.add_argument("--min-cells", type=int, default=500)
    args = ap.parse_args()

    th = HER2Thresholds.load(args.thresholds) if args.thresholds else HER2Thresholds()
    an = HER2Analyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml"), None,
                      StainMatrix.from_json(Path(args.stain_matrix)) if args.stain_matrix else StainMatrix(), th,
                      cell_method=args.cells)
    out = Path(args.out or f"cases/{args.case_id}")
    s = HER2Session(out, args.case_id, min_cells=args.min_cells)
    for p in map(Path, args.images):
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"skip {p}: not an image")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        a = an.analyze(rgb, args.um_per_px)
        r = s.add_field(rgb, a)
        if not r.accepted:
            print(f"{p.name}: skipped, overlaps field {r.duplicate_of + 1} by {r.overlap:.0%}")
            continue
        c, _ = field_counts(a)
        n = max(c.sum(), 1)
        cv2.imwrite(str(out / f"fields/{r.field_index:03d}_overlay.jpg"), cv2.cvtColor(draw_her2_overlay(rgb, a), cv2.COLOR_RGB2BGR))
        print(f"{p.name}: field {r.field_index + 1}, {c.sum()} cells | none {100 * c[0] / n:.0f}% faint {100 * c[1] / n:.0f}% "
              f"weak-mod complete {100 * c[2] / n:.0f}% strong complete {100 * c[3] / n:.0f}%"
              + "".join(f"\n   ! {f}" for f in a.artefact["flags"]))
    s.save()
    pdf = write_her2_report(s, out / f"report_{args.case_id}.pdf")
    res = s.result()
    print(f"\nHER2 IHC {res.score}{' (' + res.subgroup + ')' if res.subgroup else ''} - {res.her2_low}"
          f"{'  -> reflex ISH' if res.reflex_ish else ''}")
    print(f"cells {res.n_cells}: any staining {res.pct_any_staining}%, complete {res.pct_complete}%, strong {res.pct[3]}%")
    for f in res.flags:
        print("note:", f)
    print("report:", pdf)


if __name__ == "__main__":
    main()
