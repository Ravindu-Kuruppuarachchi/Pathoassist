"""End-to-end Feature 2 demo on held-out real IHC (BC-DeepLIIF validation fields).

Each BC-DeepLIIF tile plus its flipped / rotated / zoomed copies is treated as one "case"
(several dwell frames, most of which the de-duplication should reject). This demonstrates
the workflow and the report; it is NOT a validation (6 tiles, copies of the same tissue).
Each case goes through the full pipeline - nuclei -> cell type -> DAB -> CaseSession (dedup,
running totals) - and one case gets a PDF report. Also times each stage on CPU.

Note: BC-DeepLIIF is Ki-67, and its reference counts all nuclei, so the case-level
comparison uses the all-nuclei % positive; the tumour-only score (what an ER/PR report
uses) is shown alongside.

    python eval/demo_erpr_case.py --positivity 0.405
"""
import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np

from core.ihc.case import CaseSession
from core.ihc.celltype import CellTypeClassifier
from core.ihc.pipeline import ERPRAnalyzer
from core.ihc.report import write_report
from core.ihc.scoring import IntensityThresholds
from core.ihc.stain import StainMatrix
from core.nuclei import NucleiDetector
from evaluate_ihc_deepliif import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuclei", default=str(ROOT / "models/stardist_he_ihc_ft_int8.xml"))
    ap.add_argument("--celltype", default=str(ROOT / "models/celltype_mnv3s_ctx32.onnx"))
    ap.add_argument("--positivity", type=float, required=True,
                    help="DAB OD cut for positive (from evaluate_ihc_deepliif.py calibration on the training set)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "demo_case"))
    args = ap.parse_args()

    # Only the positive/negative cut is calibrated here; weak/moderate/strong need a pathologist.
    th = IntensityThresholds(weak=args.positivity, moderate=args.positivity + 0.25, strong=args.positivity + 0.5)
    an = ERPRAnalyzer(NucleiDetector(args.nuclei), CellTypeClassifier(args.celltype), StainMatrix(), th)

    cases = defaultdict(list)
    for name, ihc, gt, gt_pos in load("BC-DeepLIIF_Validation_Set.zip"):
        cases[name.split("_")[0]].append((name, ihc, gt_pos))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, times = [], defaultdict(list)
    report_done = False
    for cid, fields in sorted(cases.items(), key=lambda kv: -len(kv[1])):
        with tempfile.TemporaryDirectory() as tmp:
            s = CaseSession(out / f"case_{cid}" if not report_done else tmp, f"DEEPLIIF-{cid}", "ER", min_cells=500)
            all_pos = all_n = ref_pos = ref_n = 0
            skipped = 0
            for name, ihc, gt_pos in fields:
                import time
                t0 = time.perf_counter()
                a = an.analyze(ihc, um_per_px=0.25)
                times["pipeline_512px"].append(time.perf_counter() - t0)
                r = s.add_field(ihc, a)
                if not r.accepted:
                    skipped += 1
                    continue
                all_pos += int((a.intensity > 0).sum())
                all_n += len(a)
                ref_pos += int(gt_pos.sum())
                ref_n += len(gt_pos)
            sc = s.score()
            rows.append(dict(case=cid, fields=len(fields) - skipped, skipped_duplicates=skipped,
                             ref_pct_all=100 * ref_pos / max(ref_n, 1), assisted_pct_all=100 * all_pos / max(all_n, 1),
                             assisted_pct_tumour=sc.pct_positive, n_tumour=sc.n_tumour, h_score=sc.h_score,
                             allred=sc.allred_total, status=sc.status, warnings=len(sc.warnings)))
            if not report_done:
                s.save()
                write_report(s, out / f"report_{cid}.pdf", lab="Demo on BC-DeepLIIF (Ki-67) - not ER/PR")
                report_done = True
                print("report:", out / f"report_{cid}.pdf")
    for r in rows:
        print(f"case {r['case']:>3}: {r['fields']} fields ({r['skipped_duplicates']} dup) | reference {r['ref_pct_all']:5.1f}% "
              f"| assisted {r['assisted_pct_all']:5.1f}% (tumour-only {r['assisted_pct_tumour']:5.1f}%, "
              f"{r['n_tumour']} tumour nuclei, H {r['h_score']:.0f}, Allred {r['allred']})")
    print(f"CPU time per 512x512 field (full pipeline): {np.median(times['pipeline_512px']) * 1000:.0f} ms")
    (out / "cases.json").write_text(json.dumps(dict(cases=rows, note="demo only - not a validation"), indent=2))


if __name__ == "__main__":
    main()
