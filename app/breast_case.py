"""Run a complete breast case from folders of saved frames and write the combined report
(Features 6-9 + 2-4; until the live capture app exists).

    python app/breast_case.py --case-id B-001 --specimen "core biopsy" \
        --tubules frames/he_10x --um10 0.97 \
        --pleomorphism frames/he_40x --hpf frames/he_hpf --um40 0.24 \
        --er frames/er --pr frames/pr --her2 frames/her2 --ki67 frames/ki67 --um-ihc 0.24 \
        --pathologist "Dr A" [--override mitoses=3]

Any folder may be omitted (that module is skipped; a missing grade component can be given with
--override). The grade is only signed when --pathologist is given and all components are present.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

M = ROOT / "models"


def frames(folder):
    if not folder:
        return []
    out = []
    for p in sorted(Path(folder).iterdir()):
        im = cv2.imread(str(p))
        if im is not None:
            out.append((p.name, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--specimen", default="core biopsy", choices=["core biopsy", "excision", "other"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--tubules"), ap.add_argument("--pleomorphism"), ap.add_argument("--hpf")
    ap.add_argument("--er"), ap.add_argument("--pr"), ap.add_argument("--her2"), ap.add_argument("--ki67")
    ap.add_argument("--um10", type=float, default=1.0, help="µm/px of the 10x frames")
    ap.add_argument("--um40", type=float, default=0.25, help="µm/px of the 40x frames")
    ap.add_argument("--um-ihc", type=float, default=0.25, help="µm/px of the IHC frames")
    ap.add_argument("--um-her2", type=float, default=None, help="µm/px of the HER2 frames (default: --um-ihc)")
    ap.add_argument("--override", action="append", default=[], help="component=score, e.g. mitoses=3")
    ap.add_argument("--pathologist", default=None)
    ap.add_argument("--lab", default="")
    args = ap.parse_args()

    from core.grading.case import BreastCase, GradingWizard
    from core.grading.report import write_breast_report
    from core.nuclei import NucleiDetector
    out = Path(args.out or f"cases/{args.case_id}")
    case = BreastCase(out, args.case_id, args.specimen)
    det = NucleiDetector(M / "stardist_he_ihc_ft_int8.xml")

    tub = pleo = hpf = mdet = None
    if args.tubules:
        from core.region.tissue_map import TissueRegionModel
        from core.tubules.lumen import LumenClassifier
        from core.tubules.score import TubuleSession
        tub = TubuleSession(LumenClassifier(M / "tubule_lumen_lgbm.txt"), TissueRegionModel(M / "region_effb0.onnx"))
    if args.pleomorphism:
        from core.ihc.celltype import CellTypeClassifier
        from core.pleomorphism.embedding import TileEmbedder
        from core.pleomorphism.scorer import PleomorphismScorer, PleomorphismSession
        pleo = PleomorphismSession(PleomorphismScorer(M / "pleomorphism_phikon.npz", det, CellTypeClassifier(M / "celltype_mnv3s_ctx32.onnx"),
                                                      embedder=TileEmbedder(M / "phikon_v2_int8w.xml")))
    hpf_frames = frames(args.hpf)
    if hpf_frames:
        from core.mitosis.counting import HPFCounter
        from core.mitosis.detector import MitosisDetector
        hpf = HPFCounter(args.um40, hpf_frames[0][1].shape[:2])
        mdet = MitosisDetector(M / "mitosis_det.onnx", M / "mitosis_ver.onnx")
    w = GradingWizard(case, tub, pleo, hpf, mdet)
    overrides = dict(o.split("=") for o in args.override)

    for step, folder, um in (("tubules", args.tubules, args.um10), ("pleomorphism", args.pleomorphism, args.um40),
                             ("mitoses", args.hpf, args.um40)):
        fr = hpf_frames if step == "mitoses" else frames(folder)
        if fr and w.engines.get(step) is not None:
            for name, im in fr:
                r = w.add_frame(im, um)
                print(f"[{step}] {name}: {'ok' if r.get('accepted') else r.get('reason')}")
            case.add_snapshot(cv2.resize(fr[0][1], (800, int(800 * fr[0][1].shape[0] / fr[0][1].shape[1]))), f"{step} - {fr[0][0]}", step)
            w.finish_step()
            c = case.grade.c[step]
            print(f"  -> {step} score {c.auto_score}: {c.value}")
        elif step in overrides:
            w.enter_manual(int(overrides.pop(step)), "entered via --override")
        else:
            print(f"[{step}] no frames and no override - component missing")
            w.i += 1
    for k, v in overrides.items():
        w.edit(k, int(v), "override via --override")

    # biomarkers
    if args.er or args.pr:
        from core.ihc.case import CaseSession
        from core.ihc.pipeline import ERPRAnalyzer
        an = ERPRAnalyzer(det)                    # cell-type filter off on IHC (classifier not validated on IHC)
        for name, folder in (("ER", args.er), ("PR", args.pr)):
            if folder:
                s = CaseSession(out / name, args.case_id, name)
                for _, im in frames(folder):
                    s.add_field(im, an.analyze(im, args.um_ihc))
                case.set_biomarker(name, s.score(), validated=False, method="DAB nuclear OD, all nuclei")
    if args.her2:
        from core.her2.pipeline import HER2Analyzer
        from core.her2.scoring import HER2Thresholds
        from core.her2.session import HER2Session
        an = HER2Analyzer(det, thresholds=HER2Thresholds.load(M / "her2_thresholds_bci.json"))
        s = HER2Session(out / "HER2", args.case_id)
        for _, im in frames(args.her2):
            s.add_field(im, an.analyze(im, args.um_her2 or args.um_ihc))
        case.set_biomarker("HER2", s.result(), validated=False, method="membrane ring, ASCO/CAP rules")
    if args.ki67:
        from core.ki67.session import Ki67Session, ki67_analyzer
        an = ki67_analyzer(det)
        s = Ki67Session(out / "Ki67", args.case_id, celltype_filter=False)
        for _, im in frames(args.ki67):
            s.add_field(im, an.analyze(im, args.um_ihc))
        case.set_biomarker("Ki67", {"global": s.global_result().as_dict(), "hotspot": s.hotspot_result().as_dict()},
                           validated=False, method="DAB positivity, global + hotspot")

    g = case.grade
    print(f"\nGrade: {g.grade if g.complete else 'incomplete'} (total {g.total}) - " +
          ", ".join(f"{k} {c.score}" for k, c in g.c.items()))
    if args.pathologist and g.complete:
        w.i = [s for s, _ in w.STEPS].index("review")
        w.sign(args.pathologist)
    case.save()
    pdf = write_breast_report(case, out / f"report_{args.case_id}.pdf", lab=args.lab)
    print("report:", pdf, "| audit ok:", case.audit.verify()[0])


if __name__ == "__main__":
    main()
