"""Export the mitosis detector + verifier to ONNX and evaluate the CPU runtime on the
33 held-out MIDOG++ breast test slides.

Reports (MIDOG protocol, match radius 7.5 µm):
  - detection precision / recall / F1, detector alone and detector + verifier
  - per-slide count agreement (Spearman) and mitotic-score agreement: each slide (~2 mm²)
    is treated as a 10-HPF count at a 0.50 mm field diameter after area normalisation,
    scored 1-3 from ground-truth vs predicted counts -> weighted kappa

    ~/venvs/pathoassist-torch/bin/python training/mitosis/export_and_eval.py
"""
import json
import sys
import time

import cv2
import numpy as np
import torch

from common import D, ROOT, STRIDE, HeatmapDetector, load_full, match
from train_verifier import CAND_THRESH, VCROP

sys.path.insert(0, str(ROOT))
from core.ihc.scoring import weighted_kappa  # noqa: E402
from core.mitosis.counting import mitotic_score  # noqa: E402
from core.mitosis.detector import MitosisDetector  # noqa: E402


def export():
    import timm
    out = ROOT / "models"
    det = HeatmapDetector(pretrained=False)
    det.load_state_dict(torch.load(out / "mitosis_det.pt", map_location="cpu"))
    det.eval()
    torch.onnx.export(det, torch.zeros(1, 3, 512, 512), out / "mitosis_det.onnx", input_names=["image"],
                      output_names=["heatmap"], dynamic_axes={"image": {2: "h", 3: "w"}, "heatmap": {2: "h4", 3: "w4"}},
                      opset_version=17, dynamo=False)
    ver = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    ver.load_state_dict(torch.load(out / "mitosis_ver.pt", map_location="cpu"))
    ver.eval()
    torch.onnx.export(ver, torch.zeros(1, 3, VCROP, VCROP), out / "mitosis_ver.onnx", input_names=["crops"],
                      output_names=["logit"], dynamic_axes={"crops": {0: "n"}, "logit": {0: "n"}}, opset_version=17,
                      dynamo=False)
    dv = json.load(open(out / "mitosis_det_val.json"))
    vv = json.load(open(out / "mitosis_ver_val.json"))["best"]
    meta = dict(um_per_px=0.25, stride=STRIDE, cand_thresh=CAND_THRESH, det_thresh=vv["det_thresh"],
                ver_thresh=vv["ver_thresh"], det_only_thresh=dv["thresh"], ver_crop=VCROP,
                source="MIDOG++ human breast (train split), thresholds tuned on 15 val slides")
    (out / "mitosis_det.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    meta = export()
    print("exported:", meta, flush=True)
    recs = load_full("test")
    runs = {"detector_only": MitosisDetector(ROOT / "models/mitosis_det.onnx"),
            "detector+verifier": MitosisDetector(ROOT / "models/mitosis_det.onnx", ROOT / "models/mitosis_ver.onnx")}
    res = {}
    for name, md in runs.items():
        tp = fp = fn = 0
        gt_counts, pr_counts, times, areas = [], [], [], []
        for r in recs:
            img = cv2.cvtColor(cv2.imread(str(D / "full" / r["file"])), cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            cands, _ = md.detect(img, um_per_px=0.25)
            times.append(time.perf_counter() - t0)
            acc = sorted([c for c in cands if c.accepted], key=lambda c: -c.ver_score * c.det_score)
            pred = np.array([[c.x, c.y] for c in acc]).reshape(-1, 2)
            gt = np.array([[x, y] for x, y, c in r["points"] if c == 1]).reshape(-1, 2)
            a, b, c, _ = match(pred, gt)
            tp, fp, fn = tp + a, fp + b, fn + c
            gt_counts.append(len(gt))
            pr_counts.append(len(pred))
            areas.append(img.shape[0] * img.shape[1] * (0.25 / 1000) ** 2)
        from scipy.stats import spearmanr
        ref_area = 10 * np.pi * 0.25 ** 2                     # 10 HPFs of 0.50 mm diameter
        gs = [mitotic_score(g * ref_area / a, 0.50)["score"] for g, a in zip(gt_counts, areas)]
        ps = [mitotic_score(p * ref_area / a, 0.50)["score"] for p, a in zip(pr_counts, areas)]
        res[name] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                         f1=2 * tp / max(2 * tp + fp + fn, 1),
                         count_spearman=float(spearmanr(gt_counts, pr_counts)[0]),
                         score_agreement=float(np.mean(np.array(gs) == np.array(ps))),
                         score_weighted_kappa=float(weighted_kappa(np.array(gs) - 1, np.array(ps) - 1, n=3)),
                         score_confusion=np.histogram2d(gs, ps, bins=[[0.5, 1.5, 2.5, 3.5]] * 2)[0].astype(int).tolist(),
                         cpu_sec_per_slide=float(np.mean(times)), slide_px=list(img.shape[:2]),
                         per_slide=[dict(file=r["file"], scanner=r["scanner"], gt=g, pred=p)
                                    for r, g, p in zip(recs, gt_counts, pr_counts)])
        s = res[name]
        print(f"{name:18s} P {s['precision']:.3f} R {s['recall']:.3f} F1 {s['f1']:.3f} | count rho "
              f"{s['count_spearman']:.3f} | score agree {s['score_agreement']:.2f} kappa {s['score_weighted_kappa']:.3f} "
              f"| {s['cpu_sec_per_slide']:.1f}s/slide", flush=True)
    (ROOT / "eval" / "results" / "mitosis_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
