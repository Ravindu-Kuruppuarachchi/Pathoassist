"""Export the 10x hotspot model and evaluate hotspot guidance on the 33 MIDOG++ test slides.

Each test slide (~2 mm², 40x) is downscaled to 1 µm/px (a 10x view). HotspotGuide picks
the best HPF disc (0.50 mm field diameter). We compare the ground-truth mitoses inside
the chosen disc with the best possible disc on that slide and with an average disc:
  capture ratio = GT(chosen) / GT(best);  lift = GT(chosen) / GT(mean disc)
plus the Spearman correlation between predicted and true disc counts over a grid.

    ~/venvs/pathoassist-torch/bin/python training/mitosis/eval_hotspot.py
"""
import json
import sys

import cv2
import numpy as np
import torch
from scipy.stats import spearmanr

from common import D, ROOT, STRIDE, HeatmapDetector, load_full

sys.path.insert(0, str(ROOT))
from core.mitosis.hotspot import HotspotGuide  # noqa: E402

HPF_DIAM_UM = 500.0


def export():
    m = HeatmapDetector(pretrained=False)
    m.load_state_dict(torch.load(ROOT / "models/mitosis_hotspot.pt", map_location="cpu"))
    m.eval()
    torch.onnx.export(m, torch.zeros(1, 3, 512, 512), ROOT / "models/mitosis_hotspot.onnx", input_names=["image"],
                      output_names=["heatmap"], dynamic_axes={"image": {2: "h", 3: "w"}, "heatmap": {2: "h4", 3: "w4"}},
                      opset_version=17, dynamo=False)
    (ROOT / "models/mitosis_hotspot.json").write_text(json.dumps(dict(
        um_per_px=1.0, stride=STRIDE, source="MIDOG++ human breast train split, downscaled to 1 µm/px (10x)"), indent=2))


def gt_in_disc(gt_um, cx, cy, r):
    return int((np.hypot(gt_um[:, 0] - cx, gt_um[:, 1] - cy) <= r).sum()) if len(gt_um) else 0


def main():
    export()
    g = HotspotGuide(ROOT / "models/mitosis_hotspot.onnx")
    r = HPF_DIAM_UM / 2
    rows, rhos = [], []
    for rec in load_full("test"):
        img40 = cv2.cvtColor(cv2.imread(str(D / "full" / rec["file"])), cv2.COLOR_BGR2RGB)
        img10 = cv2.resize(img40, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)   # 1 µm/px
        gt = np.array([[x * 0.25, y * 0.25] for x, y, c in rec["points"] if c == 1]).reshape(-1, 2)  # µm
        spots = g.hotspots(img10, um_per_px=1.0, hpf_diameter_um=HPF_DIAM_UM, top_k=1)
        H, W = img10.shape[:2]
        grid = [(x, y) for y in np.arange(r, H - r + 1, 50) for x in np.arange(r, W - r + 1, 50)]
        truth = np.array([gt_in_disc(gt, x, y, r) for x, y in grid])
        d, k = g.density(img10, 1.0)
        disc = np.zeros((int(2 * r * k) + 1,) * 2, np.float32)
        cv2.circle(disc, (disc.shape[0] // 2,) * 2, int(r * k), 1, -1)
        score = cv2.filter2D(d, -1, disc, borderType=cv2.BORDER_CONSTANT)
        pred = np.array([score[min(int(y * k), score.shape[0] - 1), min(int(x * k), score.shape[1] - 1)] for x, y in grid])
        chosen = gt_in_disc(gt, spots[0].x, spots[0].y, r) if spots else 0
        rows.append(dict(file=rec["file"], scanner=rec["scanner"], total=len(gt), best=int(truth.max()),
                         mean=float(truth.mean()), chosen=chosen))
        if truth.std() > 0:
            rhos.append(spearmanr(truth, pred)[0])
    rows_m = [x for x in rows if x["best"] > 0]
    cap = [x["chosen"] / x["best"] for x in rows_m]
    lift = [x["chosen"] / x["mean"] for x in rows_m if x["mean"] > 0]
    res = dict(n_slides=len(rows), slides_with_mitoses=len(rows_m), hpf_diameter_um=HPF_DIAM_UM,
               mean_capture_ratio=float(np.mean(cap)), median_capture_ratio=float(np.median(cap)),
               hit_best_hpf_within_80pct=float(np.mean(np.array(cap) >= 0.8)),
               mean_lift_vs_average_hpf=float(np.mean(lift)), median_spearman_disc_counts=float(np.median(rhos)),
               per_slide=rows)
    print(json.dumps({k: v for k, v in res.items() if k != "per_slide"}, indent=2))
    (ROOT / "eval" / "results" / "mitosis_hotspot_test.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
