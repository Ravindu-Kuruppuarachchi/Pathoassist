"""Sample outputs for Features 4 (HER2) and 5 (tissue regions, auto-magnification, mosaic).

    python eval/demo_features45.py
Writes demo_outputs/4_*.jpg, 5_*.jpg and prints the numbers.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.her2.pipeline import HER2Analyzer, field_counts
from core.her2.scoring import HER2Thresholds, her2_score
from core.her2.session import draw_her2_overlay
from core.nuclei import NucleiDetector
from core.region.mosaic import Mosaic
from core.region.tissue_map import MagnificationEstimator, TissueRegionModel, draw_region_overlay

OUT = ROOT / "demo_outputs"
FONT = cv2.FONT_HERSHEY_SIMPLEX
BCI = ROOT / "training/data/bci"
BACH = ROOT / "training/data/bach"


def label(img, text, scale=0.8):
    img = img.copy()
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.rectangle(img, (0, 0), (w + 14, h + 14), (255, 255, 255), -1)
    cv2.putText(img, text, (7, h + 7), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def save(name, rgb):
    cv2.imwrite(str(OUT / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("  wrote", OUT / name)


def her2_demo():
    print("\n=== Feature 4: HER2 (BCI held-out test images, ~1 µm/px, expert-refined labels) ===")
    th_path = ROOT / "models/her2_thresholds_bci.json"
    th = HER2Thresholds.load(th_path) if th_path.exists() else HER2Thresholds()
    an = HER2Analyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml"), None, thresholds=th)
    clf = None
    if (ROOT / "models/her2_field_effb0.onnx").exists():
        import onnxruntime as ort
        clf = ort.InferenceSession(str(ROOT / "models/her2_field_effb0.onnx"), providers=["CPUExecutionProvider"])
    ref = {r["pair_id"]: r["her2_level_refined"] for r in csv.DictReader(open(BCI / "refined_labels.csv")) if r["partition"] == "test"}
    recs = [r for r in json.loads((BCI / "index.json").read_text()) if r["split"] == "test" and r["id"] in ref]
    tiles = []
    print(f"  {'image':>5} | refined label | rule engine (cells, any/complete/strong %) | cross-check classifier")
    for lab in ["0", "1+", "2+", "3+"]:
        for r in [x for x in recs if ref[x["id"]] == lab][:2]:
            im = cv2.cvtColor(cv2.imread(str(BCI / "ihc" / r["file"])), cv2.COLOR_BGR2RGB)
            a = an.analyze(im, 1.0)
            c, ii = field_counts(a)
            res = her2_score(c, ii, min_cells=0)
            cc = ""
            if clf is not None:
                x = cv2.resize(im[256:768, 256:768].astype(np.float32) / 255, (384, 384), interpolation=cv2.INTER_AREA)
                x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
                cc = ["0", "1+", "2+", "3+"][int(clf.run(None, {"field": x})[0].argmax())]
            agree = "" if not cc else ("agree" if cc == res.score else "DISAGREE -> review carefully")
            print(f"  {r['id']:>5} | {lab:>13} | {res.score:>3} {res.subgroup:9s} ({res.n_cells:4d} cells, {res.pct_any_staining:5.1f}/"
                  f"{res.pct_complete:5.1f}/{res.pct[3]:5.1f}) | {cc} {agree}")
            ov = draw_her2_overlay(im, a)[256:768, 256:768]
            tiles.append(label(np.hstack([im[256:768, 256:768], ov]), f"label {lab} | rules {res.score} | clf {cc}", 0.7))
    save("4_her2_cells.jpg", np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    print("  overlay: blue = no staining, yellow = faint/incomplete, orange = weak-moderate complete, red = strong complete")


def region_demo():
    print("\n=== Feature 5: simulated 4-5x sweep over held-out BACH slide A10 (1.87 µm/px) ===")
    slide = cv2.cvtColor(cv2.imread(str(BACH / "demo_A10_level1.jpg")), cv2.COLOR_BGR2RGB)
    gt = cv2.imread(str(BACH / "mask_A10.png"), 0)              # 1/32 of level 0 = 1/8 of level 1
    um = 1.87
    rm = TissueRegionModel(ROOT / "models/region_effb0.onnx")
    objectives = {"5x": 1.87, "10x": 0.93, "20x": 0.47, "40x": 0.23}
    mag = MagnificationEstimator(ROOT / "models/magnification_mnv3.onnx", objectives) \
        if (ROOT / "models/magnification_mnv3.onnx").exists() else None
    mos = Mosaic(map_um_per_px=8.0)
    fh, fw = 768, 1024
    H, W = slide.shape[:2]
    # sweep the tissue bounding box
    hsv = cv2.cvtColor(cv2.resize(slide, (W // 16, H // 16)), cv2.COLOR_RGB2HSV)
    ys, xs = np.nonzero(hsv[..., 1] > 25)
    y0, y1, x0, x1 = ys.min() * 16, min(H - fh, ys.max() * 16), xs.min() * 16, min(W - fw, xs.max() * 16)
    rng = np.random.default_rng(0)
    stats, mags, frames_shown = [], [], []
    statuses = {}
    for r, y in enumerate(range(y0, y1, 600)):
        for x in (range(x0, x1, 800) if r % 2 == 0 else range(x1 - 1, x0, -800)):
            fy = int(np.clip(y + rng.integers(-40, 41), 0, H - fh))
            fx = int(np.clip(x + rng.integers(-40, 41), 0, W - fw))
            fr = slide[fy:fy + fh, fx:fx + fw]
            p = mos.add_frame(fr, um)
            statuses[p.status] = statuses.get(p.status, 0) + 1
            if p.status == "no_tissue":
                continue
            m = rm.predict(fr, um)
            mos.put_overlay(p, m.tumour_prob())
            if mag is not None:
                mags.append(mag.estimate(fr)[0])
            g = cv2.resize(gt[fy // 8:(fy + fh) // 8, fx // 8:(fx + fw) // 8], m.probs.shape[1::-1], interpolation=cv2.INTER_NEAREST)
            pred = m.probs.argmax(-1)
            tissue = g != 4
            if tissue.sum() > 20:
                stats.append(((g <= 1) & tissue, (pred <= 1) & tissue, tissue))
            if len(frames_shown) < 2 and ((g <= 1) & tissue).mean() > 0.2:
                frames_shown.append(label(draw_region_overlay(fr, m), f"frame {len(mos.placements)}: {', '.join(f'{k} {v:.0%}' for k, v in m.fractions().items() if v > 0.05)}", 0.6))
    T = np.concatenate([s[0][s[2]] for s in stats])
    P = np.concatenate([s[1][s[2]] for s in stats])
    print(f"  frames: {sum(statuses.values())} ({statuses}); final map: {mos.n_islands} island(s) after {mos.merges} merges, covers {mos.coverage_mm2():.1f} mm²")
    print(f"  tumour (invasive + in situ) vs annotation, per 112 µm tissue cell: accuracy {np.mean(T == P):.3f}, "
          f"sensitivity {np.mean(P[T]):.3f}, specificity {np.mean(~P[~T]):.3f}  (n={len(T)})")
    if mags:
        vals, cnt = np.unique([m for m in mags if m], return_counts=True)
        print(f"  auto-magnification on {len(mags)} frames (true 5x): " + ", ".join(f"{v} {c}" for v, c in zip(vals, cnt)))
    mm = mos.minimap(700)
    gt_vis = np.full((*gt.shape, 3), 235, np.uint8)
    gt_vis[gt == 3] = (230, 200, 220)
    gt_vis[gt == 2] = (0, 170, 0)
    gt_vis[gt == 1] = (150, 0, 200)
    gt_vis[gt == 0] = (220, 0, 0)
    gt_vis = cv2.resize(gt_vis, (700, int(gt.shape[0] * 700 / gt.shape[1])), interpolation=cv2.INTER_NEAREST)
    h = max(mm.shape[0], gt_vis.shape[0])
    pad = lambda a: cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    save("5a_minimap_vs_annotation.jpg", np.hstack([label(pad(mm), "mosaic mini-map, red = predicted tumour", 0.6),
                                                   label(pad(gt_vis), "pathologist annotation (red inv, purple in situ, green benign)", 0.6)]))
    if frames_shown:
        save("5b_region_frames.jpg", np.vstack(frames_shown))


def main():
    OUT.mkdir(exist_ok=True)
    her2_demo()
    region_demo()


if __name__ == "__main__":
    main()
