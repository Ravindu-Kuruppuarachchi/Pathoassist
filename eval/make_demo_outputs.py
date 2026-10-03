"""Sample outputs for Features 1, 2 and 6 on held-out public data -> demo_outputs/.

    python eval/make_demo_outputs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import cv2
import numpy as np
from skimage.segmentation import find_boundaries

from core.ihc.celltype import CellTypeClassifier
from core.ihc.overlay import draw_erpr_overlay
from core.ihc.pipeline import CELL_TYPES, TUMOUR, ERPRAnalyzer
from core.ihc.scoring import IntensityThresholds, score_counts
from core.ihc.stain import StainMatrix
from core.mitosis.counting import HPFCounter
from core.mitosis.detector import MitosisDetector
from core.mitosis.hotspot import HotspotGuide
from core.nuclei import NucleiDetector, draw_overlay
from evaluate_ihc_deepliif import load

OUT = ROOT / "demo_outputs"
MIDOG = ROOT / "training/data/midog"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def save(name, rgb):
    cv2.imwrite(str(OUT / name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("  wrote", OUT / name)


def label(img, text, scale=0.9):
    img = img.copy()
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.rectangle(img, (0, 0), (w + 16, h + 16), (255, 255, 255), -1)
    cv2.putText(img, text, (8, h + 8), FONT, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def midog_slide(file):
    rec = next(r for r in json.loads((MIDOG / "full.json").read_text()) if r["file"] == file)
    img = cv2.cvtColor(cv2.imread(str(MIDOG / "full" / file)), cv2.COLOR_BGR2RGB)
    return img, rec


# ----------------------------------------------------------------------------- Feature 1
def nuclei_demo():
    print("\n=== Feature 1: nuclei detection (H&E, MIDOG++ test slide, 40x) ===")
    det = NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml")
    img, _ = midog_slide("006.jpg")
    frame = img[1000:2080, 1000:2920]                     # one 1920x1080 camera-sized frame
    d = det.detect(frame, um_per_px=0.25)
    area = np.bincount(d.labels.ravel())[1:] * 0.25 ** 2
    print(f"  {len(d)} nuclei in a 1920x1080 frame, {d.timings['total']:.2f}s on CPU; "
          f"median area {np.median(area):.1f} µm²")
    crop = (slice(300, 800), slice(600, 1300))
    save("1_nuclei_he.jpg", np.hstack([label(frame[crop], "input"),
                                       label(draw_overlay(frame, d)[crop], f"{len(d)} nuclei in full frame")]))


# ----------------------------------------------------------------------------- Feature 2
def erpr_demo():
    print("\n=== Feature 2: ER/PR scoring pipeline (real IHC, BC-DeepLIIF held-out fields) ===")
    th = IntensityThresholds(weak=0.52, moderate=0.77, strong=1.02)   # only the 0.52 cut is calibrated
    an = ERPRAnalyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml"),
                      CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx"), StainMatrix(), th)
    rows, panels = [], []
    total = np.zeros(4, int)
    for name, ihc, gt, gt_pos in load("BC-DeepLIIF_Validation_Set.zip", originals_only=True):
        a = an.analyze(ihc, um_per_px=0.25)
        ref = ihc.copy()
        b = find_boundaries(gt, mode="inner")
        pos_ids = np.nonzero(gt_pos)[0] + 1
        ref[b & np.isin(gt, pos_ids)] = (230, 0, 0)
        ref[b & ~np.isin(gt, pos_ids) & (gt > 0)] = (40, 110, 255)
        t = a.cell_type == TUMOUR
        counts = np.bincount(a.intensity[t], minlength=4)[:4]
        total += counts
        s = score_counts(counts, min_cells=0)
        all_pct = 100 * (a.intensity > 0).mean() if len(a) else 0
        types = np.bincount(a.cell_type, minlength=4)
        rows.append((name, len(gt_pos), 100 * gt_pos.mean(), len(a), all_pct, s, types))
        if len(panels) < 3:
            panels.append(np.hstack([label(ihc, f"field {name}", 0.6),
                                     label(draw_erpr_overlay(ihc, a), f"assisted: {all_pct:.0f}% pos", 0.6),
                                     label(ref, f"reference: {100 * gt_pos.mean():.0f}% pos", 0.6)]))
    print(f"  {'field':>5} | ref nuclei  ref %pos | found  assisted %pos | tumour-only: n  %pos  H-score Allred status"
          f"      | cell types (tum/lym/str/oth)")
    for name, ng, rp, nd, ap, s, ty in rows:
        print(f"  {name:>5} | {ng:10d}  {rp:7.1f}% | {nd:5d}  {ap:12.1f}% | {s.n_tumour:14d} {s.pct_positive:5.1f}% "
              f"{s.h_score:7.0f} {s.allred_total:6d} {s.status:12s} | {'/'.join(map(str, ty))}")
    s = score_counts(total, min_cells=500)
    print(f"\n  Case total over {len(rows)} fields (as if one case):")
    print(f"    tumour nuclei {s.n_tumour}, counts neg/weak/mod/strong {s.counts}")
    print(f"    % positive {s.pct_positive}%  H-score {s.h_score}  Allred {s.allred_total} "
          f"(PS {s.allred_ps} + IS {s.allred_is})  status: {s.status}")
    for w in s.warnings:
        print("    warning:", w)
    print("  NOTE: BC-DeepLIIF is Ki-67, not ER/PR; weak/moderate/strong cut-offs are placeholders.")
    save("2_erpr_ihc.jpg", np.vstack(panels))
    print("  legend: assisted outlines blue=negative, yellow=weak, orange=moderate, red=strong, grey=non-tumour;"
          " reference red=positive, blue=negative")


# ----------------------------------------------------------------------------- Feature 6
def mitosis_demo():
    print("\n=== Feature 6: mitosis detection (MIDOG++ held-out test slide 100, 44 true mitoses) ===")
    md = MitosisDetector(ROOT / "models/mitosis_det.onnx", ROOT / "models/mitosis_ver.onnx")
    img, rec = midog_slide("100.jpg")
    cands, tm = md.detect(img, um_per_px=0.25)
    gt = np.array([[x, y] for x, y, c in rec["points"] if c == 1])
    mimics = np.array([[x, y] for x, y, c in rec["points"] if c == 2]).reshape(-1, 2)
    acc = [c for c in cands if c.accepted]
    P = np.array([[c.x, c.y] for c in acc]).reshape(-1, 2)
    d = np.linalg.norm(P[:, None] - gt[None], axis=-1) if len(P) else np.zeros((0, len(gt)))
    tp_mask = (d.min(1) <= 30) if len(P) else np.zeros(0, bool)
    found = int((d.min(0) <= 30).sum()) if len(P) else 0
    print(f"  slide {img.shape[1]}x{img.shape[0]} px ({img.shape[0] * img.shape[1] * 0.25 ** 2 / 1e6:.2f} mm²), "
          f"{tm['total']:.1f}s on CPU")
    print(f"  true mitoses {len(gt)} | detections {len(acc)}: {tp_mask.sum()} correct, {(~tp_mask).sum()} false "
          f"| missed {len(gt) - found} | labelled mimics in slide {len(mimics)}, "
          f"called mitosis: {int(sum(np.linalg.norm(mimics - p, axis=1).min() <= 30 for p in P)) if len(mimics) and len(P) else 0}")

    # gallery of detections and misses (96 px crops, 2x)
    def crop(x, y, r=48):
        pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
        c = pad[int(y):int(y) + 2 * r, int(x):int(x) + 2 * r]
        return cv2.resize(c, (192, 192), interpolation=cv2.INTER_CUBIC)

    def tile(c, txt, col):
        c = c.copy()
        cv2.rectangle(c, (0, 0), (191, 191), col, 6)
        cv2.putText(c, txt, (8, 182), FONT, 0.5, col, 2, cv2.LINE_AA)
        return c

    tps = [tile(crop(c.x, c.y), f"TP {c.ver_score:.2f}", (0, 170, 0)) for c, ok in zip(acc, tp_mask) if ok][:8]
    fps = [tile(crop(c.x, c.y), f"FP {c.ver_score:.2f}", (220, 0, 0)) for c, ok in zip(acc, tp_mask) if not ok][:8]
    miss = [tile(crop(x, y), "missed", (255, 150, 0)) for (x, y), m in zip(gt, d.min(0) > 30 if len(P) else [True] * len(gt)) if m][:8]
    blank = np.full((192, 192, 3), 255, np.uint8)
    rows = [np.hstack((r + [blank] * 8)[:8]) for r in (tps, fps, miss)]
    save("6a_mitosis_gallery.jpg", np.vstack(rows))
    print("  gallery rows: correct detections (green), false positives (red), missed mitoses (orange)")

    # overview with marks
    ov = img.copy()
    for x, y in gt:
        cv2.circle(ov, (int(x), int(y)), 40, (0, 200, 0), 6)
    for c, ok in zip(acc, tp_mask):
        cv2.circle(ov, (int(c.x), int(c.y)), 55, (0, 0, 255) if ok else (255, 0, 0), 6)
    save("6b_mitosis_slide_overview.jpg", label(cv2.resize(ov, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA),
                                                 "green=true mitosis, blue=detected (correct), red=false detection", 0.6))
    return img, rec, md


def hpf_demo(img, rec, md):
    print("\n=== Feature 6: guided 10-HPF count (simulated dwell frames from slide 100) ===")
    fh, fw = 1080, 1440                                    # camera frame at 0.25 µm/px
    counter = HPFCounter(um_per_px=0.25, frame_shape=(fh, fw))
    print(f"  HPF = {fw * 0.25:.0f} x {fh * 0.25:.0f} µm = {counter.hpf_area_mm2:.3f} mm²; "
          f"equivalent field diameter {counter.field_diameter_mm:.3f} mm")
    H, W = img.shape[:2]
    origins = [(y, x) for y in range(0, H - fh + 1, fh) for x in range(0, W - fw + 1, fw)]
    origins.insert(2, (origins[1][0] + 200, origins[1][1] + 300))   # a near-repeat of frame 2 (>50% overlap)
    gt = np.array([[x, y] for x, y, c in rec["points"] if c == 1])
    gt_total = 0
    for i, (y0, x0) in enumerate(origins[:11]):
        frame = img[y0:y0 + fh, x0:x0 + fw]
        cands, _ = md.detect(frame, 0.25)
        r = counter.add_frame(frame, cands)
        n_gt = int(((gt[:, 0] >= x0) & (gt[:, 0] < x0 + fw) & (gt[:, 1] >= y0) & (gt[:, 1] < y0 + fh)).sum())
        if r["accepted"]:
            gt_total += n_gt
            print(f"  frame {i + 1:2d} at ({x0:4d},{y0:4d}): HPF {r['hpf'] + 1:2d}  +{len(r['new'])} mitoses "
                  f"({len(r['duplicates'])} already counted)   [truth in frame: {n_gt}]")
        else:
            print(f"  frame {i + 1:2d} at ({x0:4d},{y0:4d}): skipped - {r['reason']}")
    res = counter.result()
    print(f"\n  Result: {res['mitoses']} mitoses in {res['hpfs_counted']} HPFs ({res['area_mm2']} mm², "
          f"{res['per_mm2']}/mm²) -> mitotic score {res.get('score')} "
          f"(score 1 <= {res.get('score1_max')}, score 3 >= {res.get('score3_min')})   [truth: {gt_total} mitoses]")
    for n in res["warnings"] + res.get("notes", []):
        print("    note:", n)
    # pathologist review: reject one, add one
    if counter.mitoses:
        counter.set_status(0, "rejected")
        counter.add_manual(0, 500, 500)
        r2 = counter.result()
        print(f"  After rejecting 1 detection and adding 1 missed: {r2['mitoses']} mitoses "
              f"(rejected {r2['rejected']}, added {r2['added']}), score {r2.get('score')}")


def hotspot_demo(img, rec):
    print("\n=== Feature 6: 10x hotspot guidance (slide 100 downscaled to 1 µm/px) ===")
    g = HotspotGuide(ROOT / "models/mitosis_hotspot.onnx")
    img10 = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    spots = g.hotspots(img10, um_per_px=1.0, hpf_diameter_um=500, top_k=3)
    gt = np.array([[x / 4, y / 4] for x, y, c in rec["points"] if c == 1])
    ov = img10.copy()
    for x, y in gt:
        cv2.circle(ov, (int(x), int(y)), 6, (0, 200, 0), -1)
    for s in spots:
        n = int((np.hypot(gt[:, 0] - s.x, gt[:, 1] - s.y) <= 250).sum())
        cv2.circle(ov, (int(s.x), int(s.y)), 250, (255, 0, 0), 4)
        cv2.putText(ov, f"#{s.rank}", (int(s.x) - 25, int(s.y) + 12), FONT, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
        print(f"  go-here #{s.rank} at ({s.x:.0f},{s.y:.0f}) µm: {n} true mitoses inside the HPF")
    best = max(int((np.hypot(gt[:, 0] - x, gt[:, 1] - y) <= 250).sum())
               for y in range(250, img10.shape[0] - 249, 25) for x in range(250, img10.shape[1] - 249, 25))
    print(f"  best possible HPF on this slide: {best} mitoses")
    save("6c_hotspot_10x.jpg", label(ov, "green=true mitoses, red circles=suggested HPFs (0.50 mm)", 0.7))


def main():
    OUT.mkdir(exist_ok=True)
    nuclei_demo()
    erpr_demo()
    img, rec, md = mitosis_demo()
    hpf_demo(img, rec, md)
    hotspot_demo(img, rec)


if __name__ == "__main__":
    main()
