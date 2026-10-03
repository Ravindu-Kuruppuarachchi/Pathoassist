"""Run nuclei detection on image files and save an outline overlay + per-nucleus CSV.

    python app/detect_image.py frame.png --um-per-px 0.25
    python app/detect_image.py frames/*.jpg --um-per-px 0.18 --model models/stardist_he_ft_int8.xml

Outputs <name>_nuclei.png (outlines) and <name>_nuclei.csv (y, x, area_um2, score) next
to each input, or in --out-dir.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from core.nuclei import NucleiDetector, draw_overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--um-per-px", type=float, required=True, help="frame scale from calibration")
    ap.add_argument("--model", default=str(ROOT / "models" / "stardist_he_ihc_ft_int8.xml"))
    ap.add_argument("--prob-thresh", type=float, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    det = NucleiDetector(args.model, prob_thresh=args.prob_thresh)
    for path in map(Path, args.images):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip {path}: not an image")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        d = det.detect(rgb, um_per_px=args.um_per_px)
        area_px = np.bincount(d.labels.ravel(), minlength=len(d) + 1)[1:]
        out_dir = Path(args.out_dir) if args.out_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{path.stem}_nuclei.png"), cv2.cvtColor(draw_overlay(rgb, d), cv2.COLOR_RGB2BGR))
        pd.DataFrame(dict(y=d.points[:, 0].round(1), x=d.points[:, 1].round(1),
                          area_um2=(area_px * args.um_per_px ** 2).round(2), score=d.prob.round(3))
                     ).to_csv(out_dir / f"{path.stem}_nuclei.csv", index=False)
        print(f"{path.name}: {len(d)} nuclei in {d.timings['total']:.2f}s")


if __name__ == "__main__":
    main()
