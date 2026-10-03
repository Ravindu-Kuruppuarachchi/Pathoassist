"""CPU latency per 1920x1080 camera frame for Features 2 and 6 (real tissue mosaics).

    python eval/benchmark_features.py --threads 4 8 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.ihc.celltype import CellTypeClassifier
from core.ihc.pipeline import ERPRAnalyzer
from core.mitosis.detector import MitosisDetector
from core.mitosis.hotspot import HotspotGuide
from core.nuclei import NucleiDetector


def med(f, reps=5):
    f()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 0])
    args = ap.parse_args()
    # IHC-like frame: BC-DeepLIIF field tiled to 1080p; H&E frame: a MIDOG test slide crop
    sys.path.insert(0, str(ROOT / "eval"))
    from evaluate_ihc_deepliif import load
    ihc = next(load("BC-DeepLIIF_Validation_Set.zip", originals_only=True))[1]
    ihc_frame = np.tile(ihc, (3, 4, 1))[:1080, :1920]
    he = cv2.cvtColor(cv2.imread(str(ROOT / "training/data/midog/full/006.jpg")), cv2.COLOR_BGR2RGB)
    he_frame = he[1000:2080, 1000:2920]
    he10x = cv2.resize(he, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)   # ~1 µm/px

    rows = []
    for th in args.threads:
        t = th or None
        an = ERPRAnalyzer(NucleiDetector(ROOT / "models/stardist_he_ihc_ft_int8.xml", num_threads=t),
                          CellTypeClassifier(ROOT / "models/celltype_mnv3s_ctx32.onnx", num_threads=t))
        md = MitosisDetector(ROOT / "models/mitosis_det.onnx", ROOT / "models/mitosis_ver.onnx", num_threads=t)
        hg = HotspotGuide(ROOT / "models/mitosis_hotspot.onnx", num_threads=t)
        n = len(an.analyze(ihc_frame, 0.25))
        r = dict(threads=th or "all", erpr_frame_s=med(lambda: an.analyze(ihc_frame, 0.25)), erpr_nuclei=n,
                 mitosis_frame_s=med(lambda: md.detect(he_frame, 0.25)),
                 hotspot_10x_s=med(lambda: hg.hotspots(he10x, 1.0, 500)), hotspot_px=list(he10x.shape[:2]))
        rows.append(r)
        print(f"threads={str(r['threads']):4s} ER/PR 1080p frame {r['erpr_frame_s']:.2f}s ({n} nuclei) | "
              f"mitosis 1080p frame {r['mitosis_frame_s']:.2f}s | hotspot 10x view {r['hotspot_10x_s']:.2f}s", flush=True)
    (ROOT / "eval/results/benchmark_features.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
