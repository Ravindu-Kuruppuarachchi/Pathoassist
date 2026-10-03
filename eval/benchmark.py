"""CPU latency of the full detection pipeline on camera-sized frames.

Frames are mosaics of real PanNuke test tissue (so NMS sees a realistic nucleus density).
Thread caps approximate a lab desktop; this workstation's 40 threads are not representative.

    python eval/benchmark.py models/stardist_he_ft_int8.xml models/stardist_he_ft_fp32.xml
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from core.nuclei import NucleiDetector
from training.data import load_split

FRAMES = {"1920x1080": (1080, 1920), "2448x2048 (5MP)": (2048, 2448)}


def mosaic(X, shape, seed=0):
    rng = np.random.default_rng(seed)
    H, W = shape
    out = np.zeros((-(-H // 256) * 256, -(-W // 256) * 256, 3), np.uint8)
    for y in range(0, out.shape[0], 256):
        for x in range(0, out.shape[1], 256):
            out[y:y + 256, x:x + 256] = X[rng.integers(len(X))]
    return out[:H, :W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, nargs="+", default=[4, 8, 0])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    X, _, T = load_split("test")
    X = X[T == "Breast"]
    rows = []
    for name, shape in FRAMES.items():
        frame = mosaic(X, shape)
        for model in args.models:
            for th in args.threads:
                det = NucleiDetector(model, num_threads=th or None)
                det.detect(frame)  # warm-up
                ts = [det.detect(frame).timings for _ in range(args.reps)]
                d = det.detect(frame)
                med = {k: float(np.median([t[k] for t in ts])) for k in ts[0]}
                rows.append(dict(frame=name, model=Path(model).name, threads=th or "all",
                                 nuclei=len(d), **{k: round(v, 3) for k, v in med.items()}))
                r = rows[-1]
                print(f"{name:16s} {r['model']:32s} threads={str(r['threads']):4s} nuclei={r['nuclei']:5d} "
                      f"total={r['total']:.3f}s (pre {r['preprocess']:.3f} net {r['network']:.3f} "
                      f"post {r['postprocess']:.3f})", flush=True)
    out = ROOT / "eval" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
