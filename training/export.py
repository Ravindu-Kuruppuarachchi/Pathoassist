"""Export a trained StarDist2D model for the CPU engine.

Writes, next to each other in --out-dir:
    <name>.onnx            FP32, dynamic N/H/W, outputs named "prob" and "dist"
    <name>_fp32.xml/.bin   OpenVINO IR (FP32)
    <name>_int8.xml/.bin   OpenVINO IR, INT8 post-training quantised with NNCF
    one <stem>.json per model file with grid, native µm/px and thresholds

Usage:
    python training/export.py --model pretrained --name stardist_he_pretrained
    python training/export.py --model training/runs/he_ft --name stardist_he_ft
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from training.data import load_split, normalize


def load_stardist(model):
    from stardist.models import StarDist2D
    if model == "pretrained":
        return StarDist2D.from_pretrained("2D_versatile_he")
    p = Path(model)
    return StarDist2D(None, name=p.name, basedir=str(p.parent))


def to_onnx(sd, onnx_path):
    import tensorflow as tf
    import tf2onnx
    km = sd.keras_model  # StarDist already names its output layers "prob" and "dist"
    spec = (tf.TensorSpec((None, None, None, 3), tf.float32, name="image"),)
    tf2onnx.convert.from_keras(km, input_signature=spec, opset=17, output_path=str(onnx_path))
    import onnx
    m = onnx.load(str(onnx_path))
    names = [o.name for o in m.graph.output]
    assert names == ["prob", "dist"], names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="'pretrained' or a StarDist model dir")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--um-per-px", type=float, default=0.25, help="native pixel size the model expects")
    ap.add_argument("--calib", type=int, default=300, help="# calibration images for INT8")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = load_stardist(args.model)
    meta = dict(grid=list(sd.config.grid), n_rays=sd.config.n_rays, um_per_px=args.um_per_px,
                prob_thresh=float(sd.thresholds.prob), nms_thresh=float(sd.thresholds.nms),
                norm_percentiles=[1.0, 99.8], source=str(args.model))

    onnx_path = out / f"{args.name}.onnx"
    to_onnx(sd, onnx_path)
    (out / f"{args.name}.json").write_text(json.dumps(meta, indent=2))
    print("wrote", onnx_path)

    import nncf
    import openvino as ov
    ov_model = ov.convert_model(str(onnx_path))
    fp32 = out / f"{args.name}_fp32.xml"
    ov.save_model(ov_model, str(fp32), compress_to_fp16=False)
    shutil.copy(out / f"{args.name}.json", fp32.with_suffix(".json"))
    print("wrote", fp32)

    # Calibrate on training images (never the test fold), same normalisation as inference.
    X, _, _ = load_split("train")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(args.calib, len(X)), replace=False)
    calib = [normalize(X[i])[None].astype(np.float32) for i in idx]
    q = nncf.quantize(ov_model, nncf.Dataset(calib), preset=nncf.QuantizationPreset.MIXED,
                      subset_size=len(calib))
    int8 = out / f"{args.name}_int8.xml"
    ov.save_model(q, str(int8))
    shutil.copy(out / f"{args.name}.json", int8.with_suffix(".json"))
    print("wrote", int8)


if __name__ == "__main__":
    main()
