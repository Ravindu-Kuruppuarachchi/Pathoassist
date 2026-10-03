"""Fine-tune StarDist 2D_versatile_he on PanNuke (folds 1+2) with heavy augmentation.

    CUDA_VISIBLE_DEVICES=0 python training/train_stardist.py --name he_ft --epochs 100

Starts from the pretrained weights, trains, then re-optimises the prob/NMS thresholds
on the validation split. Output: training/runs/<name>/ (weights_best.h5, thresholds.json).
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from stardist.models import StarDist2D
from csbdeep.models.pretrained import get_model_folder

from training.augment import augmenter
from training.data import load_split, normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="he_ft")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--init", default=None, help="start from this run dir instead of the pretrained model")
    ap.add_argument("--deepliif", type=int, default=0, help="add BC-DeepLIIF IHC training fields, oversampled N times")
    args = ap.parse_args()

    runs = ROOT / "training" / "runs"
    run = runs / args.name
    if run.exists():
        raise SystemExit(f"{run} exists - pick another --name")
    src = Path(args.init) if args.init else get_model_folder(StarDist2D, "2D_versatile_he")
    shutil.copytree(src, run, ignore=shutil.ignore_patterns("*.zip", "logs", "weights_last.h5", "weights_now.h5"))
    cfg = json.loads((run / "config.json").read_text())
    cfg.update(train_patch_size=[256, 256], train_learning_rate=args.lr, train_epochs=args.epochs,
               train_steps_per_epoch=args.steps, train_batch_size=args.batch, train_n_val_patches=None,
               train_reduce_lr={"factor": 0.5, "patience": 15, "min_delta": 0}, train_tensorboard=True)
    (run / "config.json").write_text(json.dumps(cfg))
    model = StarDist2D(None, name=args.name, basedir=str(runs))  # loads pretrained weights_best.h5

    X, Y, _ = load_split("train")
    Xv, Yv, _ = load_split("val")
    X = [x.astype(np.float32) / 255 for x in X]          # raw; augmenter normalises
    Xv = [normalize(x) for x in Xv]
    Y, Yv = list(Y), list(Yv)
    if args.deepliif:
        sys.path.insert(0, str(ROOT / "eval"))
        from evaluate_ihc_deepliif import load as load_deepliif
        ihc = [(x, y) for _, x, y, _ in load_deepliif("BC-DeepLIIF_Training_Set.zip")]
        X += [x.astype(np.float32) / 255 for x, _ in ihc] * args.deepliif
        Y += [y.astype(np.uint16) for _, y in ihc] * args.deepliif
        print(f"added {len(ihc)} BC-DeepLIIF IHC fields x{args.deepliif}")
    print(f"train {len(X)}  val {len(Xv)}")

    model.train(X, Y, validation_data=(Xv, Yv), augmenter=augmenter, workers=args.workers)
    model.optimize_thresholds(Xv, Yv)
    print("thresholds:", model.thresholds)


if __name__ == "__main__":
    main()
