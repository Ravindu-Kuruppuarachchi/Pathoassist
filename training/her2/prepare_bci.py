"""Pull HER2 IHC images from the BCI dataset (Liu et al., CVPR-W 2022) without downloading the
whole 2.9 GB zip: the zip is read over HTTP range requests and only IHC members are fetched.

BCI: breast HER2 IHC patches (1024x1024, 20x), label 0 / 1+ / 2+ / 3+ in the file name.
Licence: non-commercial research use only -> used here for calibration / validation, and the
optional cross-check classifier trained on it must not ship in a commercial product.

    ~/venvs/pathoassist-torch/bin/python training/her2/prepare_bci.py --train 1500
"""
import argparse
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_zip import RemoteZip  # noqa: E402

URL = "https://drive.usercontent.google.com/download?id=1jPPh-rxUaj_bCB58F9mfg9hjY7JjVt3O&export=download&confirm=t"
OUT = Path(__file__).resolve().parents[1] / "data" / "bci"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1500, help="number of training images to keep (all test images kept)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    z = RemoteZip(URL)
    names = [n for n in z.namelist() if n.startswith("BCI_dataset/IHC/") and n.endswith(".png")]
    test = [n for n in names if "/test/" in n]
    train = sorted(n for n in names if "/train/" in n)
    random.Random(0).shuffle(train)
    todo = test + train[:args.train]
    (OUT / "ihc").mkdir(parents=True, exist_ok=True)

    def fetch(n):
        stem = Path(n).stem                       # e.g. 00001_test_2+
        out = OUT / "ihc" / f"{stem}.jpg"
        if not out.exists():
            Image.open(io.BytesIO(z.read(n))).convert("RGB").save(out, quality=92)
        idx, split, label = stem.split("_")
        return dict(file=out.name, split=split, label=label, id=idx)

    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(fetch, todo))
    (OUT / "index.json").write_text(json.dumps(recs, indent=1))
    from collections import Counter
    print(Counter((r["split"], r["label"]) for r in recs))


if __name__ == "__main__":
    main()
