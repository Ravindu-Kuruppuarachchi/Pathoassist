"""Stream MIDOG++ human-breast images and keep only what training needs (disk is tight).

MIDOG++ (Aubreville et al. 2023): 150 labelled human breast cancer regions (~2 mm², 40x,
~0.25 µm/px, 4 scanners). Labels are points (50x50 px boxes) of mitotic figures (cat 1)
and "imposters"/mimics (cat 2, hard negatives).

For each image: download (~130 MB tiff) -> extract -> delete the tiff.
  train split : 512x512 JPEG crops around every mitosis and mimic (jittered) + random
                tissue crops, with all points inside each crop -> crops/ + crops.json
  val/test    : the whole image as JPEG (q92) + its points -> full/ + full.json
Val = 15 train-split slides held out (stratified by scanner) for threshold tuning.

    python training/mitosis/prepare_midog.py --workers 3
"""
import argparse
import csv
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

D = Path(__file__).resolve().parents[1] / "data" / "midog"
CROP = 512
N_RANDOM = 20


def load_meta():
    coco = json.loads((D / "MIDOG++.json").read_text())
    ids = json.loads((D / "file_ids.json").read_text())
    rows = {int(r["Slide"]): r for r in csv.DictReader(open(D / "datasets_xvalidation.csv"), delimiter=";")}
    imgs = {i["id"]: i for i in coco["images"] if i["tumor_type"] == "human breast cancer" and i["id"] in rows}
    pts = {k: [] for k in imgs}
    for a in coco["annotations"]:
        if a["image_id"] in imgs:
            x0, y0, x1, y1 = a["bbox"]
            pts[a["image_id"]].append(((x0 + x1) / 2, (y0 + y1) / 2, a["category_id"]))
    # val: 15 train slides, spread over scanners
    rng = random.Random(0)
    by_scanner = {}
    for k in imgs:
        if rows[k]["Dataset"] == "train":
            by_scanner.setdefault(rows[k]["Scanner"], []).append(k)
    val = set()
    for s, ks in sorted(by_scanner.items()):
        rng.shuffle(ks)
        val.update(ks[:round(15 * len(ks) / sum(map(len, by_scanner.values())))])
    split = {k: ("val" if k in val else rows[k]["Dataset"]) for k in imgs}
    return imgs, pts, split, ids, rows


def tissue_mask(img):
    small = cv2.resize(img, None, fx=1 / 16, fy=1 / 16, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] > 25) & (hsv[..., 2] < 235)


def process(k, img_meta, points, split, file_id, scanner):
    name = img_meta["file_name"]
    done = D / "done" / f"{k}.json"
    if done.exists():
        return json.loads(done.read_text())
    tmp = D / "tmp" / name
    tmp.parent.mkdir(exist_ok=True)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{file_id}", tmp)
            img = tifffile.imread(tmp)
            break
        except Exception as e:  # truncated download -> retry
            print(f"{name}: retry {attempt} ({e})", flush=True)
    else:
        raise RuntimeError(f"failed {name}")
    tmp.unlink()
    img = np.ascontiguousarray(img[..., :3])
    H, W = img.shape[:2]
    rec = []
    if split == "train":
        rng = np.random.default_rng(k)
        centers = [(x, y) for x, y, _ in points]
        tm = tissue_mask(img)
        ty, tx = np.nonzero(tm)
        for _ in range(N_RANDOM if len(ty) else 0):
            i = rng.integers(len(ty))
            centers.append((tx[i] * 16 + 8, ty[i] * 16 + 8))
        P = np.array([(x, y) for x, y, _ in points]).reshape(-1, 2)
        C = np.array([c for _, _, c in points])
        (D / "crops").mkdir(exist_ok=True)
        for j, (cx, cy) in enumerate(centers):
            x0 = int(np.clip(cx + rng.integers(-180, 181) - CROP // 2, 0, W - CROP))
            y0 = int(np.clip(cy + rng.integers(-180, 181) - CROP // 2, 0, H - CROP))
            inside = (P[:, 0] >= x0) & (P[:, 0] < x0 + CROP) & (P[:, 1] >= y0) & (P[:, 1] < y0 + CROP) if len(P) else []
            fn = f"{k:03d}_{j:03d}.jpg"
            cv2.imwrite(str(D / "crops" / fn), cv2.cvtColor(img[y0:y0 + CROP, x0:x0 + CROP], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            rec.append(dict(file=fn, slide=k, scanner=scanner,
                            points=[[float(P[i, 0] - x0), float(P[i, 1] - y0), int(C[i])] for i in np.nonzero(inside)[0]]))
    else:
        (D / "full").mkdir(exist_ok=True)
        fn = f"{k:03d}.jpg"
        cv2.imwrite(str(D / "full" / fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rec.append(dict(file=fn, slide=k, scanner=scanner, split=split, width=W, height=H,
                        points=[[x, y, c] for x, y, c in points]))
    done.parent.mkdir(exist_ok=True)
    done.write_text(json.dumps(rec))
    print(f"{name} [{split}] {W}x{H}: {sum(c == 1 for *_, c in points)} mitoses, {len(rec)} records", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    imgs, pts, split, ids, rows = load_meta()
    jobs = sorted(imgs)
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda k: (k, process(k, imgs[k], pts[k], split[k], ids[imgs[k]["file_name"]],
                                                     rows[k]["Scanner"])), jobs))
    crops = [r for k, rec in results if split[k] == "train" for r in rec]
    full = [r for k, rec in results if split[k] != "train" for r in rec]
    (D / "crops.json").write_text(json.dumps(crops))
    (D / "full.json").write_text(json.dumps(full))
    print(f"train crops {len(crops)}, full images {len(full)} "
          f"(val {sum(r['split'] == 'val' for r in full)}, test {sum(r['split'] == 'test' for r in full)})")


if __name__ == "__main__":
    main()
