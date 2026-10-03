"""Stream the MITOS-ATYPIA-14 training archives (pCloud, 0.6-1.6 GB each) and keep only what
Feature 7 needs: x40 frames (JPEG) + nuclear-atypia scores. Nothing large is stored.

MITOS-ATYPIA-14 (Roux et al., ICPR 2014 contest): breast cancer biopsies, 11 training slides
scanned twice (Aperio A*, 0.2455 µm/px; Hamamatsu H*, 0.2273 µm/px). Each x20 frame (inside
tumour) has a nuclear atypia score 1/2/3 from 2-3 senior pathologists ("decision" = consensus);
each x20 frame is split into 4 x40 frames (suffix a-d), some with 6 atypia criteria scored 1-3
by 3 pathologists.

    ~/venvs/pathoassist-torch/bin/python training/pleomorphism/prepare_mitos_atypia.py --workers 4
"""
import argparse
import io
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "data" / "mitos_atypia"
FOLDERS = {"aperio": ("kZet41ZEtMfzpQIohFSxpadymInMSdENrcy", 0.2455),
           "hamamatsu": ("kZxjh1ZX32mlJHd9l4XH2hmoGY1thHoeaNk", 0.2273)}


def files(code):
    m = requests.get(f"https://eapi.pcloud.com/showpublink?code={code}", timeout=60).json()["metadata"]
    return [(c["name"], c["fileid"]) for c in m["contents"] if c["name"].endswith(".tar.gz")]


class Counting:
    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def read(self, k=-1):
        b = self.raw.read(k)
        self.n += len(b)
        return b


def one(job):
    code, scanner, um, name, fileid = job
    slide = name.split(".")[0]
    done = OUT / "done" / f"{slide}.json"
    if done.exists():
        return json.loads(done.read_text())
    for attempt in range(4):
        try:
            d = requests.get(f"https://eapi.pcloud.com/getpublinkdownload?code={code}&fileid={fileid}", timeout=60).json()
            r = requests.get("https://" + d["hosts"][0] + d["path"], stream=True, timeout=120)
            stream = Counting(r.raw)
            x20, x40, frames = {}, {}, []
            with tarfile.open(fileobj=stream, mode="r|gz") as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    n = m.name
                    if n.endswith("_cna_score_decision.csv"):
                        x20.setdefault(Path(n).name.replace("_cna_score_decision.csv", ""), {})["decision"] = \
                            int(tf.extractfile(m).read().decode().strip() or 0)
                    elif n.endswith("_cna_score_all.csv"):
                        x20.setdefault(Path(n).name.replace("_cna_score_all.csv", ""), {})["raters"] = \
                            [int(v) for v in tf.extractfile(m).read().decode().strip().split(",") if v]
                    elif n.endswith("_cna_criteria.csv"):
                        crit = {}
                        for line in tf.extractfile(m).read().decode().strip().splitlines():
                            p = line.split(",")
                            if len(p) > 1:
                                crit[p[0]] = [int(v) for v in p[1:] if v]
                        if crit:
                            x40[Path(n).name.replace("_cna_criteria.csv", "")] = crit
                    elif "/frames/x40/" in n and n.lower().endswith((".tiff", ".tif", ".png", ".bmp")):
                        im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
                        fn = Path(n).stem + ".jpg"
                        (OUT / "x40").mkdir(parents=True, exist_ok=True)
                        im.save(OUT / "x40" / fn, quality=88)
                        frames.append(fn)
            for k, v in x20.items():          # decision may be missing when raters agree in older files
                if "decision" not in v and v.get("raters"):
                    v["decision"] = int(np.bincount(v["raters"]).argmax())
            rec = dict(slide=slide, scanner=scanner, um_per_px=um, x20=x20, x40_criteria=x40, x40_frames=sorted(frames))
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(json.dumps(rec))
            print(f"{slide}: {len(frames)} x40 frames, {len(x20)} scored x20 frames, {len(x40)} x40 criteria, "
                  f"{stream.n / 2 ** 20:.0f} MB streamed", flush=True)
            return rec
        except Exception as e:  # noqa: BLE001 - network hiccup: retry the whole archive
            print(f"{slide}: attempt {attempt} failed ({e})", flush=True)
    raise RuntimeError(slide)


def backfill_raters(job):
    """Re-read only the atypia score CSVs (they sit at the start of each archive) and add the
    per-pathologist scores to an existing record."""
    code, scanner, um, name, fileid = job
    slide = name.split(".")[0]
    done = OUT / "done" / f"{slide}.json"
    rec = json.loads(done.read_text())
    d = requests.get(f"https://eapi.pcloud.com/getpublinkdownload?code={code}&fileid={fileid}", timeout=60).json()
    r = requests.get("https://" + d["hosts"][0] + d["path"], stream=True, timeout=120)
    seen_atypia = False
    with tarfile.open(fileobj=Counting(r.raw), mode="r|gz") as tf:
        for m in tf:
            if "/atypia/" in m.name:
                seen_atypia = True
            elif seen_atypia:
                break
            if m.isfile() and m.name.endswith("_cna_score_all.csv"):
                k = Path(m.name).name.replace("_cna_score_all.csv", "")
                rec["x20"].setdefault(k, {})["raters"] = [int(v) for v in tf.extractfile(m).read().decode().strip().split(",") if v]
    r.close()
    done.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backfill-raters", action="store_true")
    args = ap.parse_args()
    if args.backfill_raters:
        jobs = [(code, sc, um, n, fid) for sc, (code, um) in FOLDERS.items() for n, fid in files(code)]
        with ThreadPoolExecutor(args.workers) as ex:
            recs = list(ex.map(backfill_raters, jobs))
        (OUT / "index.json").write_text(json.dumps(recs, indent=1))
        print("raters per x20 frame:", np.bincount([len(v.get("raters", [])) for r in recs for v in r["x20"].values()]).tolist())
        return
    jobs = [(code, sc, um, n, fid) for sc, (code, um) in FOLDERS.items() for n, fid in files(code)]
    with ThreadPoolExecutor(args.workers) as ex:
        recs = list(ex.map(one, jobs))
    (OUT / "index.json").write_text(json.dumps(recs, indent=1))
    dist = np.bincount([v["decision"] for r in recs for v in r["x20"].values()], minlength=4)[1:]
    print(f"{len(recs)} archives; x20 atypia scores 1/2/3: {dist.tolist()}")


if __name__ == "__main__":
    main()
