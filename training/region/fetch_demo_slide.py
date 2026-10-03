"""Save a held-out BACH slide (A10) at level 1 (~1.87 µm/px) as JPEG for the Feature 5 demo / sweep test."""
import sys
from pathlib import Path

import cv2
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_zip import RemoteZip  # noqa: E402
from region.prepare_bach import OUT, ROOT, URL  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "A10"
tmp = OUT / f"tmp_demo_{name}.svs"
RemoteZip(URL).extract_to(ROOT + f"WSI/{name}.svs", tmp)
with tifffile.TiffFile(tmp) as t:
    lv = t.series[0].levels[1].asarray()[..., :3]
cv2.imwrite(str(OUT / f"demo_{name}_level1.jpg"), cv2.cvtColor(lv, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
tmp.unlink()
print(name, lv.shape, "saved")
