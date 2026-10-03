"""USB (UVC) camera access through OpenCV. Qt-free so it can be tested headless.

Backends: V4L2 on Linux, DirectShow on Windows (MSMF opens slowly and ignores many
controls). Vendor-SDK cameras (ToupTek, Basler, ...) later get their own `Camera` subclass.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraInfo:
    index: int
    name: str

    def __str__(self):
        return f"{self.name} (#{self.index})"


def _backend() -> int:
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def list_cameras(max_index: int = 10) -> list[CameraInfo]:
    """Cameras that actually deliver frames. On Linux, UVC cameras also expose metadata-only
    /dev/videoN nodes; those fail to read and are skipped."""
    if sys.platform.startswith("linux"):
        nodes = sorted(Path("/dev").glob("video[0-9]*"), key=lambda p: int(p.name[5:]))
        candidates = [int(p.name[5:]) for p in nodes]
    else:
        candidates = list(range(max_index))
    found = []
    for i in candidates:
        cap = cv2.VideoCapture(i, _backend())
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(CameraInfo(i, _device_name(i)))
        finally:
            cap.release()
    return found


def _device_name(index: int) -> str:
    p = Path(f"/sys/class/video4linux/video{index}/name")
    try:
        return p.read_text().strip()
    except OSError:
        return "USB camera"


class Camera:
    """Interface every camera backend implements."""

    def open(self, width: int, height: int) -> tuple[int, int]: ...
    def read(self) -> np.ndarray | None: ...      # RGB uint8 (H, W, 3) or None
    def set_auto(self, auto: bool) -> None: ...   # auto exposure + white balance on/off
    def close(self) -> None: ...


class OpenCVCamera(Camera):
    def __init__(self, index: int):
        self.index = index
        self._cap: cv2.VideoCapture | None = None

    def open(self, width: int, height: int) -> tuple[int, int]:
        """Returns the resolution the camera actually granted."""
        self.close()
        cap = cv2.VideoCapture(self.index, _backend())
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera #{self.index}")
        # MJPG lets USB2 cameras deliver high resolutions at usable frame rates (YUYV is uncompressed).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        return int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, bgr = self._cap.read()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None

    def set_auto(self, auto: bool) -> None:
        """Locking exposure and white balance keeps stain colours comparable between fields.
        Best effort: some cameras ignore these controls."""
        if self._cap is None:
            return
        if sys.platform == "win32":
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto else 0.25)
        else:  # V4L2: 3 = aperture priority (auto), 1 = manual
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto else 1)
        self._cap.set(cv2.CAP_PROP_AUTO_WB, 1 if auto else 0)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
