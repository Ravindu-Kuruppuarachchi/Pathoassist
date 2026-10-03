"""Main window: camera picker, live view, freeze/capture, nuclei detection on the shown frame."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QLabel, QMainWindow,
                               QMessageBox, QPushButton, QSizePolicy, QToolBar)

from . import __version__
from .camera import CameraInfo, OpenCVCamera, list_cameras
from .paths import DEFAULT_NUCLEI_MODEL, captures_dir, models_dir

RESOLUTIONS = [(640, 480), (1280, 720), (1280, 960), (1920, 1080), (2592, 1944), (3840, 2160)]


class CaptureThread(QThread):
    frame = Signal(object)        # RGB ndarray
    failed = Signal(str)
    opened = Signal(int, int)     # granted resolution

    def __init__(self, info: CameraInfo, width: int, height: int, auto: bool):
        super().__init__()
        self.info, self.size = info, (width, height)
        self._auto = auto
        self._auto_dirty = True
        self._running = True
        self.fps = 0.0

    def set_auto(self, auto: bool):
        self._auto, self._auto_dirty = auto, True

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        cam = OpenCVCamera(self.info.index)
        try:
            self.opened.emit(*cam.open(*self.size))
            misses, t_last = 0, time.perf_counter()
            while self._running:
                if self._auto_dirty:
                    cam.set_auto(self._auto)
                    self._auto_dirty = False
                rgb = cam.read()
                if rgb is None:
                    misses += 1
                    if misses > 50:
                        self.failed.emit("Camera stopped sending frames (unplugged?)")
                        return
                    self.msleep(20)
                    continue
                misses = 0
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_last, 1e-3)
                t_last = now
                self.frame.emit(rgb)
        except Exception as e:  # noqa: BLE001 - surface any camera error in the UI
            self.failed.emit(str(e))
        finally:
            cam.close()


class DetectWorker(QObject):
    """Runs on its own thread; the detector is loaded once on first use."""
    done = Signal(object, int, float)   # overlay RGB, n nuclei, seconds
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._det = None

    @Slot(object, float)
    def run(self, rgb: np.ndarray, um_per_px: float):
        try:
            from core.nuclei import NucleiDetector, draw_overlay
            if self._det is None:
                self._det = NucleiDetector(models_dir() / DEFAULT_NUCLEI_MODEL)
            d = self._det.detect(rgb, um_per_px=um_per_px, return_labels=False)
            self.done.emit(draw_overlay(rgb, d, thickness=2), len(d), d.timings["total"])
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


class MainWindow(QMainWindow):
    request_detect = Signal(object, float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"PathoAssist {__version__} — research use only")
        self.resize(1280, 860)
        self._capture: CaptureThread | None = None
        self._raw: np.ndarray | None = None      # frame being shown (never the overlay)
        self._shown: np.ndarray | None = None    # what is drawn (raw or overlay)
        self._frozen = False
        self._source = "camera"

        self.view = QLabel("No camera running. Choose a camera and press Start,\nor File → Open image.")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.view.setStyleSheet("background:#111; color:#aaa; font-size:15px;")
        self.setCentralWidget(self.view)
        self._build_toolbar()
        self._build_menu()

        self._det_thread = QThread(self)
        self._worker = DetectWorker()
        self._worker.moveToThread(self._det_thread)
        self.request_detect.connect(self._worker.run)
        self._worker.done.connect(self._on_detected)
        self._worker.failed.connect(lambda m: self._error("Detection failed", m))
        self._det_thread.start()

        self.refresh_cameras()

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self):
        tb = QToolBar("Camera")
        tb.setMovable(False)
        self.addToolBar(tb)
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(220)
        tb.addWidget(QLabel(" Camera "))
        tb.addWidget(self.cam_combo)
        btn = QPushButton("↻")
        btn.setToolTip("Rescan cameras")
        btn.clicked.connect(self.refresh_cameras)
        tb.addWidget(btn)
        self.res_combo = QComboBox()
        for w, h in RESOLUTIONS:
            self.res_combo.addItem(f"{w}×{h}", (w, h))
        self.res_combo.setCurrentIndex(3)
        tb.addWidget(self.res_combo)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.toggle_camera)
        tb.addWidget(self.start_btn)
        self.lock_box = QCheckBox("Lock exposure/WB")
        self.lock_box.setToolTip("Turn off auto exposure and auto white balance so colours stay "
                                 "comparable between fields. Set the light first, then lock.")
        self.lock_box.toggled.connect(lambda on: self._capture and self._capture.set_auto(not on))
        tb.addWidget(self.lock_box)
        tb.addSeparator()
        self.freeze_btn = QPushButton("Freeze")
        self.freeze_btn.setCheckable(True)
        self.freeze_btn.toggled.connect(self.set_frozen)
        tb.addWidget(self.freeze_btn)
        cap_btn = QPushButton("Capture")
        cap_btn.clicked.connect(self.capture)
        tb.addWidget(cap_btn)
        tb.addSeparator()
        tb.addWidget(QLabel(" µm/px "))
        self.scale_box = QDoubleSpinBox()
        self.scale_box.setDecimals(3)
        self.scale_box.setRange(0.05, 5.0)
        self.scale_box.setSingleStep(0.01)
        self.scale_box.setValue(0.25)
        self.scale_box.setToolTip("Frame scale from calibration (stage micrometer) for the current objective")
        tb.addWidget(self.scale_box)
        self.detect_btn = QPushButton("Detect nuclei")
        self.detect_btn.clicked.connect(self.detect)
        tb.addWidget(self.detect_btn)

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        act = QAction("Open image…", self)
        act.triggered.connect(self.open_image)
        m.addAction(act)
        act = QAction("Open captures folder", self)
        act.triggered.connect(self.open_captures)
        m.addAction(act)
        m.addSeparator()
        act = QAction("Quit", self)
        act.triggered.connect(self.close)
        m.addAction(act)

    def _status(self, msg: str):
        self.statusBar().showMessage(msg)

    def _error(self, title: str, msg: str):
        self.detect_btn.setEnabled(True)
        QMessageBox.warning(self, title, msg)

    # ------------------------------------------------------------------ camera
    def refresh_cameras(self):
        if self._capture is not None:
            return  # probing would fight the running camera for the device
        self.cam_combo.clear()
        cams = list_cameras()
        for c in cams:
            self.cam_combo.addItem(str(c), c)
        self.start_btn.setEnabled(bool(cams))
        self._status(f"{len(cams)} camera(s) found" if cams else "No camera found — plug in the USB camera and press ↻")

    def toggle_camera(self):
        if self._capture is not None:
            self.stop_camera()
            return
        info = self.cam_combo.currentData()
        if info is None:
            return
        w, h = self.res_combo.currentData()
        self._capture = CaptureThread(info, w, h, auto=not self.lock_box.isChecked())
        self._capture.frame.connect(self._on_frame)
        self._capture.failed.connect(self._on_camera_failed)
        self._capture.opened.connect(lambda gw, gh: self._status(f"{info.name}: {gw}×{gh}"))
        self._capture.start()
        self._source = info.name
        self.freeze_btn.setChecked(False)
        self.start_btn.setText("Stop")
        self.cam_combo.setEnabled(False)
        self.res_combo.setEnabled(False)

    def stop_camera(self):
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self.start_btn.setText("Start")
        self.cam_combo.setEnabled(True)
        self.res_combo.setEnabled(True)

    @Slot(str)
    def _on_camera_failed(self, msg: str):
        self.stop_camera()
        self._error("Camera error", msg)

    @Slot(object)
    def _on_frame(self, rgb: np.ndarray):
        if self._frozen:
            return
        self._raw = self._shown = rgb
        self._paint()
        if self._capture is not None:
            h, w = rgb.shape[:2]
            self._status(f"{self._source}: {w}×{h}  {self._capture.fps:.1f} fps")

    def set_frozen(self, on: bool):
        self._frozen = on
        if not on:
            self._shown = self._raw
            self._paint()

    # ------------------------------------------------------------------ display
    def _paint(self):
        if self._shown is None:
            return
        pm = to_pixmap(self._shown)
        self.view.setPixmap(pm.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._paint()

    # ------------------------------------------------------------------ actions
    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open image", str(Path.home()),
                                              "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not path:
            return
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            self._error("Open image", f"Not a readable image:\n{path}")
            return
        self.freeze_btn.setChecked(True)
        self._raw = self._shown = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._source = Path(path).name
        self._paint()
        self._status(f"{self._source}: {bgr.shape[1]}×{bgr.shape[0]}")

    def capture(self):
        """Saves the raw frame losslessly (PNG) with a JSON sidecar; never the overlay."""
        if self._raw is None:
            return
        day = captures_dir() / datetime.now().strftime("%Y-%m-%d")
        day.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("%H%M%S_%f")
        cv2.imwrite(str(day / f"{stem}.png"), cv2.cvtColor(self._raw, cv2.COLOR_RGB2BGR))
        meta = dict(source=self._source, um_per_px=self.scale_box.value(),
                    exposure_locked=self.lock_box.isChecked(),
                    width=int(self._raw.shape[1]), height=int(self._raw.shape[0]),
                    app_version=__version__, captured_at=datetime.now().isoformat(timespec="seconds"))
        (day / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        self._status(f"Saved {day / stem}.png")

    def detect(self):
        if self._raw is None:
            return
        self.freeze_btn.setChecked(True)
        self.detect_btn.setEnabled(False)
        self._status("Detecting nuclei… (first run loads the model)")
        self.request_detect.emit(self._raw, self.scale_box.value())

    @Slot(object, int, float)
    def _on_detected(self, overlay: np.ndarray, n: int, secs: float):
        self.detect_btn.setEnabled(True)
        if not self._frozen:   # user unfroze while it ran: result is stale
            return
        self._shown = overlay
        self._paint()
        self._status(f"{n} nuclei detected in {secs:.2f} s  (Unfreeze to return to live view)")

    def open_captures(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(captures_dir())))

    def closeEvent(self, e):
        self.stop_camera()
        self._det_thread.quit()
        self._det_thread.wait(5000)
        super().closeEvent(e)
