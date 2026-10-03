"""Shared Qt pieces: image canvas (zoom / pan / click), background job runner, camera thread, small helpers."""
from __future__ import annotations

import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, QPointF, QPropertyAnimation, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QFrame, QGraphicsOpacityEffect, QGraphicsPixmapItem, QGraphicsScene,
                               QGraphicsSimpleTextItem, QGraphicsView, QLabel, QMessageBox, QSizePolicy)

from .camera import CameraInfo, OpenCVCamera
from .microscope import FlatField, GateState, QualityGate


def to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb[..., :3])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def to_pixmap(rgb: np.ndarray) -> QPixmap:
    return QPixmap.fromImage(to_qimage(rgb))


def error_box(parent, title, msg):
    QMessageBox.warning(parent, title, str(msg))


def hline():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setFrameShadow(QFrame.Sunken)
    return f


class Pill(QLabel):
    """Coloured status pill (e.g. READY / MOVING / BLUR)."""

    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(110)
        self.set_state(text, "#666")

    def set_state(self, text, color):
        self.setText(text)
        self.setStyleSheet(f"QLabel {{ background:{color}; color:white; font-weight:bold; border-radius:9px;"
                           " padding:3px 10px; }")


# ----------------------------------------------------------------------------------------------- canvas
class ImageCanvas(QGraphicsView):
    """Shows an RGB image; wheel zoom, middle-drag / Space+drag pan, left click -> clicked(x, y, mods)
    in image pixel coordinates. `F` fits the image to the window."""
    clicked = Signal(float, float, object)       # x, y, set of modifiers {"shift", "ctrl", "alt"}
    right_clicked = Signal(float, float)

    def __init__(self, parent=None, message=""):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#1F2228")))
        self.setFrameShape(QFrame.NoFrame)
        self.hud = dict(chip=None, chip_color="#666", note=None, um_per_px=None)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.StrongFocus)
        self._pix = QGraphicsPixmapItem()
        self._pix.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._pix)
        self._msg = QGraphicsSimpleTextItem()
        self._msg.setBrush(QBrush(QColor("#9AA3B2")))
        f = QFont()
        f.setPointSize(13)
        self._msg.setFont(f)
        self.scene().addItem(self._msg)
        self._size = None
        self._fit_mode = True
        self._press = None
        self._pan_last = None
        self._space = False
        self.set_message(message)

    @property
    def has_image(self):
        return self._size is not None

    def set_message(self, text):
        self._msg.setText(text)
        self._msg.setVisible(bool(text) and self._size is None)
        if self._size is None:
            r = self._msg.boundingRect()
            self.scene().setSceneRect(QRectF(-r.width() / 2 - 20, -r.height() / 2 - 20, r.width() + 40, r.height() + 40))
            self._msg.setPos(-r.width() / 2, -r.height() / 2)
            self.resetTransform()

    def clear(self, message=""):
        self._pix.setPixmap(QPixmap())
        self._size = None
        self.set_message(message)

    def set_image(self, rgb: np.ndarray, fit: bool = False):
        h, w = rgb.shape[:2]
        self._pix.setPixmap(to_pixmap(rgb))
        new = self._size != (w, h)
        self._size = (w, h)
        self._msg.setVisible(False)
        if new:
            self.scene().setSceneRect(QRectF(0, 0, w, h))
        if new or fit or self._fit_mode:
            self.fit()

    def fit(self):
        if self._size:
            self.fitInView(QRectF(0, 0, *self._size), Qt.KeepAspectRatio)
            self._fit_mode = True

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit_mode:
            self.fit()

    def wheelEvent(self, e):
        if not self._size:
            return
        f = 1.25 if e.angleDelta().y() > 0 else 0.8
        s = self.transform().m11() * f
        if 0.02 < s < 40:
            self.scale(f, f)
            self._fit_mode = False
            self.viewport().update()

    def _mods(self, e):
        m = set()
        if e.modifiers() & Qt.ShiftModifier:
            m.add("shift")
        if e.modifiers() & Qt.ControlModifier:
            m.add("ctrl")
        if e.modifiers() & Qt.AltModifier:
            m.add("alt")
        return m

    def _start_pan(self, e):
        self._pan_last = e.position()
        self.setCursor(Qt.ClosedHandCursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._space):
            self._start_pan(e)
            return
        if e.button() == Qt.LeftButton:
            self._press = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._pan_last is not None:
            d = e.position() - self._pan_last
            self._pan_last = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            self._fit_mode = False
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._pan_last is not None:
            self._pan_last = None
            self.unsetCursor()
            return
        if e.button() == Qt.LeftButton and self._press is not None and self._size:
            if (e.position() - self._press).manhattanLength() < 5:
                p = self.mapToScene(e.position().toPoint())
                if 0 <= p.x() < self._size[0] and 0 <= p.y() < self._size[1]:
                    self.clicked.emit(p.x(), p.y(), self._mods(e))
            self._press = None
        elif e.button() == Qt.RightButton and self._size:
            p = self.mapToScene(e.position().toPoint())
            self.right_clicked.emit(p.x(), p.y())
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = True
            self.setCursor(Qt.OpenHandCursor)
        elif e.key() == Qt.Key_F:
            self.fit()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key_Space and not e.isAutoRepeat():
            self._space = False
            self.unsetCursor()
        else:
            super().keyReleaseEvent(e)

    # -------------------------------------------------------------- heads-up overlay
    def set_hud(self, **kw):
        """chip / chip_color: status pill top-left; note: grey pill top-right; um_per_px: draws a scale bar."""
        if any(self.hud.get(k) != v for k, v in kw.items()):
            self.hud.update(kw)
            self.viewport().update()

    def drawForeground(self, p, rect):
        if not self._size:
            return
        p.save()
        p.resetTransform()
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        fm = p.fontMetrics()
        vw, vh = self.viewport().width(), self.viewport().height()

        def pill(x, y, text, bg, fg="#FFFFFF", right=False):
            w = fm.horizontalAdvance(text) + 22
            x = x - w if right else x
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(bg))
            p.drawRoundedRect(QRectF(x, y, w, 24), 12, 12)
            p.setPen(QColor(fg))
            p.drawText(QRectF(x, y, w, 24), Qt.AlignCenter, text)

        if self.hud.get("chip"):
            pill(12, 12, self.hud["chip"], self.hud.get("chip_color") or "#555")
        if self.hud.get("note"):
            pill(vw - 12, 12, self.hud["note"], QColor(20, 22, 27, 200), "#E6E9EF", right=True)
        um = self.hud.get("um_per_px")
        s = self.transform().m11()
        if um and s:
            um_per_screen = um / s
            nice = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
            L = max([n for n in nice if n / um_per_screen <= 170] or [nice[0]])
            px = L / um_per_screen
            text = f"{L} µm" if L < 1000 else f"{L / 1000:g} mm"
            bw = max(px, fm.horizontalAdvance(text)) + 24
            x0, y0 = 12, vh - 46
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(20, 22, 27, 190))
            p.drawRoundedRect(QRectF(x0, y0, bw, 34), 8, 8)
            p.setBrush(QColor("#FFFFFF"))
            p.drawRect(QRectF(x0 + 12, y0 + 24, px, 4))
            p.setPen(QColor("#FFFFFF"))
            p.drawText(QRectF(x0 + 12, y0 + 3, bw, 18), Qt.AlignLeft | Qt.AlignVCenter, text)
        p.restore()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.viewport().update()          # keep the overlay pinned while panning

    def screen_px(self) -> float:
        """Image pixels per screen pixel (for click tolerances)."""
        s = self.transform().m11()
        return 1.0 / s if s else 1.0


# ----------------------------------------------------------------------------------------------- jobs
class _Runner(QObject):
    done = Signal(int, object)
    failed = Signal(int, str, str)

    @Slot(int, object)
    def run(self, job_id, fn):
        try:
            self.done.emit(job_id, fn())
        except Exception as e:  # noqa: BLE001 - every failure is shown to the user
            self.failed.emit(job_id, str(e) or type(e).__name__, traceback.format_exc())


class JobRunner(QObject):
    """One background thread for all model work (models are not shared across threads). Jobs run in order;
    on_done / on_error are called on the GUI thread."""
    _submit = Signal(int, object)
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = QThread(self)
        self._runner = _Runner()
        self._runner.moveToThread(self._thread)
        self._submit.connect(self._runner.run)
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)
        self._thread.start()
        self._next = 0
        self._cb = {}
        self.log_error = None

    @property
    def pending(self):
        return len(self._cb)

    def submit(self, fn, on_done=None, on_error=None):
        self._next += 1
        self._cb[self._next] = (on_done, on_error)
        if len(self._cb) == 1:
            self.busy_changed.emit(True)
        self._submit.emit(self._next, fn)
        return self._next

    def _pop(self, jid):
        cb = self._cb.pop(jid, (None, None))
        if not self._cb:
            self.busy_changed.emit(False)
        return cb

    @Slot(int, object)
    def _on_done(self, jid, res):
        ok, _ = self._pop(jid)
        if ok:
            ok(res)

    @Slot(int, str, str)
    def _on_failed(self, jid, msg, tb):
        _, err = self._pop(jid)
        if self.log_error:
            self.log_error(tb)
        if err:
            err(msg)

    def stop(self):
        self._thread.quit()
        self._thread.wait(10000)


# ----------------------------------------------------------------------------------------------- camera
class CaptureThread(QThread):
    """Camera loop in its own thread: read -> flat-field -> motion/focus gate -> emit.
    frame(rgb, GateState) for every frame; the GateState.dwell flag marks the frame to analyse."""
    frame = Signal(object, object)
    failed = Signal(str)
    opened = Signal(int, int)

    def __init__(self, info: CameraInfo, width: int, height: int, auto: bool, flat: FlatField):
        super().__init__()
        self.info, self.size = info, (width, height)
        self._auto, self._auto_dirty = auto, True
        self._running = True
        self.flat = flat
        self.flat_on = True
        self.gate = QualityGate()
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
                if self.flat_on:
                    rgb = self.flat.apply(rgb)
                state = self.gate.update(rgb)
                now = time.perf_counter()
                self.fps = 0.9 * self.fps + 0.1 / max(now - t_last, 1e-3)
                t_last = now
                self.frame.emit(rgb, state)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))
        finally:
            cam.close()


class Toast(QLabel):
    """Short confirmation at the bottom of the window that fades out (answers the user's action)."""
    TONES = {"good": "#237A57", "warn": "#A35F14", "bad": "#B42318", "neutral": "#1B2230"}

    def __init__(self, parent):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.hide()
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._anim = QPropertyAnimation(self._fx, b"opacity", self)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._fade)

    def show_text(self, text, tone="neutral", ms=3500):
        self.setStyleSheet(f"QLabel {{ background:{self.TONES.get(tone, '#1B2230')}; color:white; border-radius:10px;"
                           " padding:10px 18px; font-size:13px; font-weight:500; }")
        self.setText(text)
        par = self.parentWidget()
        self.setMaximumWidth(min(640, par.width() - 40))
        self.adjustSize()
        self.move((par.width() - self.width()) // 2, par.height() - self.height() - 48)
        self._anim.stop()
        self._fx.setOpacity(1.0)
        self.show()
        self.raise_()
        self._timer.start(ms)

    def _fade(self):
        self._anim.setDuration(400)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self.hide)
        self._anim.start()


GATE = {"ready": ("Ready", "#237A57"), "moving": ("Moving", "#A35F14"), "blur": ("Not sharp", "#B42318"),
        "off": ("No camera", "#5A6475")}


def gate_hud(st: GateState | None, live: bool):
    if not live or st is None:
        return GATE["off"]
    if st.ready:
        return GATE["ready"]
    return GATE["moving"] if not st.still else GATE["blur"]


def gate_pill(pill: Pill, st: GateState | None, live: bool):
    if not live or st is None:
        pill.set_state("NO CAMERA", "#666")
    elif st.ready:
        pill.set_state("READY", "#1b8a3a")
    elif not st.still:
        pill.set_state("MOVING", "#b36b00")
    else:
        pill.set_state("NOT SHARP", "#a33")
