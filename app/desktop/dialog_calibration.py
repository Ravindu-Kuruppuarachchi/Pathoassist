"""Calibration wizard (Feature 0): stage micrometer -> µm/px and field of view per objective, stored in the
microscope profile; plus flat-field blank capture and microscope profile management."""
from __future__ import annotations

import cv2
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QHeaderView, QInputDialog, QLabel, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from .microscope import MicroscopeProfile, um_per_px_from_ticks
from .ui_common import ImageCanvas, error_box

HELP = ("<b>Calibrating an objective</b><ol>"
        "<li>Put the <b>stage micrometer</b> on the stage and focus on its scale with the chosen objective. Use the "
        "camera resolution you will work at (the scale follows resolution changes automatically).</li>"
        "<li>Press <b>Grab frame</b>. Zoom in (mouse wheel) and click <b>two tick marks far apart</b> - the further "
        "apart, the more precise (e.g. 0 and 1000 µm at 4x, 0 and 100 µm at 40x).</li>"
        "<li>Enter the true distance between the two ticks and press <b>Save calibration</b>.</li></ol>"
        "Without a micrometer you can type a known µm/px (marked 'entered manually' - less trustworthy).")


class CalibrationDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle(f"Calibrate objectives - {ctx.profile.name}")
        self.resize(1200, 800)
        self.pts = []
        self.rgb = None
        v = QHBoxLayout(self)
        left = QVBoxLayout()
        self.canvas = ImageCanvas(message="Press 'Grab frame' with the stage micrometer in focus.")
        self.canvas.clicked.connect(self._click)
        left.addWidget(self.canvas, 1)
        h = QHBoxLayout()
        b = QPushButton("Grab frame")
        b.clicked.connect(self.grab)
        h.addWidget(b)
        b = QPushButton("Clear points")
        b.clicked.connect(self._clear)
        h.addWidget(b)
        h.addStretch(1)
        left.addLayout(h)
        v.addLayout(left, 3)

        right = QVBoxLayout()
        lab = QLabel(HELP)
        lab.setWordWrap(True)
        right.addWidget(lab)
        f = QFormLayout()
        self.obj = QComboBox()
        self.obj.setEditable(True)
        self.obj.addItems(ctx.profile.objective_names())
        self.obj.setCurrentText(ctx.objective)
        f.addRow("Objective", self.obj)
        self.dist = QDoubleSpinBox()
        self.dist.setRange(1, 20000)
        self.dist.setDecimals(1)
        self.dist.setValue(100)
        self.dist.setSuffix(" µm")
        self.dist.valueChanged.connect(self._update)
        f.addRow("Distance between ticks", self.dist)
        self.result = QLabel("-")
        self.result.setWordWrap(True)
        f.addRow("Result", self.result)
        right.addLayout(f)
        b = QPushButton("Save calibration")
        b.setStyleSheet("font-weight:bold")
        b.clicked.connect(self.save_ticks)
        right.addWidget(b)
        b = QPushButton("Enter µm/px manually…")
        b.clicked.connect(self.save_manual)
        right.addWidget(b)
        right.addWidget(QLabel("<b>Calibration of this microscope</b>"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Objective", "µm/px (now)", "Field of view", "Method", "Date"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        right.addWidget(self.table, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        right.addWidget(bb)
        v.addLayout(right, 2)
        self._fill_table()
        if ctx.last_frame is not None:
            self.grab()

    def grab(self):
        if self.ctx.last_frame is None:
            error_box(self, "No frame", "Start the camera first.")
            return
        self.rgb = self.ctx.last_frame.copy()
        self._clear()

    def _clear(self):
        self.pts = []
        if self.rgb is not None:
            self.canvas.set_image(self.rgb, fit=True)
        self._update()

    def _click(self, x, y, mods):
        if self.rgb is None:
            return
        if len(self.pts) == 2:
            self.pts = []
        self.pts.append((x, y))
        img = self.rgb.copy()
        t = max(2, self.rgb.shape[1] // 600)
        for p in self.pts:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), (255, 0, 0), cv2.MARKER_CROSS, 40 * t, t)
        if len(self.pts) == 2:
            cv2.line(img, tuple(map(int, self.pts[0])), tuple(map(int, self.pts[1])), (255, 255, 0), t)
        self.canvas.set_image(img)
        self._update()

    def _value(self):
        if len(self.pts) != 2:
            return None
        return um_per_px_from_ticks(self.pts[0], self.pts[1], self.dist.value())

    def _update(self):
        try:
            um = self._value()
        except ValueError as e:
            self.result.setText(f"<span style='color:#b00'>{e}</span>")
            return
        if um is None or self.rgb is None:
            self.result.setText(f"Click two tick marks ({len(self.pts)}/2).")
            return
        h, w = self.rgb.shape[:2]
        fov = MicroscopeProfile.field_of_view(um, w, h)
        self.result.setText(f"<b>{um:.4f} µm/px</b> at {w}×{h}<br>field {fov['width_um']:.0f} × {fov['height_um']:.0f} µm, "
                            f"{fov['area_mm2']:.3f} mm² (equivalent diameter {fov['equiv_diameter_mm']:.3f} mm)")

    def _store(self, um, method, width):
        name = self.obj.currentText().strip()
        if not name:
            error_box(self, "Objective", "Name the objective (e.g. 40x).")
            return
        self.ctx.profile.set_calibration(name, um, width, method)
        self.ctx.profile.save()
        self.ctx.set_objective(name)
        self.ctx.profile_changed.emit()
        if self.obj.findText(name) < 0:
            self.obj.addItem(name)
        self._fill_table()
        QMessageBox.information(self, "Saved", f"{name}: {um:.4f} µm/px saved ({method}).")

    def save_ticks(self):
        try:
            um = self._value()
        except ValueError as e:
            error_box(self, "Calibration", e)
            return
        if um is None:
            error_box(self, "Calibration", "Click two tick marks first.")
            return
        self._store(um, "stage micrometer", self.rgb.shape[1])

    def save_manual(self):
        w = self.rgb.shape[1] if self.rgb is not None else (self.ctx.last_frame.shape[1] if self.ctx.last_frame is not None else None)
        if w is None:
            w, ok = QInputDialog.getInt(self, "Frame width", "Camera frame width (px) this value applies to:", 1920, 100, 20000)
            if not ok:
                return
        um, ok = QInputDialog.getDouble(self, "µm per pixel", f"µm per pixel at {w} px frame width:", 0.25, 0.01, 20, 4)
        if ok:
            self._store(um, "entered manually", w)

    def _fill_table(self):
        p = self.ctx.profile
        names = p.objective_names()
        self.table.setRowCount(len(names))
        w = self.ctx.last_frame.shape[1] if self.ctx.last_frame is not None else None
        h = self.ctx.last_frame.shape[0] if self.ctx.last_frame is not None else None
        for i, n in enumerate(names):
            o = p.objectives[n]
            um = p.um_per_px(n, w) if w else o.um_per_px
            fov = MicroscopeProfile.field_of_view(um, w, h) if (um and w) else None
            vals = [n, f"{um:.4f}" if um else "not calibrated",
                    f"{fov['width_um']:.0f}×{fov['height_um']:.0f} µm" if fov else "-", o.method or "-", o.calibrated_at[:10]]
            for j, t in enumerate(vals):
                it = QTableWidgetItem(t)
                if not um:
                    it.setForeground(Qt.red)
                self.table.setItem(i, j, it)
