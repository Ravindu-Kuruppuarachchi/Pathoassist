"""Annotate page: pathologists label new local data for every feature, with task instructions, model pre-fill,
QuPath-compatible export, and threshold calibration from the labels (Qt side of app/desktop/annotation.py)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QKeySequence, QPainterPath, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFileDialog, QFormLayout, QFrame, QGraphicsItem, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPlainTextEdit, QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from .annotation import TASK_BY_KEY, TASKS, AnnotationStore, Item, calibrate, instructions_html, prefill
from .casestore import STAINS, read_meta, read_rgb
from .paths import annotations_dir, calibration_dir
from .theme import C, button, hline, icon, label, panel, section
from .ui_common import ImageCanvas, error_box

GEOMETRY = {"point": "points", "polygon": "outlines", "field": "one score per image"}
DEFAULT_STAIN = {"erpr_intensity": "ER", "ki67_positivity": "Ki67", "her2_membrane": "HER2", "celltype": "ER"}


# ------------------------------------------------------------------------------------------- drawing layer
class AnnotationLayer(QGraphicsItem):
    def __init__(self):
        super().__init__()
        self.doc = None
        self.task = None
        self.draft: list = []
        self.selected = None
        self._rect = QRectF()
        self._colors = {}
        self.setZValue(10)

    def set_doc(self, doc, task):
        self.prepareGeometryChange()
        self.doc, self.task = doc, task
        self._rect = QRectF(0, 0, doc.width, doc.height) if doc else QRectF()
        self._colors = {c.name: QColor(*c.color) for c in task.classes} if task else {}
        self.draft = []
        self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, p, option, widget=None):
        if self.doc is None:
            return
        s = p.worldTransform().m11() or 1.0
        r = 5.0 / s
        for i, it in enumerate(self.doc.items):
            col = self._colors.get(it.cls, QColor(128, 128, 128))
            pen = QPen(col, 2 if i != self.selected else 4)
            pen.setCosmetic(True)
            if it.source == "model":
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            if it.kind == "point":
                x, y = it.pts[0]
                p.setBrush(QBrush(col) if it.source == "annotator" else Qt.NoBrush)
                p.drawEllipse(QPointF(x, y), r, r)
            else:
                p.setBrush(Qt.NoBrush)
                p.drawPolygon(QPolygonF([QPointF(x, y) for x, y in it.pts]))
        if self.draft:
            pen = QPen(QColor(255, 255, 255), 2)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            path = QPainterPath(QPointF(*self.draft[0]))
            for x, y in self.draft[1:]:
                path.lineTo(QPointF(x, y))
            p.drawPath(path)
            for x, y in self.draft:
                p.drawEllipse(QPointF(x, y), 2.5 / s, 2.5 / s)


class AnnotationCanvas(ImageCanvas):
    add_point = Signal(float, float)
    polygon_done = Signal(list)
    classify_at = Signal(float, float)
    delete_at = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent, "Choose Live frame or From file to start labelling.")
        self.layer = AnnotationLayer()
        self.scene().addItem(self.layer)
        self.tool = "point"
        self._stroke = None          # freehand: list of points while the button is held
        self._freehand = False
        self._down = None
        self.clicked.connect(self._on_click)
        self.right_clicked.connect(lambda x, y: self.delete_at.emit(x, y))

    def set_tool(self, tool):
        self.tool = tool
        self.cancel_draft()

    def cancel_draft(self):
        self.layer.draft = []
        self.layer.update()

    def finish_draft(self):
        if len(self.layer.draft) >= 3:
            pts = [list(p) for p in self.layer.draft]
            self.layer.draft = []
            self.polygon_done.emit(pts)
        self.layer.update()

    def _on_click(self, x, y, mods):
        if self.layer.doc is None:
            return
        if self.tool == "point":
            self.add_point.emit(x, y)
        elif self.tool == "classify":
            self.classify_at.emit(x, y)
        elif self.tool == "polygon" and not self._freehand:
            d = self.layer.draft
            tol = 8 * self.screen_px()
            if len(d) >= 3 and (d[0][0] - x) ** 2 + (d[0][1] - y) ** 2 <= tol ** 2:
                self.finish_draft()
                return
            d.append((x, y))
            self.layer.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.tool == "polygon" and not self._space and self.layer.doc is not None:
            p = self.mapToScene(e.position().toPoint())
            self._down = e.position()
            self._stroke = [(p.x(), p.y())]
            self._freehand = False
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._stroke is not None and e.buttons() & Qt.LeftButton:
            if not self._freehand and (e.position() - self._down).manhattanLength() > 6:
                self._freehand = True
                self.layer.draft = list(self._stroke)
            if self._freehand:
                p = self.mapToScene(e.position().toPoint())
                lx, ly = self.layer.draft[-1]
                if (p.x() - lx) ** 2 + (p.y() - ly) ** 2 >= (2 * self.screen_px()) ** 2:
                    self.layer.draft.append((p.x(), p.y()))
                    self.layer.update()
                return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._stroke is not None and e.button() == Qt.LeftButton:
            fh = self._freehand
            self._stroke = None
            if fh:
                self._freehand = False
                self._press = None
                self.finish_draft()
                return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        if self.tool == "polygon":
            self.finish_draft()
            return
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        if self.tool == "polygon" and e.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.finish_draft()
        elif e.key() == Qt.Key_Escape:
            self.cancel_draft()
        elif e.key() == Qt.Key_Backspace and self.layer.draft:
            self.layer.draft.pop()
            self.layer.update()
        else:
            super().keyPressEvent(e)


# ------------------------------------------------------------------------------------------- page
class AnnotatePage(QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.store = AnnotationStore(annotations_dir())
        self.doc = None
        self.rgb = None
        self.undo_stack = []
        self.dirty = False
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(lambda: self.save(quiet=True) if self.dirty else None)
        self._timer.start(30000)
        self._on_task()

    @property
    def task(self):
        it = self.task_list.currentItem()
        return TASK_BY_KEY[it.data(Qt.UserRole) if it is not None else TASKS[0].key]

    def select_task(self, key):
        for i in range(self.task_list.count()):
            if self.task_list.item(i).data(Qt.UserRole) == key:
                self.task_list.setCurrentRow(i)
                return

    def _build(self):
        self.setObjectName("page")
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(12)

        # ---------------- left: task + images
        left, lv = panel(margins=(12, 14, 12, 12), spacing=8)
        left.setFixedWidth(270)
        lv.addWidget(section("What to label", "Each task feeds one model or threshold."))
        self.task_list = QListWidget()
        self.task_list.setStyleSheet("QListWidget { border: none; }")
        for t in TASKS:
            it = QListWidgetItem(f"{t.title}\n{GEOMETRY[t.geometry]}, {t.objective}")
            it.setData(Qt.UserRole, t.key)
            it.setToolTip(t.prefill or "")
            self.task_list.addItem(it)
        self.task_list.setFixedHeight(258)
        self.task_list.currentRowChanged.connect(lambda _: self._on_task())
        lv.addWidget(self.task_list)
        lv.addWidget(hline())
        h = QHBoxLayout()
        h.addWidget(label("Images", "h2"))
        h.addStretch(1)
        h.addWidget(button("", "folder", "ghost", "Open the task folder", self.open_folder))
        h.addWidget(button("", "export", "ghost", "Write manifest.csv for the task (one row per image)", self.export_manifest))
        lv.addLayout(h)
        row = QHBoxLayout()
        row.addWidget(button("Live frame", "camera", "primary", "Label the current microscope frame", self.new_from_camera))
        row.addWidget(button("From file", "image", tip="Label a saved image", slot=self.new_from_file))
        lv.addLayout(row)
        self.docs = QListWidget()
        self.docs.itemClicked.connect(lambda it: it.data(Qt.UserRole) and self.open_doc(Path(it.data(Qt.UserRole))))
        self.docs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.docs.customContextMenuRequested.connect(self._doc_menu)
        lv.addWidget(self.docs, 1)
        f = QFormLayout()
        f.setSpacing(6)
        self.annotator = QLineEdit(self.ctx.operator)
        self.annotator.setPlaceholderText("kept with each label")
        f.addRow(label("Annotator", "caption"), self.annotator)
        self.stain = QComboBox()
        self.stain.addItems(STAINS)
        f.addRow(label("Stain", "caption"), self.stain)
        lv.addLayout(f)
        root.addWidget(left)

        # ---------------- centre: tools + stage
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(8)
        bar, tools = panel("bar", margins=(10, 8, 10, 8), spacing=6, horizontal=True)
        self.tool_group = QButtonGroup(self)
        self.tool_btns = {}
        for i, (key, text, ic, tip) in enumerate((("point", "Point", "point", "Add points (P)"),
                                                  ("polygon", "Outline", "polygon", "Draw outlines (D)"),
                                                  ("classify", "Classify", "tag", "Change the class of an object (C)"))):
            b = button("", ic, "seg", f"{text}: {tip}")
            b.setProperty("pos", "first" if i == 0 else ("last" if i == 2 else "mid"))
            b.setCheckable(True)
            b.setFixedWidth(44)
            b.toggled.connect(lambda on, b=b, ic=ic: b.setIcon(icon(ic, "#FFFFFF" if on else C["ink2"], 18)))
            b.toggled.connect(lambda on, k=key: on and self.canvas.set_tool(k))
            self.tool_group.addButton(b)
            self.tool_btns[key] = b
        seg = QHBoxLayout()
        seg.setSpacing(0)
        for b in self.tool_btns.values():
            seg.addWidget(b)
        tools.addLayout(seg)
        tools.addSpacing(10)
        self.class_box = QHBoxLayout()
        self.class_box.setSpacing(6)
        tools.addLayout(self.class_box)
        tools.addStretch(1)
        self.prefill_btn = button("Pre-fill", "wand", tip="", slot=self.run_prefill)
        tools.addWidget(self.prefill_btn)
        tools.addWidget(button("Confirm all", "check", tip="Accept every dashed model proposal, after checking them",
                               slot=self.confirm_all))
        tools.addWidget(button("", "undo", "ghost", "Undo (Ctrl+Z)", self.undo))
        mv.addWidget(bar)
        frame = QFrame()
        frame.setProperty("role", "stage")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        self.canvas = AnnotationCanvas()
        self.canvas.add_point.connect(self._add_point)
        self.canvas.polygon_done.connect(self._add_polygon)
        self.canvas.classify_at.connect(self._classify)
        self.canvas.delete_at.connect(self._delete)
        fl.addWidget(self.canvas)
        mv.addWidget(frame, 1)
        self.counts = label("", "muted", wrap=True)
        mv.addWidget(self.counts)
        root.addWidget(mid, 1)

        # ---------------- right: instructions + labels + status
        right = QWidget()
        right.setFixedWidth(370)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(12)
        p, lay = panel(spacing=6)
        lay.addWidget(label("How to label", "h2"))
        self.instr = QTextBrowser()
        self.instr.setFrameShape(QFrame.NoFrame)
        self.instr.setStyleSheet("QTextBrowser { border: none; padding: 0; }")
        lay.addWidget(self.instr)
        rv.addWidget(p, 1)
        self.label_box, self.label_form_lay = panel(spacing=6)
        self.label_box.layout().addWidget(label("Field labels", "h2"))
        self.label_form = QFormLayout()
        self.label_box.layout().addLayout(self.label_form)
        rv.addWidget(self.label_box)
        p, lay = panel(spacing=8)
        self.status_lbl = label("", "muted", wrap=True)
        lay.addWidget(self.status_lbl)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Notes on this image (optional)")
        self.notes.setFixedHeight(56)
        self.notes.textChanged.connect(self._mark_dirty)
        lay.addWidget(self.notes)
        h = QHBoxLayout()
        h.addWidget(button("Save", "save", tip="Ctrl+S", slot=lambda: self.save()))
        self.complete_btn = button("Mark complete", "check", "primary", slot=self.toggle_complete)
        h.addWidget(self.complete_btn, 1)
        lay.addLayout(h)
        rv.addWidget(p)
        self.calib_panel, lay = panel(spacing=6)
        lay.addWidget(label("Thresholds from your labels", "h2"))
        self.calib_lbl = label("", "caption", wrap=True)
        lay.addWidget(self.calib_lbl)
        self.calib_btn = button("Fit thresholds", "tune", slot=self.run_calibration)
        lay.addWidget(self.calib_btn)
        rv.addWidget(self.calib_panel)
        root.addWidget(right)

        for seq, fn in (("Ctrl+Z", self.undo), ("Ctrl+S", lambda: self.save()), ("P", lambda: self._tool("point")),
                        ("D", lambda: self._tool("polygon")), ("C", lambda: self._tool("classify"))):
            s = QShortcut(QKeySequence(seq), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)
        for i in range(9):
            s = QShortcut(QKeySequence(str(i + 1)), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(lambda i=i: self._pick_class(i))
        self.class_group = QButtonGroup(self)
        self.task_list.setCurrentRow(0)

    # ------------------------------------------------------------------ task / docs
    def _tool(self, k):
        if k in self.tool_btns and self.tool_btns[k].isEnabled():
            self.tool_btns[k].setChecked(True)

    def _doc_menu(self, pos):
        it = self.docs.itemAt(pos)
        if it is None or not it.data(Qt.UserRole):
            return
        m = QMenu(self)
        a = m.addAction("Delete this image and its labels")
        if m.exec(self.docs.mapToGlobal(pos)) == a:
            self.docs.setCurrentItem(it)
            self.delete_doc()

    def _on_task(self):
        if not hasattr(self, "class_group"):
            return
        self.save(quiet=True)
        t = self.task
        self.instr.setHtml(instructions_html(t))
        while self.class_box.count():
            w = self.class_box.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.class_group = QButtonGroup(self)
        for i, c in enumerate(t.classes):
            r, g, b = c.color
            col = f"rgb({r},{g},{b})"
            btn = QPushButton(f"{i + 1}  {c.name}")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Key {i + 1}")
            btn.setMinimumWidth(btn.fontMetrics().horizontalAdvance(btn.text()) + 34)
            btn.setStyleSheet(f"QPushButton {{ border:1.5px solid {col}; color:{col}; background:white; border-radius:13px;"
                              f" padding:3px 11px; font-weight:600; }} QPushButton:checked {{ background:{col}; color:white; }}")
            self.class_group.addButton(btn, i)
            self.class_box.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        geo = t.geometry
        self.tool_btns["point"].setEnabled(geo == "point")
        self.tool_btns["polygon"].setEnabled(geo == "polygon")
        self.tool_btns["classify"].setEnabled(geo != "field")
        for b in self.tool_btns.values():
            b.setVisible(geo != "field")
        self._tool("point" if geo == "point" else "polygon")
        self.prefill_btn.setEnabled(bool(t.prefill))
        self.prefill_btn.setVisible(geo != "field")
        self.prefill_btn.setToolTip(f"Let the current model propose {t.prefill}" if t.prefill else "")
        self.calib_panel.setVisible(bool(t.calibrates))
        self.calib_btn.setToolTip(f"Fits the {t.calibrates}" if t.calibrates else "")
        self.stain.setCurrentText(DEFAULT_STAIN.get(t.key, "H&E"))
        self._build_labels()
        self._show_calib()
        self.doc = None
        self.canvas.layer.set_doc(None, None)
        self.canvas.clear("Choose Live frame to label what is under the microscope,\nor From file for a saved image.")
        self.refresh_docs()
        self._update_status()

    def _build_labels(self):
        while self.label_form.rowCount():
            self.label_form.removeRow(0)
        self.label_widgets = {}
        for name, choices in self.task.field_labels.items():
            cb = QComboBox()
            cb.addItems(choices)
            cb.currentTextChanged.connect(lambda v, n=name: self._set_label(n, v))
            self.label_form.addRow(name.replace("_", " "), cb)
            self.label_widgets[name] = cb
        self.label_box.setVisible(bool(self.task.field_labels))

    def refresh_docs(self):
        self.docs.clear()
        if not self.store.list(self.task.key):
            it = QListWidgetItem("No images yet for this task.")
            it.setFlags(Qt.NoItemFlags)
            self.docs.addItem(it)
        for d in self.store.list(self.task.key):
            n = sum(d.counts().values())
            lab = ", ".join(f"{v}" for v in d.field_labels.values() if v)
            state = "Complete" if d.status == "complete" else "In progress"
            it = QListWidgetItem(icon("check", C["ok"]) if d.status == "complete" else icon("annotate", C["dab"]),
                                 f"{d.id}\n{state}, {n} labels{', score ' + lab if lab else ''}")
            it.setData(Qt.UserRole, str(d.dir))
            it.setToolTip(f"{d.status} · {d.annotator} · {d.modified} · µm/px {d.um_per_px}")
            self.docs.addItem(it)
            if self.doc is not None and d.id == self.doc.id:
                it.setSelected(True)

    def _new_doc(self, rgb, um, meta):
        self.save(quiet=True)
        if not self.annotator.text().strip():
            name, ok = QInputDialog.getText(self, "Annotator", "Your name (stored with every label):", text=self.ctx.operator)
            if not ok or not name.strip():
                return
            self.annotator.setText(name.strip())
        doc = self.store.new(self.task.key, rgb, um, self.annotator.text().strip(), stain=self.stain.currentText(), **meta)
        self.refresh_docs()
        self.open_doc(doc.dir)

    def new_from_camera(self):
        if self.ctx.last_frame is None or not self.ctx.live:
            error_box(self, "No frame", "Start the camera first (toolbar), then focus on the field to annotate.")
            return
        rgb = self.ctx.last_frame.copy()
        um = self.ctx.um_per_px(rgb.shape[1])
        if not um:
            error_box(self, "Not calibrated", "Calibrate the current objective first - annotations need the scale.")
            return
        cid = self.ctx.case.case_id if self.ctx.case else ""
        self._new_doc(rgb, um, dict(objective=self.ctx.objective, source=f"microscope {self.ctx.profile.name}", case_id=cid))

    def new_from_file(self):
        start = str(self.ctx.case.dir / "frames") if self.ctx.case else str(Path.home())
        f, _ = QFileDialog.getOpenFileName(self, "Image to annotate", start, "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not f:
            return
        rgb = read_rgb(Path(f))
        if rgb is None:
            error_box(self, "Open", "Not a readable image.")
            return
        meta = read_meta(Path(f))
        um = meta.get("um_per_px")
        if not um:
            um, ok = QInputDialog.getDouble(self, "Scale", "µm per pixel of this image:",
                                            self.ctx.um_per_px(rgb.shape[1]) or 0.25, 0.05, 10.0, 4)
            if not ok:
                return
        if meta.get("stain"):
            self.stain.setCurrentText(meta["stain"])
        self._new_doc(rgb, float(um), dict(objective=meta.get("objective", ""), source=Path(f).name,
                                           case_id=meta.get("case_id", "")))

    def open_doc(self, d: Path):
        if self.doc is not None and self.doc.dir == d:
            return
        self.save(quiet=True)
        try:
            doc = self.store.load(d)
        except Exception as e:  # noqa: BLE001
            error_box(self, "Open", e)
            return
        rgb = read_rgb(doc.image_path)
        if rgb is None:
            error_box(self, "Open", f"image missing: {doc.image_path}")
            return
        self.doc, self.rgb, self.undo_stack, self.dirty = doc, rgb, [], False
        self.canvas.set_image(rgb, fit=True)
        self.canvas.layer.set_doc(doc, self.task)
        for name, cb in self.label_widgets.items():
            cb.blockSignals(True)
            cb.setCurrentText(doc.field_labels.get(name, ""))
            cb.blockSignals(False)
        self.notes.blockSignals(True)
        self.notes.setPlainText(doc.notes)
        self.notes.blockSignals(False)
        self._update_status()

    def delete_doc(self):
        it = self.docs.currentItem()
        if it is None:
            return
        d = self.store.load(Path(it.data(Qt.UserRole)))
        if QMessageBox.question(self, "Delete", f"Delete image {d.id} and all its annotations?") != QMessageBox.Yes:
            return
        if self.doc is not None and self.doc.id == d.id:
            self.doc = None
            self.dirty = False
            self.canvas.layer.set_doc(None, None)
            self.canvas.clear("Deleted.")
        self.store.delete(d)
        self.refresh_docs()
        self._update_status()

    # ------------------------------------------------------------------ editing
    def _cls(self):
        i = self.class_group.checkedId()
        return self.task.classes[i].name if 0 <= i < len(self.task.classes) else None

    def _pick_class(self, i):
        b = self.class_group.button(i)
        if b is not None:
            b.setChecked(True)

    def _editable(self):
        if self.doc is None:
            return False
        if self.doc.status == "complete":
            self.status_lbl.setText("<span style='color:#b36b00'>This image is marked complete - press "
                                    "<b>Reopen</b> to edit it.</span>")
            return False
        return True

    def _push(self):
        self.undo_stack.append(copy.deepcopy(self.doc.items))
        del self.undo_stack[:-50]

    def _changed(self):
        self.dirty = True
        self.canvas.layer.update()
        self._update_status()

    def _mark_dirty(self):
        if self.doc is not None:
            self.doc.notes = self.notes.toPlainText()
            self.dirty = True

    def _tol(self):
        return max(6 * self.canvas.screen_px(), 3.0 / (self.doc.um_per_px or 0.25))

    def _add_point(self, x, y):
        if not self._editable() or not self._cls():
            return
        self._push()
        i = self.doc.item_at(x, y, self._tol())
        if i is not None and self.doc.items[i].kind == "point":
            it = self.doc.items[i]
            it.cls, it.source = self._cls(), "annotator"
        else:
            self.doc.items.append(Item("point", self._cls(), [[round(x, 1), round(y, 1)]]))
        self._changed()

    def _add_polygon(self, pts):
        if not self._editable() or not self._cls():
            return
        self._push()
        self.doc.items.append(Item("polygon", self._cls(), [[round(x, 1), round(y, 1)] for x, y in pts]))
        self._changed()

    def _classify(self, x, y):
        if not self._editable() or not self._cls():
            return
        i = self.doc.item_at(x, y, self._tol())
        if i is None:
            return
        self._push()
        it = self.doc.items[i]
        it.cls, it.source = self._cls(), "annotator"
        self._changed()

    def _delete(self, x, y):
        if not self._editable():
            return
        if self.canvas.layer.draft:
            self.canvas.cancel_draft()
            return
        i = self.doc.item_at(x, y, self._tol())
        if i is None:
            return
        self._push()
        self.doc.items.pop(i)
        self._changed()

    def undo(self):
        if self.doc is None or not self.undo_stack or self.doc.status == "complete":
            return
        self.doc.items = self.undo_stack.pop()
        self._changed()

    def confirm_all(self):
        if not self._editable():
            return
        n = self.doc.n_unconfirmed()
        if not n:
            return
        if QMessageBox.question(self, "Confirm all", f"Accept all {n} model proposals still shown dashed? Do this only "
                                                     "after checking every one of them.") != QMessageBox.Yes:
            return
        self._push()
        for it in self.doc.items:
            it.source = "annotator"
        self._changed()

    def _set_label(self, name, value):
        if self.doc is None:
            return
        if value:
            self.doc.field_labels[name] = value
        else:
            self.doc.field_labels.pop(name, None)
        self.dirty = True
        self._update_status()

    def run_prefill(self):
        if not self._editable():
            return
        if not self.doc.um_per_px:
            error_box(self, "Pre-fill", "This image has no scale (µm/px).")
            return
        doc, rgb, task = self.doc, self.rgb, self.task
        self.status_lbl.setText("Running the model…")

        def done(items):
            if self.doc is not doc:
                return
            self._push()
            tol = 3.0 / doc.um_per_px
            keep = [it for it in doc.items if it.source == "annotator"]
            pts = np.array([it.pts[0] for it in keep if it.kind == "point"]) if keep else np.zeros((0, 2))
            new = []
            for it in items:
                if it.kind == "point" and len(pts) and np.hypot(*(pts - it.pts[0]).T).min() <= tol:
                    continue
                new.append(it)
            doc.items = keep + new
            self._changed()
            self.status_lbl.setText(f"{len(new)} model proposals added (dashed). Check each one.")

        self.ctx.jobs.submit(lambda: prefill(task, rgb, doc.um_per_px, self.ctx.hub), done,
                             lambda e: error_box(self, "Pre-fill failed", e))

    # ------------------------------------------------------------------ save / status
    def save(self, quiet=False):
        if self.doc is None or not self.dirty:
            return
        self.doc.annotator = self.doc.annotator or self.annotator.text().strip()
        try:
            self.store.save(self.doc)
        except Exception as e:  # noqa: BLE001
            if not quiet:
                error_box(self, "Save", e)
            return
        self.dirty = False
        if not quiet:
            self.status_lbl.setText(f"Saved {self.doc.id}.")
        self.refresh_docs()

    def toggle_complete(self):
        if self.doc is None:
            return
        if self.doc.status == "complete":
            self.doc.status = "in progress"
        else:
            t = self.task
            n_unc = self.doc.n_unconfirmed()
            missing = [k for k in t.field_labels if not self.doc.field_labels.get(k)]
            if t.geometry == "field" and missing:
                error_box(self, "Complete", "Set the field label first.")
                return
            msgs = []
            if n_unc:
                msgs.append(f"{n_unc} model proposals are not confirmed and will NOT be used for training.")
            n = sum(self.doc.counts().values())
            if t.min_items and n < t.min_items and t.geometry != "field":
                msgs.append(f"Only {n} labelled objects (recommended at least {t.min_items} per image).")
            if missing and t.geometry != "field":
                msgs.append(f"Field label not set: {', '.join(missing)}.")
            if msgs and QMessageBox.question(self, "Mark complete", "\n".join(msgs) + "\n\nMark complete anyway?") != QMessageBox.Yes:
                return
            self.doc.status = "complete"
        self.dirty = True
        self.save(quiet=True)
        self._update_status()
        self._show_calib()

    def _update_status(self):
        d = self.doc
        if d is None:
            self.counts.setText("")
            self.status_lbl.setText("No image open. Start with Live frame or From file.")
            self.complete_btn.setText("Mark complete")
            return
        c = d.counts()
        parts = [f"{k} {v}" for k, v in c.items()]
        if d.n_unconfirmed():
            parts.append(f"{d.n_unconfirmed()} model proposals to check (dashed)")
        self.counts.setText(", ".join(parts) if parts else "No labels yet.")
        col = C["ok"] if d.status == "complete" else C["dab"]
        state = "Complete" if d.status == "complete" else "In progress"
        self.status_lbl.setText(f"<b>{d.id}</b><br><span style='color:{col}; font-weight:600'>{state}</span>. "
                                f"{d.width} × {d.height} px at {d.um_per_px or '?'} µm/px, {d.stain} {d.objective}, "
                                f"labelled by {d.annotator or 'unknown'}.")
        self.complete_btn.setText("Reopen for editing" if d.status == "complete" else "Mark complete")

    # ------------------------------------------------------------------ export / calibration
    def open_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.task_dir(self.task.key))))

    def export_manifest(self):
        self.save(quiet=True)
        p = self.store.export_manifest(self.task.key)
        QMessageBox.information(self, "Export", f"Wrote {p}\n\nEach image folder holds image.png, annotation.geojson "
                                                "(QuPath: File → Import objects), points.csv / mask.png and annotation.json.")

    def _show_calib(self):
        t = self.task
        if not t.calibrates:
            self.calib_lbl.clear()
            return
        names = {"erpr_intensity": "ER/PR intensity", "ki67_positivity": "Ki-67 positivity", "her2_membrane": "HER2 membrane"}
        cur = self.ctx.hub.calibration_status().get(names[t.key])
        n = sum(1 for d in self.store.list(t.key) if d.status == "complete")
        self.calib_lbl.setText((f"The app uses thresholds fitted to your labels ({cur.get('n_calibration', cur.get('source', ''))} "
                                "labels)." if cur else "The app still uses public-data thresholds.")
                               + f" {n} complete image{'s' * (n != 1)} in this task. Fitting re-measures every labelled cell.")

    def run_calibration(self):
        self.save(quiet=True)
        t = self.task
        if QMessageBox.question(self, "Calibrate", f"Fit the {t.calibrates} to all images of this task marked complete?\n\n"
                                "New analysis runs will use the new thresholds (the old file is replaced; earlier "
                                "results in cases are unchanged).") != QMessageBox.Yes:
            return
        self.calib_lbl.setText("Calibrating… (re-measures every labelled cell)")

        def done(rep):
            self._show_calib()
            QMessageBox.information(self, "Calibration", json.dumps(rep, indent=1, default=str))

        self.ctx.jobs.submit(lambda: calibrate(t.key, self.store, self.ctx.hub, calibration_dir()), done,
                             lambda e: (self._show_calib(), error_box(self, "Calibration", e)))

    def shutdown(self):
        self.save(quiet=True)
