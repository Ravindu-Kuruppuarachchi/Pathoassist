"""Microscope page: camera bar, module list (one per feature), the image stage with READY / scale-bar overlay,
and the result panel (headline result, progress to the minimum count, fields with click-to-correct, save)."""
from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QProgressBar, QScrollArea, QTextBrowser,
                               QVBoxLayout, QWidget)

from .casestore import FrameLogger, read_meta, read_rgb
from .modules import MODULES, MODULE_BY_KEY
from .theme import C, Chip, button, hline, label, panel, tone_color
from .ui_common import ImageCanvas, error_box, gate_hud, to_pixmap
from .widgets import ModuleRow

PAINT_INTERVAL = 0.04        # live view repaint at most 25 fps


class AnalysisPage(QWidget):
    def __init__(self, ctx, camera_bar: QWidget, toast=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.toast = toast or (lambda *a, **k: None)
        self.modules = {}             # key -> module instance for the open case
        self.view_fid = None          # field shown (None = live)
        self._last_paint = 0.0
        self._skipped = None          # (rgb, um, meta) of the last duplicate, for "Add anyway"
        self._rows = {}
        self._build(camera_bar)
        ctx.frame.connect(self._on_frame)
        ctx.case_changed.connect(self._on_case)
        ctx.case_updated.connect(self._update_status_chips)
        ctx.objective_changed.connect(lambda _: self._update_scale())
        ctx.profile_changed.connect(self._update_scale)
        ctx.jobs.busy_changed.connect(self._on_busy)
        self._on_case(ctx.case)

    # ================================================================== layout
    def _build(self, camera_bar):
        self.setObjectName("page")
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(12)
        root.addWidget(camera_bar)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        # ---- module list
        p, lay = panel(margins=(8, 12, 8, 8), spacing=6)
        p.setFixedWidth(250)
        lay.addWidget(label("  Analysis", "h2"))
        self.mod_list = QListWidget()
        self.mod_list.setObjectName("modules")
        self.mod_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        group = None
        for m in MODULES:
            if m.group != group:
                group = m.group
                h = QListWidgetItem(group)
                h.setFlags(Qt.NoItemFlags)
                h.setForeground(tone_color("neutral"))
                f = h.font()
                f.setPixelSize(12)
                f.setWeight(f.Weight.DemiBold)
                h.setFont(f)
                h.setSizeHint(QSize(200, 30))
                self.mod_list.addItem(h)
            it = QListWidgetItem()
            it.setData(Qt.UserRole, m.key)
            row = ModuleRow(m)
            it.setSizeHint(row.sizeHint())
            self.mod_list.addItem(it)
            self.mod_list.setItemWidget(it, row)
            self._rows[m.key] = (it, row)
        self.mod_list.currentItemChanged.connect(lambda *_: self._on_module())
        lay.addWidget(self.mod_list, 1)
        body.addWidget(p)

        # ---- stage
        stage = QWidget()
        sv = QVBoxLayout(stage)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(8)
        frame = QFrame()
        frame.setProperty("role", "stage")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas(message="The live microscope image appears here.\n\nStart the camera above, "
                                          "or use Open images to analyse saved frames.")
        self.canvas.clicked.connect(self._on_click)
        fl.addWidget(self.canvas)
        sv.addWidget(frame, 1)

        bar, bl = panel("bar", margins=(10, 8, 10, 8), spacing=8, horizontal=True)
        self.analyse_btn = button("Analyse field", "analyse", "primary", "Analyse the current frame (Enter)",
                                  lambda: self.analyse(manual=True))
        bl.addWidget(self.analyse_btn)
        self.auto_box = QCheckBox("Analyse automatically when Ready")
        self.auto_box.setToolTip("Each time the stage stops on a sharp field, that field is analysed.")
        bl.addWidget(self.auto_box)
        self.force_btn = button("Count anyway", tip="Count the last skipped field although it overlaps an earlier one",
                                slot=self._force_add)
        self.force_btn.setVisible(False)
        bl.addWidget(self.force_btn)
        bl.addStretch(1)
        self.busy = Chip("")
        bl.addWidget(self.busy)
        self.live_btn = button("Back to live", "live", "ghost", "Return to the live image (Esc)", self.back_to_live)
        bl.addWidget(self.live_btn)
        bl.addWidget(button("Open images", "files", "ghost", "Analyse saved frames from disk", self.load_files))
        sv.addWidget(bar)
        self.msg = label("", "muted", wrap=True)
        sv.addWidget(self.msg)
        body.addWidget(stage, 1)

        # ---- result panel
        rp = QWidget()
        rp.setMaximumWidth(356)
        rv = QVBoxLayout(rp)
        rv.setContentsMargins(0, 0, 4, 0)
        rv.setSpacing(12)

        p, lay = panel(spacing=6)
        h = QHBoxLayout()
        self.mod_title = label("", "h2")
        h.addWidget(self.mod_title, 1)
        self.obj_chip = Chip("")
        h.addWidget(self.obj_chip)
        lay.addLayout(h)
        self.value = label("-", "value")
        lay.addWidget(self.value)
        self.caption = label("", "sub", wrap=True)
        lay.addWidget(self.caption)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        lay.addWidget(self.progress)
        self.progress_lbl = label("", "caption")
        lay.addWidget(self.progress_lbl)
        lay.addWidget(hline())
        self.details = QTextBrowser()
        self.details.setFrameShape(QFrame.NoFrame)
        self.details.setStyleSheet("QTextBrowser { border: none; padding: 0; }")
        self.details.setMinimumHeight(150)
        lay.addWidget(self.details)
        row = QHBoxLayout()
        self.save_btn = button("Save to case", "save", "primary", slot=self.save_to_case)
        row.addWidget(self.save_btn, 1)
        self.more_btn = button("", "settings", tip="New run, run folder")
        self.more_btn.clicked.connect(self._more_menu)
        row.addWidget(self.more_btn)
        lay.addLayout(row)
        rv.addWidget(p)

        p, lay = panel(spacing=6)
        lay.addWidget(label("Fields", "h2"))
        self.click_help = label("", "caption", wrap=True)
        lay.addWidget(self.click_help)
        self.fields = QListWidget()
        self.fields.setViewMode(QListWidget.IconMode)
        self.fields.setIconSize(QSize(132, 99))
        self.fields.setResizeMode(QListWidget.Adjust)
        self.fields.setMovement(QListWidget.Static)
        self.fields.setSpacing(4)
        self.fields.setMinimumHeight(140)
        self.fields.setStyleSheet("QListWidget { border: none; }")
        self.fields.itemClicked.connect(lambda it: self.show_field(it.data(Qt.UserRole)))
        self.fields.setContextMenuPolicy(Qt.CustomContextMenu)
        self.fields.customContextMenuRequested.connect(self._field_menu)
        lay.addWidget(self.fields, 1)
        rv.addWidget(p, 1)

        p, lay = panel(spacing=6)
        self.howto_btn = button("How to do this", "help", "ghost")
        self.howto_btn.setCheckable(True)
        lay.addWidget(self.howto_btn)
        self.howto = QTextBrowser()
        self.howto.setFrameShape(QFrame.NoFrame)
        self.howto.setStyleSheet("QTextBrowser { border: none; padding: 0; }")
        self.howto.setMinimumHeight(170)
        self.howto.setVisible(False)
        self.howto_btn.toggled.connect(self.howto.setVisible)
        lay.addWidget(self.howto)
        self.opt_box = QWidget()
        self.opt_layout = QVBoxLayout(self.opt_box)
        self.opt_layout.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.opt_box)
        self.minimap = label("")
        self.minimap.setAlignment(Qt.AlignCenter)
        self.minimap.setVisible(False)
        lay.addWidget(self.minimap)
        rv.addWidget(p)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rp)
        scroll.setFixedWidth(372)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body.addWidget(scroll)

        for key, fn in (("Return", lambda: self.analyse(manual=True)), ("Enter", lambda: self.analyse(manual=True)),
                        ("Escape", self.back_to_live)):
            s = QShortcut(QKeySequence(key), self)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(fn)
        self.select_module(self.ctx.settings.value("module", "ER", str))

    # ================================================================== state
    @property
    def key(self):
        it = self.mod_list.currentItem()
        return it.data(Qt.UserRole) if it is not None else MODULES[0].key

    @property
    def module(self):
        return self.modules.get(self.key)

    def _module_cls(self):
        return MODULE_BY_KEY[self.key]

    def select_module(self, key):
        if key in self._rows:
            self.mod_list.setCurrentItem(self._rows[key][0])

    def has_unsaved(self) -> list[str]:
        return [m.title for m in self.modules.values() if m.dirty and m.fields]

    def _on_case(self, case):
        self.modules.clear()
        self.back_to_live()
        self._on_module()
        self._update_status_chips()

    def _update_status_chips(self):
        c = self.ctx.case
        for key, (_, row) in self._rows.items():
            m = self.modules.get(key)
            saved = False
            if c is not None:
                saved = (key in c.biomarkers or (key in ("mitoses", "pleomorphism", "tubules") and c.grade.c[key].auto_score
                                                 is not None) or (key == "region" and "Tumour region" in c.extras)
                         or (key == "nuclei" and "Nuclei" in c.extras))
            if m is not None and m.dirty and m.fields:
                row.set_status(f"{len(m.fields)} unsaved", "warn")
            elif saved:
                row.set_status("Saved", "good")
            else:
                row.set_status("", "neutral")

    def _on_module(self):
        cls = self._module_cls()
        self.ctx.settings.setValue("module", cls.key)
        self.mod_title.setText(cls.title)
        self.howto.setHtml(cls.how_to)
        self.click_help.setText(_plain(cls.click_help) if cls.click_help else "Click a field to look at it again.")
        self.save_btn.setToolTip(f"Writes the {cls.saves_to} into the case. The change is recorded in the audit log.")
        while self.opt_layout.count():
            w = self.opt_layout.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        m = self.module
        for name, text, default in cls.options_spec:
            cb = QCheckBox(text)
            cb.setChecked(bool(m.options.get(name, default)) if m else
                          self.ctx.settings.value(f"opt/{cls.key}/{name}", default, bool))
            cb.setEnabled(m is None or not m.fields)
            if m is not None and m.fields:
                cb.setToolTip("Applies to a new run")
            cb.toggled.connect(lambda on, n=name: self._set_option(n, on))
            self.opt_layout.addWidget(cb)
        self.minimap.setVisible(cls.key == "region")
        self._update_scale()
        self._refresh_fields()
        self._refresh_result()
        self.back_to_live()

    def _set_option(self, name, on):
        self.ctx.settings.setValue(f"opt/{self.key}/{name}", on)
        m = self.module
        if m is not None and not m.fields:
            m.options[name] = on

    def _options(self):
        cls = self._module_cls()
        return {n: self.ctx.settings.value(f"opt/{cls.key}/{n}", d, bool) for n, _, d in cls.options_spec}

    def _ensure_module(self):
        if self.ctx.case is None:
            QMessageBox.information(self, "Open a case first",
                                    "Results are stored under the patient's case. Create or open a case on the Cases page.")
            return None
        if self.ctx.case.signed_by:
            error_box(self, "Case is signed", f"Case {self.ctx.case.case_id} is signed and can no longer change. "
                                              "Open another case to analyse.")
            return None
        if self.module is None:
            self.modules[self.key] = self._module_cls()(
                self.ctx.hub, self.ctx.case, self.ctx.operator, self._options(),
                dict(app_version=self.ctx.app_version, hpf_diameter_um=self.ctx.hpf_diameter_um()))
        return self.module

    def _update_scale(self):
        cls = self._module_cls()
        um = self.ctx.um_per_px()
        obj = self.ctx.objective
        ok = obj == cls.objective or (cls.objective in ("20x", "40x") and obj in ("20x", "40x"))
        self.obj_chip.set(f"Use {cls.objective}" if not ok else f"{obj}", "warn" if not ok else "accent")
        self.obj_chip.setToolTip(f"Recommended objective {cls.objective}; current {obj}.")
        if self.view_fid is None:
            self.canvas.set_hud(um_per_px=um, note=None if um else f"{obj} is not calibrated")

    # ================================================================== live view
    def _on_frame(self, rgb, st):
        if self.auto_box.isChecked() and st.dwell and self.view_fid is None and self.ctx.jobs.pending == 0:
            self.analyse(manual=False)
        if self.view_fid is not None or not self.isVisible():
            return
        now = time.perf_counter()
        if now - self._last_paint >= PAINT_INTERVAL:
            self._last_paint = now
            text, col = gate_hud(st, True)
            self.canvas.set_hud(chip=text, chip_color=col)
            self.canvas.set_image(rgb)

    def camera_stopped(self):
        text, col = gate_hud(None, False)
        self.canvas.set_hud(chip=text, chip_color=col)

    def back_to_live(self):
        self.view_fid = None
        self.fields.clearSelection()
        um = self.ctx.um_per_px()
        if self.ctx.last_frame is not None and self.ctx.live:
            self.canvas.set_hud(note=None if um else f"{self.ctx.objective} is not calibrated", um_per_px=um)
            self.canvas.set_image(self.ctx.last_frame, fit=True)
        self.live_btn.setVisible(False)

    def show_field(self, fid):
        m = self.module
        if m is None or fid not in m.fields:
            return
        self.view_fid = fid
        self.live_btn.setVisible(True)
        hint = "click cells to correct" if m.click_help else "review"
        self.canvas.set_hud(chip=f"Field {fid + 1}", chip_color=C["haem"], note=hint, um_per_px=m.fields[fid].um_per_px)
        self.ctx.jobs.submit(lambda: m.render(fid), lambda img: self._show_render(fid, img),
                             lambda e: error_box(self, "Could not draw the field", e))

    def _show_render(self, fid, img):
        if self.view_fid == fid:
            self.canvas.set_image(img)

    # ================================================================== analyse
    def analyse(self, manual=True, rgb=None, um=None, meta=None, force=False, quiet=False):
        if rgb is None:
            if self.view_fid is not None and manual:
                self.back_to_live()
            if self.ctx.last_frame is None or not self.ctx.live:
                if manual:
                    QMessageBox.information(self, "No live image", "Start the camera in the bar above, or use Open images "
                                                                   "to analyse saved frames.")
                return
            rgb = self.ctx.last_frame.copy()
            um = self.ctx.um_per_px(rgb.shape[1])
            meta = self.ctx.frame_meta()
            if not um:
                error_box(self, "Objective not calibrated",
                          f"{self.ctx.objective} on {self.ctx.profile.name} has no scale yet. Use Calibrate in the bar "
                          "above with a stage micrometer; every measurement depends on it.")
                self.auto_box.setChecked(False)
                return
        m = self._ensure_module()
        if m is None:
            self.auto_box.setChecked(False)
            return
        self.msg.setText("Analysing the field…")

        def job():
            out = m.add_frame(rgb, um, meta, force=force)
            return out, m.summary_html(), m.minimap(), m.headline()

        self.ctx.jobs.submit(job, lambda r: self._analysed(m, r, manual and not quiet, (rgb, um, meta)),
                             lambda e: self._msg_err(f"Analysis failed: {e}"))

    def _analysed(self, m, r, show, frame):
        out, html, mm, head = r
        if m is not self.module:
            return
        self.msg.setText(out.message)
        if not out.accepted:
            self.toast(out.message, "warn")
        self._skipped = None if out.accepted else frame
        self.force_btn.setVisible(not out.accepted and self.key in ("ER", "PR", "Ki67", "HER2") and "overlaps" in out.message)
        self._set_result(html, mm, head)
        if out.accepted:
            self._add_field_item(out.field_id)
            self._refresh_options_enabled()
            self._update_status_chips()
            if show:
                self.show_field(out.field_id)

    def _set_result(self, html, mm, head):
        self.details.setHtml(html)
        self._set_minimap(mm)
        self.value.setText(head["value"])
        self.value.setStyleSheet(f"color: {tone_color(head['tone']).name()};" if head["tone"] != "neutral" else "")
        self.caption.setText(head["caption"])
        pr = head.get("progress")
        self.progress.setVisible(pr is not None)
        self.progress_lbl.setVisible(pr is not None)
        if pr is not None:
            cur, target, unit = pr
            self.progress.setMaximum(1000)
            self.progress.setValue(int(1000 * min(cur / target, 1.0)) if target else 1000)
            done = cur >= target
            self.progress.setProperty("done", "true" if done else "false")
            self.progress.style().unpolish(self.progress)
            self.progress.style().polish(self.progress)
            cur_s = f"{cur:g}" if isinstance(cur, float) else f"{cur:,}"
            self.progress_lbl.setText(f"{cur_s} of {target:g} {unit}" + ("  (enough)" if done else ""))

    def _force_add(self):
        if self._skipped is not None:
            rgb, um, meta = self._skipped
            self._skipped = None
            self.force_btn.setVisible(False)
            self.analyse(rgb=rgb, um=um, meta=meta, force=True)

    def _msg_err(self, text):
        self.msg.setText(text)
        self.toast(text, "bad")

    def load_files(self):
        if self._ensure_module() is None:
            return
        start = str(FrameLogger.case_frames_dir(self.ctx.case, self.module.stain))
        files, _ = QFileDialog.getOpenFileNames(self, "Open images to analyse", start,
                                                "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if not files:
            return
        asked = None
        for f in files:
            p = Path(f)
            rgb = read_rgb(p)
            if rgb is None:
                continue
            meta = read_meta(p)
            um = meta.get("um_per_px")
            if not um:
                if asked is None:
                    v, ok = QInputDialog.getDouble(self, "Image scale", f"Micrometres per pixel for these images.\n"
                                                   f"{p.name} has no calibration record.",
                                                   self.ctx.um_per_px(rgb.shape[1]) or 0.25, 0.05, 10.0, 4)
                    if not ok:
                        return
                    asked = v
                um = asked
            inside = self.ctx.case.dir in p.resolve().parents
            m2 = dict(objective=meta.get("objective", ""), source_file=p.name, already_saved=inside)
            self.analyse(manual=False, rgb=rgb, um=float(um), meta=m2)

    # ================================================================== fields / corrections
    def _add_field_item(self, fid):
        m = self.module
        rec = m.fields[fid]
        it = QListWidgetItem(QIcon(to_pixmap(rec.thumb)), f"{fid + 1}  {rec.caption}")
        it.setData(Qt.UserRole, fid)
        it.setToolTip(rec.caption)
        it.setSizeHint(QSize(150, 140))
        self.fields.addItem(it)
        self.fields.scrollToBottom()

    def _refresh_fields(self):
        self.fields.clear()
        m = self.module
        if m is None:
            return
        for fid in m.fields:
            self._add_field_item(fid)

    def _refresh_result(self):
        m = self.module
        if m is None:
            self._set_result("<p style='color:#8A93A3'>Move to a field of invasive tumour, wait for Ready, then press "
                             "Analyse field.</p>", None, dict(value="-", caption="No fields analysed yet",
                                                               tone="neutral", progress=None))
            return
        self.ctx.jobs.submit(lambda: (m.summary_html(), m.minimap(), m.headline()),
                             lambda r: self._set_result(*r) if m is self.module else None)

    def _set_minimap(self, mm):
        if mm is None or self.key != "region":
            self.minimap.clear()
            return
        self.minimap.setPixmap(to_pixmap(mm).scaledToWidth(320, Qt.SmoothTransformation))

    def _refresh_options_enabled(self):
        for i in range(self.opt_layout.count()):
            w = self.opt_layout.itemAt(i).widget()
            if w:
                w.setEnabled(False)
                w.setToolTip("Applies to a new run")

    def _on_click(self, x, y, mods):
        m = self.module
        fid = self.view_fid
        if m is None or fid is None or not m.click_help or self.ctx.case is None or self.ctx.case.signed_by:
            return

        def job():
            msg = m.click(fid, x, y, mods)
            if msg is None:
                return None
            m.refresh_thumb(fid)
            return msg, m.render(fid), m.summary_html(), m.headline()

        self.ctx.jobs.submit(job, lambda r: self._clicked(m, fid, r), lambda e: self._msg_err(f"Correction failed: {e}"))

    def _clicked(self, m, fid, r):
        if r is None:
            self.msg.setText("Nothing under the pointer. Click on a nucleus, cell or circle.")
            return
        msg, img, html, head = r
        self.msg.setText(f"Field {fid + 1}: {msg}. Kept as a training label.")
        if self.view_fid == fid:
            self.canvas.set_image(img)
        self._set_result(html, None, head)
        self._update_status_chips()
        for i in range(self.fields.count()):
            it = self.fields.item(i)
            if it.data(Qt.UserRole) == fid:
                rec = m.fields[fid]
                it.setIcon(QIcon(to_pixmap(rec.thumb)))
                it.setText(f"{fid + 1}  {rec.caption}")

    def _field_menu(self, pos):
        it = self.fields.itemAt(pos)
        if it is None or self.module is None:
            return
        fid = it.data(Qt.UserRole)
        menu = QMenu(self)
        a_show = menu.addAction("Review and correct")
        a_del = menu.addAction("Remove from the count")
        act = menu.exec(self.fields.mapToGlobal(pos))
        if act == a_show:
            self.show_field(fid)
        elif act == a_del:
            m = self.module
            try:
                m.remove(fid)
            except ValueError as e:
                error_box(self, "Cannot remove this field", e)
                return
            self.fields.takeItem(self.fields.row(it))
            if self.view_fid == fid:
                self.back_to_live()
            self._refresh_result()

    def _more_menu(self):
        menu = QMenu(self)
        a_new = menu.addAction("Start a new run")
        a_dir = menu.addAction("Open the run folder")
        act = menu.exec(self.more_btn.mapToGlobal(self.more_btn.rect().bottomLeft()))
        if act == a_new:
            self.new_run()
        elif act == a_dir:
            self.open_run_folder()

    # ================================================================== save
    def save_to_case(self):
        m = self.module
        case = self.ctx.case
        if m is None or case is None:
            error_box(self, "Nothing to save", "Analyse at least one field first.")
            return
        if case.signed_by:
            error_box(self, "Case is signed", "The case is signed and can no longer change.")
            return
        ok, why = m.can_save()
        if not ok:
            error_box(self, "Not ready to save", why)
            return
        existing = m.key in case.biomarkers
        if m.key in ("mitoses", "pleomorphism", "tubules"):
            existing = case.grade.c[m.key].score is not None
        if existing and QMessageBox.question(self, "Replace the saved result?",
                                             f"Case {case.case_id} already has a {m.title} result. Replace it with this run? "
                                             "The audit log keeps both.") != QMessageBox.Yes:
            return
        self.msg.setText("Saving to the case…")

        def done(text):
            m.dirty = False
            self.ctx.save_case()
            self.msg.setText(f"Saved to {case.case_id}: {text}")
            self.toast(f"Saved to case {case.case_id}", "good")
            self._update_status_chips()

        self.ctx.jobs.submit(m.save_to_case, done, lambda e: error_box(self, "Save failed", e))

    def new_run(self):
        m = self.module
        if m is not None and m.dirty and m.fields and QMessageBox.question(
                self, "Start a new run?", f"The {len(m.fields)} {m.title} fields not saved to the case will be "
                                          "discarded.") != QMessageBox.Yes:
            return
        self.modules.pop(self.key, None)
        self.msg.clear()
        self._on_module()
        self._update_status_chips()

    def open_run_folder(self):
        m = self.module
        if m is None:
            return
        p = m.run_dir if m.run_dir.exists() else m.run_dir.parent
        p.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _on_busy(self, busy):
        self.busy.set("Working…" if busy else "", "accent")


def _plain(html):
    import re
    return re.sub(r"<[^>]+>", "", html)
