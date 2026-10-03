"""Main window: sidebar (navigation, the open case as a slide label, operator) and the pages
Home, Cases, Microscope, Report and Annotate. The camera bar lives on the Microscope page."""
from __future__ import annotations

import logging
import time

from PySide6.QtCore import QSize, Qt, QUrl, Slot
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
                               QStackedWidget, QTextBrowser, QVBoxLayout, QWidget)

from . import __version__
from .camera import list_cameras
from .casestore import FrameLogger
from .context import AppContext
from .dialog_calibration import CalibrationDialog
from .microscope import MicroscopeProfile
from .page_analysis import AnalysisPage
from .page_annotate import AnnotatePage
from .page_cases import CasesPage
from .page_home import HomePage
from .page_report import ReportPage
from .paths import captures_dir, logs_dir, profiles_dir, user_data_dir
from .theme import C, button, icon, label, panel, pixmap
from .ui_common import CaptureThread, Toast, error_box
from .widgets import Segmented, SlideLabel

RESOLUTIONS = [(640, 480), (1280, 720), (1280, 960), (1920, 1080), (2592, 1944), (3840, 2160)]
PAGES = [("home", "Home", "home"), ("cases", "Cases", "cases"), ("scope", "Microscope", "scope"),
         ("report", "Report", "report"), ("annotate", "Annotate", "annotate")]

GUIDE = """<h2>Using PathoAssist</h2>
<p><b>Research use only.</b> Every automatic result must be reviewed by the pathologist. The models were trained on
public data and are not yet validated on local cases.</p>
<h3>Once per microscope</h3><ol>
<li>Plug in the USB camera. On <b>Microscope</b>, choose it and press <b>Start</b>. Set the light, then tick
<b>Lock colour</b> so exposure and white balance stay fixed.</li>
<li>Press <b>Calibrate</b> and photograph a stage micrometer at every objective you use (4x, 10x, 20x, 40x).</li>
<li>For each objective, move to empty glass and choose <b>Capture blank</b> from the camera menu (⋯).</li></ol>
<h3>For each case</h3><ol>
<li><b>Cases → New case</b>: accession number, <b>patient ID</b>, specimen. The slide label in the sidebar always
shows the open case and patient.</li>
<li><b>Microscope</b>: pick an analysis on the left (ER, PR, Ki-67, HER2, tumour map, mitoses, pleomorphism,
tubules, nuclei). Choose the objective (keys 1-4 or Auto). When the stage is still and sharp the image shows
<b>Ready</b>: press <b>Enter</b>, or tick <i>Analyse automatically when Ready</i> and move from field to field.</li>
<li>Click a field thumbnail to look at it; click cells to correct them. Each correction is kept as a training
label. Press <b>Save to case</b>. Every analysis is separate and can be run again.</li>
<li><b>Report</b>: check the grade components and biomarkers, change anything with a reason, create a draft PDF,
then <b>Sign</b>. Signing locks the case; every step is in the audit log.</li></ol>
<h3>Annotating new data</h3>
<p><b>Annotate</b>: pick a task. Each one says exactly what to label. Take the live frame or a saved image, let the
model pre-fill if you like, correct it, and mark the image complete. ER/PR, Ki-67 and HER2 labels can re-fit the
app's thresholds to your laboratory.</p>
<h3>Keys</h3><p>Ctrl+1…5 pages. Microscope: Enter analyse, Esc back to live, 1-4 objective, C save frame,
wheel zoom, middle-drag or Space-drag pan, F fit. Annotate: P point, D polygon, C classify, 1-9 class,
right-click delete, Ctrl+Z undo, Ctrl+S save.</p>"""


class SettingsDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)
        f = QFormLayout(self)
        f.setContentsMargins(20, 20, 20, 20)
        f.setSpacing(12)
        self.op = QLineEdit(ctx.operator)
        f.addRow("Your name", self.op)
        self.lab = QLineEdit(ctx.settings.value("lab_name", "", str))
        f.addRow("Laboratory (report header)", self.lab)
        f.addRow(label(f"Data is stored in {user_data_dir()}. Set PATHOASSIST_DATA to move it.", "caption", wrap=True))
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


class CameraBar(QFrame):
    """Camera, resolution, start/stop, colour lock, objective and the frame actions."""

    def __init__(self, win):
        super().__init__()
        self.setProperty("role", "bar")
        h = QHBoxLayout(self)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(8)
        cam_icon = QLabel()
        cam_icon.setPixmap(pixmap("camera", C["ink2"], 18))
        h.addWidget(cam_icon)
        self.cam = QComboBox()
        self.cam.setMinimumWidth(190)
        self.cam.setToolTip("Microscope camera")
        h.addWidget(self.cam)
        h.addWidget(button("", "refresh", "ghost", "Look for cameras again", win.refresh_cameras))
        self.res = QComboBox()
        for w, hh in RESOLUTIONS:
            self.res.addItem(f"{w} × {hh}", (w, hh))
        self.res.setToolTip("Camera resolution (the scale follows it automatically)")
        h.addWidget(self.res)
        self.start = button("Start", "play", "primary", "Start or stop the camera", win.toggle_camera)
        self.start.setMinimumWidth(92)
        h.addWidget(self.start)
        self.lock = QCheckBox("Lock colour")
        self.lock.setToolTip("Fix exposure and white balance so stain colours stay comparable between fields. "
                             "Set the light first, then lock.")
        h.addWidget(self.lock)
        self.flat = QCheckBox("Flat-field")
        self.flat.setChecked(True)
        self.flat.setToolTip("Correct uneven light with the blank of this objective")
        h.addWidget(self.flat)
        sep = QFrame()
        sep.setFixedSize(1, 24)
        sep.setStyleSheet(f"background:{C['line']};")
        h.addSpacing(6)
        h.addWidget(sep)
        h.addSpacing(6)
        h.addWidget(label("Objective", "muted"))
        self.obj = Segmented()
        h.addWidget(self.obj)
        h.addWidget(button("Auto", "wand", tip="Recognise the objective from the image", slot=lambda: win.detect_objective(True)))
        h.addStretch(1)
        h.addWidget(button("Calibrate", "ruler", tip="Measure µm per pixel with a stage micrometer", slot=win.calibrate))
        h.addWidget(button("Save frame", "image", tip="Save the current frame into the case (C)", slot=win.capture_frame))
        self.more = button("⋯", None, "ghost", "Flat-field blank, microscope profiles")
        self.more.clicked.connect(win.camera_menu)
        h.addWidget(self.more)

    def set_running(self, on):
        self.start.setText("Stop" if on else "Start")
        self.start.setIcon(icon("stop" if on else "play", "#FFFFFF", 18))
        self.cam.setEnabled(not on)
        self.res.setEnabled(not on)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PathoAssist")
        self.resize(1560, 980)
        self.setMinimumSize(1200, 760)
        self.ctx = AppContext(self)
        self._capture: CaptureThread | None = None
        self._last_status = 0.0
        self._auto_mag_busy = False
        self.toast = Toast(self)

        self.cambar = CameraBar(self)
        self.cambar.res.setCurrentIndex(int(self.ctx.settings.value("resolution_index", 3)))
        self.cambar.lock.toggled.connect(lambda on: self._capture and self._capture.set_auto(not on))
        self.cambar.flat.toggled.connect(self._flat_toggled)
        self.cambar.obj.changed.connect(self.ctx.set_objective)

        self.cases = CasesPage(self.ctx)
        self.analysis = AnalysisPage(self.ctx, self.cambar, toast=self.toast.show_text)
        self.report = ReportPage(self.ctx, goto_module=self.goto_module, toast=self.toast.show_text)
        self.annotate = AnnotatePage(self.ctx)
        self.home = HomePage(self.ctx, dict(new_case=self.new_case, start_camera=self.start_from_home,
                                            calibrate=self.calibrate, blank=self.capture_blank,
                                            annotate=lambda: self.goto("annotate")))
        self.home.go.connect(self.goto)
        self.home.open_case.connect(self.open_case_id)
        self.pages = dict(home=self.home, cases=self.cases, scope=self.analysis, report=self.report, annotate=self.annotate)

        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        for k, _, _ in PAGES:
            self.stack.addWidget(self.pages[k])
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self._build_menu()

        self.fps_lbl = QLabel()
        self.statusBar().addPermanentWidget(self.fps_lbl)
        self.ctx.status.connect(lambda m: self.statusBar().showMessage(m, 5000))
        self.ctx.operator_changed.connect(lambda _: (self._update_operator(), self.home.refresh()))
        self.ctx.case_changed.connect(self._on_case)
        self.ctx.case_updated.connect(lambda: self.slide.set_case(self.ctx.case))
        self.ctx.case_updated.connect(self.cases.refresh)
        self.ctx.objective_changed.connect(self._on_objective)
        self.ctx.profile_changed.connect(self._fill_objectives)
        self._orig_set_case = self.ctx.set_case
        self.ctx.set_case = self._guarded_set_case

        for i in range(4):
            s = QShortcut(QKeySequence(str(i + 1)), self.analysis)
            s.setContext(Qt.WidgetWithChildrenShortcut)
            s.activated.connect(lambda i=i: self._objective_hotkey(i))
        s = QShortcut(QKeySequence("C"), self.analysis)
        s.setContext(Qt.WidgetWithChildrenShortcut)
        s.activated.connect(self.capture_frame)
        for i, (k, _, _) in enumerate(PAGES):
            s = QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self)
            s.activated.connect(lambda k=k: self.goto(k))

        self._fill_objectives()
        self._update_operator()
        self.refresh_cameras()
        self.goto("home")

    # ================================================================== sidebar / navigation
    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(232)
        v = QVBoxLayout(side)
        v.setContentsMargins(0, 18, 0, 14)
        v.setSpacing(2)
        brand = QHBoxLayout()
        brand.setContentsMargins(16, 0, 16, 0)
        mark = QLabel()
        mark.setPixmap(pixmap("scope", C["haem"], 26, 2.0))
        brand.addWidget(mark)
        bv = QVBoxLayout()
        bv.setSpacing(0)
        t = label("PathoAssist")
        t.setStyleSheet("font-size:17px; font-weight:600;")
        bv.addWidget(t)
        bv.addWidget(label("Breast pathology assistant", "caption"))
        brand.addLayout(bv, 1)
        v.addLayout(brand)
        v.addSpacing(18)
        self.nav = QButtonGroup(self)
        self.nav_btns = {}
        for k, text, ic in PAGES:
            b = QPushButton(f"  {text}")
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setIcon(icon(ic, C["ink2"], 20))
            b.setIconSize(QSize(20, 20))
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=k: self.goto(k))
            self.nav.addButton(b)
            self.nav_btns[k] = b
            v.addWidget(b)
        v.addStretch(1)
        box = QVBoxLayout()
        box.setContentsMargins(14, 0, 14, 0)
        box.setSpacing(8)
        box.addWidget(label("Open case", "caption"))
        self.slide = SlideLabel()
        self.slide.clicked.connect(lambda: self.goto("report" if self.ctx.case else "cases"))
        box.addWidget(self.slide)
        v.addLayout(box)
        v.addSpacing(12)
        row = QHBoxLayout()
        row.setContentsMargins(10, 0, 10, 0)
        self.op_btn = button("", "user", "ghost", "Your name and settings", self.settings_dialog)
        self.op_btn.setStyleSheet("text-align:left;")
        row.addWidget(self.op_btn, 1)
        row.addWidget(button("", "help", "ghost", "How to use PathoAssist (F1)", self.show_guide))
        v.addLayout(row)
        v.addWidget(label(f"  v{__version__}, research use only", "caption"))
        return side

    def goto(self, key):
        w = self.pages[key]
        self.stack.setCurrentWidget(w)
        for k, b in self.nav_btns.items():
            b.setChecked(k == key)
            b.setIcon(icon(dict((p[0], p[2]) for p in PAGES)[k], C["haem"] if k == key else C["ink2"], 20))
        if w is self.analysis:
            self.analysis.back_to_live()
        elif w is self.report:
            self.report.refresh()
        elif w is self.cases:
            self.cases.refresh()

    def goto_module(self, key):
        self.analysis.select_module(key)
        self.goto("scope")

    def _build_menu(self):
        self.menuBar().hide()        # navigation lives in the sidebar; keep the shortcuts
        for text, fn, key in (("New case", self.new_case, "Ctrl+N"), ("Quit", self.close, "Ctrl+Q"),
                              ("Guide", self.show_guide, "F1")):
            a = QAction(text, self)
            a.setShortcut(QKeySequence(key))
            a.triggered.connect(fn)
            self.addAction(a)
        self.auto_mag = QAction("Recognise the objective on every Ready field", self, checkable=True)
        self.auto_mag.setChecked(self.ctx.settings.value("auto_mag", False, bool))
        self.auto_mag.toggled.connect(lambda on: self.ctx.settings.setValue("auto_mag", on))

    def camera_menu(self):
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        m.addAction(f"Capture blank for {self.ctx.objective}…", self.capture_blank)
        m.addAction(f"Remove blank for {self.ctx.objective}", self.clear_blank)
        m.addSeparator()
        m.addAction(self.auto_mag)
        m.addSeparator()
        m.addAction("Switch microscope…", self.switch_profile)
        m.addAction("Add a microscope…", self.new_profile)
        m.addSeparator()
        m.addAction("Status of models and thresholds", self.show_status)
        m.addAction("Open the data folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_data_dir()))))
        m.addAction("Open the log folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_dir()))))
        b = self.cambar.more
        m.exec(b.mapToGlobal(b.rect().bottomLeft()))

    # ================================================================== case
    def new_case(self):
        self.goto("cases")
        self.cases.new_case()

    def open_case_id(self, cid):
        try:
            case = self.ctx.store.open(cid, self.ctx.operator or None)
        except Exception as e:  # noqa: BLE001
            error_box(self, "Could not open the case", e)
            return
        self.ctx.set_case(case)

    def _guarded_set_case(self, case):
        cur = self.ctx.case
        if cur is not None and (case is None or case.case_id != cur.case_id):
            unsaved = self.analysis.has_unsaved()
            if unsaved and QMessageBox.question(
                    self, "Discard unsaved results?", f"These results are not saved to case {cur.case_id}: "
                                                      f"{', '.join(unsaved)}. Switch case and discard them?") != QMessageBox.Yes:
                return
        self._orig_set_case(case)

    def _on_case(self, case):
        self.slide.set_case(case)
        self.cases.refresh()
        if case is not None:
            self.ctx.settings.setValue("last_case", case.case_id)
            self.toast.show_text(f"Case {case.case_id} open, patient {case.patient.get('patient_id', '')}", "neutral")
            self.goto("scope")
        self.setWindowTitle(f"PathoAssist — {case.case_id}" if case else "PathoAssist")

    def _update_operator(self):
        self.op_btn.setText(f"  {self.ctx.operator or 'Set your name'}")

    def settings_dialog(self):
        d = SettingsDialog(self.ctx, self)
        if d.exec() == QDialog.Accepted:
            self.ctx.set_operator(d.op.text())
            self.ctx.settings.setValue("lab_name", d.lab.text().strip())
            self._update_operator()
            self.home.refresh()

    def ask_operator(self):
        if self.ctx.operator:
            return
        name, ok = QInputDialog.getText(self, "Welcome to PathoAssist",
                                        "Your name. It is recorded with every change you make to a case.")
        if ok and name.strip():
            self.ctx.set_operator(name)
            self._update_operator()
            self.home.refresh()

    # ================================================================== objectives / profile
    def _fill_objectives(self):
        p = self.ctx.profile
        names = p.objective_names()
        tips = {n: (f"{p.objectives[n].um_per_px:.3f} µm/px at {p.objectives[n].ref_width} px"
                    if p.objectives[n].calibrated else "Not calibrated yet") for n in names}
        marks = [n for n in names if not p.objectives[n].calibrated]
        self.cambar.obj.set_items(names, tips, marks)
        self.cambar.obj.set_current(self.ctx.objective)

    def _on_objective(self, name):
        self.cambar.obj.set_current(name)
        if not self.ctx.um_per_px():
            self.toast.show_text(f"{name} is not calibrated yet. Press Calibrate before measuring.", "warn")

    def _objective_hotkey(self, i):
        names = self.ctx.profile.objective_names()
        if i < len(names):
            self.ctx.set_objective(names[i])

    def detect_objective(self, interactive=True):
        f = self.ctx.last_frame
        if f is None:
            if interactive:
                error_box(self, "No live image", "Start the camera first.")
            return
        scales = self.ctx.profile.calibrated_scales(f.shape[1])
        if len(scales) < 2:
            if interactive:
                error_box(self, "Calibrate first", "Auto needs at least two calibrated objectives to choose between.")
            return
        rgb = f.copy()
        self._auto_mag_busy = True

        def done(r):
            self._auto_mag_busy = False
            name, um, conf = r
            if name is None:
                if interactive:
                    self.toast.show_text("No tissue in view, so the objective could not be recognised.", "warn")
                return
            unsure = name.startswith("20") or conf < 0.6
            if unsure and not interactive:
                self.statusBar().showMessage(f"Objective unclear (looks like {name}). Press 1-4 to choose.", 5000)
                return
            if unsure and QMessageBox.question(self, "Check the objective",
                                               f"The image looks like {name} ({um:.2f} µm/px, {conf:.0%} agreement). "
                                               f"20x and low-agreement guesses are often wrong. Switch to {name}?") != QMessageBox.Yes:
                return
            if name != self.ctx.objective:
                self.ctx.set_objective(name)
            self.toast.show_text(f"Objective {name}", "neutral")

        def fail(e):
            self._auto_mag_busy = False
            if interactive:
                error_box(self, "Objective not recognised", e)

        self.ctx.jobs.submit(lambda: self.ctx.hub.magnification(scales).estimate(rgb), done, fail)

    def calibrate(self):
        CalibrationDialog(self.ctx, self).exec()
        self._fill_objectives()
        self.analysis._update_scale()
        self.home.refresh()

    def capture_blank(self):
        if self.ctx.last_frame is None:
            error_box(self, "No live image", "Start the camera first, then move to empty glass.")
            return
        if QMessageBox.question(self, f"Capture blank for {self.ctx.objective}",
                                "Move to an empty area of the slide with no tissue. Keep the light and focus as for "
                                "work. Capture now?") != QMessageBox.Yes:
            return
        raw = self.ctx.last_frame.copy()
        if self.cambar.flat.isChecked() and self.ctx.flat.active and self.ctx.flat.shape == raw.shape:
            raw = (raw.astype("float32") / self.ctx.flat.gain).clip(0, 255).astype("uint8")
        try:
            self.ctx.flat.set_blank(raw)
        except ValueError as e:
            error_box(self, "Blank not usable", e)
            return
        self.ctx.profile.save_blank(self.ctx.objective, raw)
        self.ctx.profile.save()
        self.toast.show_text(f"Blank saved for {self.ctx.objective}. Flat-field correction is on.", "good")
        self.home.refresh()

    def clear_blank(self):
        self.ctx.profile.clear_blank(self.ctx.objective)
        self.ctx.profile.save()
        self.ctx.flat.clear()
        self.toast.show_text(f"Blank removed for {self.ctx.objective}", "neutral")

    def _flat_toggled(self, on):
        if self._capture is not None:
            self._capture.flat_on = on

    def switch_profile(self):
        ps = MicroscopeProfile.list(profiles_dir())
        names = [p.name for p in ps]
        name, ok = QInputDialog.getItem(self, "Switch microscope", "Microscope:", names,
                                        names.index(self.ctx.profile.name) if self.ctx.profile.name in names else 0, False)
        if ok:
            self.ctx.set_profile(ps[names.index(name)])
            self.toast.show_text(f"Using {name}", "neutral")

    def new_profile(self):
        name, ok = QInputDialog.getText(self, "Add a microscope", "Name, for example 'Olympus CX43, room 2':")
        if not ok or not name.strip():
            return
        p = MicroscopeProfile.new(name.strip(), profiles_dir())
        if p.path.exists():
            error_box(self, "Name in use", "A microscope with this name already exists.")
            return
        p.camera = self.cambar.cam.currentText()
        p.save()
        self.ctx.set_profile(p)
        self.toast.show_text(f"{p.name} added. Calibrate its objectives next.", "neutral")

    # ================================================================== camera
    def refresh_cameras(self):
        if self._capture is not None:
            return
        self.cambar.cam.clear()
        cams = list_cameras()
        for c in cams:
            self.cambar.cam.addItem(c.name, c)
        if not cams:
            self.cambar.cam.addItem("No camera found")
        self.cambar.start.setEnabled(bool(cams))

    def start_from_home(self):
        self.goto("scope")
        if self._capture is None:
            self.toggle_camera()

    def toggle_camera(self):
        if self._capture is not None:
            self.stop_camera()
            return
        info = self.cambar.cam.currentData()
        if info is None:
            error_box(self, "No camera", "Plug the microscope camera into a USB port and press the refresh button.")
            return
        w, h = self.cambar.res.currentData()
        self.ctx.settings.setValue("resolution_index", self.cambar.res.currentIndex())
        self._capture = CaptureThread(info, w, h, auto=not self.cambar.lock.isChecked(), flat=self.ctx.flat)
        self._capture.flat_on = self.cambar.flat.isChecked()
        self._capture.frame.connect(self._on_frame)
        self._capture.failed.connect(self._on_camera_failed)
        self._capture.opened.connect(lambda gw, gh: self.statusBar().showMessage(f"{info.name}: {gw} × {gh}", 5000))
        self._capture.start()
        self.cambar.set_running(True)

    def stop_camera(self):
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self.ctx.live = False
        self.cambar.set_running(False)
        self.fps_lbl.setText("")
        self.analysis.camera_stopped()

    @Slot(str)
    def _on_camera_failed(self, msg):
        self.stop_camera()
        error_box(self, "Camera stopped", f"{msg}\n\nCheck the USB cable, then press Start again.")

    @Slot(object, object)
    def _on_frame(self, rgb, st):
        new_size = self.ctx.last_frame is None or self.ctx.last_frame.shape != rgb.shape
        self.ctx.last_frame, self.ctx.last_state, self.ctx.live = rgb, st, True
        if new_size:
            self.analysis._update_scale()
        self.ctx.frame.emit(rgb, st)
        now = time.perf_counter()
        if now - self._last_status > 0.5 and self._capture is not None:
            self._last_status = now
            h, w = rgb.shape[:2]
            ff = "flat-field on" if self.ctx.flat.active and self.cambar.flat.isChecked() else "no flat-field"
            self.fps_lbl.setText(f"{w} × {h}   {self._capture.fps:.0f} fps   focus {st.focus:.0f}   {ff}")
        if st.dwell and self.auto_mag.isChecked() and not self._auto_mag_busy and self.ctx.jobs.pending == 0:
            self.detect_objective(interactive=False)

    def capture_frame(self):
        f = self.ctx.last_frame
        if f is None or not self.ctx.live:
            error_box(self, "No live image", "Start the camera first.")
            return
        stain = self.analysis._module_cls().stain
        meta = dict(self.ctx.frame_meta(), um_per_px=self.ctx.um_per_px(f.shape[1]), stain=stain, kind="capture")
        logger = FrameLogger(__version__)
        if self.ctx.case is not None:
            meta["case_id"] = self.ctx.case.case_id
            logger.save(FrameLogger.case_frames_dir(self.ctx.case, stain), f, meta)
            self.toast.show_text(f"Frame saved to case {self.ctx.case.case_id} ({stain})", "good")
        else:
            p = logger.save(captures_dir() / time.strftime("%Y-%m-%d"), f, meta)
            self.toast.show_text(f"Frame saved to {p.parent.name} (no case open)", "neutral")

    # ================================================================== help
    def show_guide(self):
        d = QDialog(self)
        d.setWindowTitle("How to use PathoAssist")
        d.resize(760, 700)
        v = QVBoxLayout(d)
        v.setContentsMargins(16, 16, 16, 16)
        t = QTextBrowser()
        t.setHtml(GUIDE)
        v.addWidget(t)
        d.exec()

    def show_status(self):
        h = self.ctx.hub
        cal = h.calibration_status()
        lines = [f"<b>Models loaded:</b> {', '.join(h.loaded()) or 'none yet (each loads on first use)'}",
                 f"<b>Nuclei model:</b> {h.nuclei_model}", "<b>Thresholds:</b>"]
        lines += [f"&nbsp;&nbsp;{k}: {'<b>local calibration</b>' if v else 'public-data defaults'}" for k, v in cal.items()]
        p = self.ctx.profile
        lines.append(f"<b>Microscope:</b> {p.name}, calibrated: "
                     + (", ".join(n for n in p.objective_names() if p.objectives[n].calibrated) or "none"))
        lines.append(f"<b>Data folder:</b> {user_data_dir()}")
        QMessageBox.information(self, "Models and thresholds", "<br>".join(lines))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.toast.isVisible():
            self.toast.move((self.width() - self.toast.width()) // 2, self.height() - self.toast.height() - 48)

    # ================================================================== close
    def closeEvent(self, e):
        unsaved = self.analysis.has_unsaved()
        if unsaved and QMessageBox.question(self, "Quit without saving?",
                                            f"These results are not saved to the case: {', '.join(unsaved)}. Quit anyway?"
                                            ) != QMessageBox.Yes:
            e.ignore()
            return
        self.annotate.shutdown()
        if self.ctx.case is not None:
            try:
                self.ctx.case.save()
            except Exception:  # noqa: BLE001
                logging.exception("saving case on exit")
        self.stop_camera()
        self.ctx.jobs.stop()
        super().closeEvent(e)
