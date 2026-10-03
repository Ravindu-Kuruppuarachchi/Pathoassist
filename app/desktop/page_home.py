"""Home: what to do next. Set-up steps for the microscope, quick actions, recent cases, local-data status."""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from .annotation import TASKS, AnnotationStore
from .paths import annotations_dir
from .theme import Chip, button, hline, label, page_layout, panel, section
from .widgets import StepRow


def greeting(name):
    h = time.localtime().tm_hour
    part = "Good morning" if h < 12 else ("Good afternoon" if h < 18 else "Good evening")
    return f"{part}, {name}" if name else part


class HomePage(QWidget):
    go = Signal(str)                  # page key to show
    open_case = Signal(str)           # case ID

    def __init__(self, ctx, actions: dict, parent=None):
        """actions: callables for 'new_case', 'start_camera', 'calibrate', 'blank', 'annotate'."""
        super().__init__(parent)
        self.ctx, self.actions = ctx, actions
        v = page_layout(self)
        self.hello = label("", "h1")
        v.addWidget(self.hello)
        self.today = label("", "sub")
        v.addWidget(self.today)
        v.addSpacing(6)

        cols = QHBoxLayout()
        cols.setSpacing(16)
        v.addLayout(cols, 1)
        left = QVBoxLayout()
        left.setSpacing(16)
        right = QVBoxLayout()
        right.setSpacing(16)
        cols.addLayout(left, 5)
        cols.addLayout(right, 6)

        # ---- set-up steps (a real sequence)
        p, lay = panel()
        lay.addWidget(section("Get the microscope ready", "Once per microscope; the camera step every session."))
        self.steps = {
            "camera": StepRow(1, "Connect the camera", "Start camera", actions["start_camera"]),
            "calibrate": StepRow(2, "Calibrate the objectives", "Calibrate", actions["calibrate"]),
            "blank": StepRow(3, "Capture a blank field", "Capture blank", actions["blank"]),
            "case": StepRow(4, "Open a case", "New case", actions["new_case"]),
        }
        for i, st in enumerate(self.steps.values()):
            if i:
                lay.addWidget(hline())
            lay.addWidget(st)
        left.addWidget(p)

        # ---- local data
        p, lay = panel()
        h = QHBoxLayout()
        h.addWidget(section("Your lab's training data", "Complete annotated images per task. Each one helps the "
                                                        "models fit your slides."), 1)
        lay.addLayout(h)
        self.data_lbl = label("", "muted", wrap=True)
        lay.addWidget(self.data_lbl)
        self.calib_chip = Chip("")
        lay.addWidget(self.calib_chip, 0, Qt.AlignLeft)
        lay.addWidget(button("Annotate data", "annotate", slot=lambda: self.go.emit("annotate")), 0, Qt.AlignLeft)
        left.addWidget(p)
        left.addStretch(1)

        # ---- quick actions
        p, lay = panel()
        lay.addWidget(section("Start working"))
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(button("New case", "plus", "primary", slot=actions["new_case"]))
        self.resume_btn = button("Continue last case", "back", slot=self._resume)
        row.addWidget(self.resume_btn)
        row.addWidget(button("Microscope", "scope", slot=lambda: self.go.emit("scope")))
        row.addStretch(1)
        lay.addLayout(row)
        right.addWidget(p)

        # ---- recent cases
        p, lay = panel()
        h = QHBoxLayout()
        h.addWidget(section("Recent cases"), 1)
        h.addWidget(button("All cases", "cases", "ghost", slot=lambda: self.go.emit("cases")))
        lay.addLayout(h)
        self.recent = QListWidget()
        self.recent.setStyleSheet("QListWidget { border: none; } QListWidget::item { padding: 0; }")
        self.recent.setSelectionMode(QListWidget.NoSelection)
        lay.addWidget(self.recent, 1)
        right.addWidget(p, 1)

        ctx.case_changed.connect(lambda _: self.refresh())
        ctx.case_updated.connect(self.refresh)
        ctx.profile_changed.connect(self.refresh)

    def _resume(self):
        cid = self.ctx.settings.value("last_case", "", str)
        if cid:
            self.open_case.emit(cid)

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()

    def refresh(self):
        ctx = self.ctx
        self.hello.setText(greeting(ctx.operator))
        self.today.setText(time.strftime("%A %d %B %Y") + ".  Research use only: every result needs your review.")
        live = ctx.live
        self.steps["camera"].set_done(live, "Streaming." if live else "Plug the microscope camera into USB, then start it.")
        prof = ctx.profile
        cal = [n for n in prof.objective_names() if prof.objectives[n].calibrated]
        self.steps["calibrate"].set_done(bool(cal), f"{prof.name}: {', '.join(cal)} calibrated." if cal else
                                         "Photograph a stage micrometer at each objective and click two ticks.")
        blanks = [n for n in prof.objective_names() if prof.objectives[n].blank_file]
        self.steps["blank"].set_done(bool(blanks), f"Blank saved for {', '.join(blanks)}." if blanks else
                                     "Move to empty glass; this removes uneven light and colour cast.")
        c = ctx.case
        self.steps["case"].set_done(c is not None, f"{c.case_id} is open (patient {c.patient.get('patient_id', '')})."
                                    if c else "Enter the accession number and patient ID.")
        self.resume_btn.setEnabled(bool(ctx.settings.value("last_case", "", str)) and ctx.store.exists(
            ctx.settings.value("last_case", "", str)))

        self.recent.clear()
        cases = ctx.store.list()[:8]
        if not cases:
            it = QListWidgetItem("No cases yet. Create the first one with New case.")
            it.setFlags(Qt.NoItemFlags)
            self.recent.addItem(it)
        for s in cases:
            it = QListWidgetItem()
            row = CaseRow(s, lambda cid=s.case_id: self.open_case.emit(cid))
            it.setSizeHint(row.sizeHint())
            self.recent.addItem(it)
            self.recent.setItemWidget(it, row)

        st = AnnotationStore(annotations_dir())
        parts = []
        for t in TASKS:
            d = st.task_dir(t.key)
            n = sum(1 for p in d.glob("*/annotation.json") if '"status": "complete"' in p.read_text())
            if n:
                parts.append(f"{t.title}: {n}")
        self.data_lbl.setText(", ".join(parts) if parts else "No complete annotations yet. Use Annotate to label fields from your slides.")
        cal = [k for k, v in ctx.hub.calibration_status().items() if v]
        self.calib_chip.set(f"Local thresholds: {', '.join(cal)}" if cal else "Thresholds: public-data defaults",
                            "good" if cal else "warn")


class CaseRow(QWidget):
    def __init__(self, s, on_open):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 8, 4, 8)
        h.setSpacing(12)
        v = QVBoxLayout()
        v.setSpacing(1)
        t = label(f"{s.case_id}")
        t.setStyleSheet("font-weight:600; font-size:14px;")
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(t)
        top.addWidget(label(f"Patient {s.patient_id}" + (f", {s.name}" if s.name else ""), "muted"))
        top.addStretch(1)
        v.addLayout(top)
        v.addWidget(label(", ".join(s.results) if s.results else "No results yet", "caption"))
        h.addLayout(v, 1)
        h.addWidget(Chip("Signed" if s.signed_by else "Draft", "good" if s.signed_by else "warn"))
        h.addWidget(button("Open", kind="ghost", slot=on_open))

    def sizeHint(self):
        from PySide6.QtCore import QSize
        return QSize(400, 58)
