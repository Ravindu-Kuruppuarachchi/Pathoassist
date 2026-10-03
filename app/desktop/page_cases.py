"""Cases page: list / search / create / open cases (patient ID kept with each case) and browse a case's frames."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from core.grading.case import PATIENT_FIELDS, SPECIMENS

from .casestore import STAINS, FrameLogger, read_rgb, valid_case_id
from .modules import resize_w
from .theme import C, TONE, Chip, PageHeader, button, icon, label, page_layout, panel
from .ui_common import ImageCanvas, error_box, to_pixmap


class CaseDialog(QDialog):
    """New case, or edit the details of an open (unsigned) case."""

    def __init__(self, parent=None, case=None):
        super().__init__(parent)
        self.setWindowTitle("Edit case details" if case else "New case")
        self.setMinimumWidth(520)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.addWidget(label("Edit case details" if case else "New case", "h1"))
        outer.addWidget(label("Fields marked * are required. Patient details stay in this case and its report.", "caption", wrap=True))
        f = QFormLayout()
        f.setSpacing(10)
        f.setLabelAlignment(Qt.AlignRight)
        outer.addLayout(f)
        self.case_id = QLineEdit(case.case_id if case else "")
        self.case_id.setPlaceholderText("accession number, e.g. B26-01234")
        self.case_id.setEnabled(case is None)
        f.addRow("Case ID *", self.case_id)
        self.fields = {}
        for k, text in PATIENT_FIELDS:
            w = QPlainTextEdit() if k == "clinical" else QLineEdit()
            if k == "clinical":
                w.setFixedHeight(60)
            v = case.patient.get(k, "") if case else ""
            w.setPlainText(v) if k == "clinical" else w.setText(v)
            f.addRow(text + (" *" if k == "patient_id" else ""), w)
            self.fields[k] = w
        self.specimen = QComboBox()
        self.specimen.addItems(SPECIMENS)
        if case:
            self.specimen.setCurrentText(case.specimen)
        f.addRow("Specimen", self.specimen)
        note = label("Saved frames and annotations carry the case ID only, never these patient details.", "caption", wrap=True)
        outer.addWidget(note)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(button("Cancel", slot=self.reject))
        ok = button("Save changes" if case else "Create case", "check" if case else "plus", "primary", slot=self._ok)
        ok.setDefault(True)
        row.addWidget(ok)
        outer.addLayout(row)

    def values(self):
        p = {k: (w.toPlainText() if isinstance(w, QPlainTextEdit) else w.text()).strip() for k, w in self.fields.items()}
        return self.case_id.text().strip(), p, self.specimen.currentText()

    def _ok(self):
        cid, p, _ = self.values()
        if not valid_case_id(cid):
            error_box(self, "Check the case ID", "Use 1 to 64 letters, digits, '.', '_' or '-'. It becomes a folder name.")
            return
        if not p["patient_id"]:
            error_box(self, "Patient ID needed", "Enter the patient ID; every result is filed under it.")
            return
        self.accept()


class ImageDialog(QDialog):
    def __init__(self, parent, rgb, title):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 750)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        c = ImageCanvas()
        v.addWidget(c)
        c.set_image(rgb, fit=True)


class CasesPage(QWidget):
    COLS = ["Case", "Patient ID", "Patient name", "Specimen", "Created", "Results", "Status"]

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        v = page_layout(self)
        head = PageHeader("Cases", "Every result, frame and report belongs to a case and its patient.")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search case, patient ID or name")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(280)
        self.search.textChanged.connect(self.refresh)
        head.add(self.search)
        head.add(button("New case", "plus", "primary", "Ctrl+N", self.new_case))
        v.addWidget(head)

        sp = QSplitter(Qt.Vertical)
        sp.setHandleWidth(12)
        p, lay = panel(margins=(0, 0, 0, 0), spacing=0)
        self.stack = QStackedWidget()
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setFrameShape(QFrame.NoFrame)
        self.table.setStyleSheet("QTableWidget { border: none; border-radius: 10px; }")
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(44)
        self.table.doubleClicked.connect(self.open_selected)
        self.table.itemSelectionChanged.connect(self._selection)
        self.stack.addWidget(self.table)
        empty = QWidget()
        ev = QVBoxLayout(empty)
        ev.addStretch(1)
        for w in (label("No cases yet", "h2"), label("Create a case with the accession number and patient ID, then go to "
                                                      "the microscope.", "muted")):
            w.setAlignment(Qt.AlignCenter)
            ev.addWidget(w)
        b = button("New case", "plus", "primary", slot=self.new_case)
        ev.addWidget(b, 0, Qt.AlignCenter)
        ev.addStretch(1)
        self.stack.addWidget(empty)
        lay.addWidget(self.stack)
        sp.addWidget(p)

        p, lay = panel()
        h = QHBoxLayout()
        self.frames_title = label("Frames", "h2")
        h.addWidget(self.frames_title)
        h.addStretch(1)
        self.stain = QComboBox()
        self.stain.addItems(["All stains"] + STAINS)
        self.stain.currentIndexChanged.connect(self.show_frames)
        h.addWidget(self.stain)
        self.open_btn = button("Open case", "folder", "primary", slot=self.open_selected)
        h.addWidget(self.open_btn)
        h.addWidget(button("", "files", "ghost", "Show the case folder", self.show_folder))
        lay.addLayout(h)
        self.frames = QListWidget()
        self.frames.setViewMode(QListWidget.IconMode)
        self.frames.setIconSize(QSize(170, 128))
        self.frames.setResizeMode(QListWidget.Adjust)
        self.frames.setMovement(QListWidget.Static)
        self.frames.setSpacing(6)
        self.frames.setStyleSheet("QListWidget { border: none; }")
        self.frames.itemDoubleClicked.connect(self.view_frame)
        lay.addWidget(self.frames)
        sp.addWidget(p)
        sp.setSizes([460, 280])
        v.addWidget(sp, 1)
        self.refresh()

    def _selected_id(self):
        r = self.table.currentRow()
        if r < 0 or self.table.item(r, 0) is None:
            return None
        return self.table.item(r, 0).text()

    def _selection(self):
        cid = self._selected_id()
        self.frames_title.setText(f"Frames of {cid}" if cid else "Frames")
        self.open_btn.setEnabled(bool(cid))
        self.show_frames()

    def refresh(self):
        q = self.search.text().strip().lower()
        all_cases = self.ctx.store.list()
        rows = [c for c in all_cases
                if not q or q in c.case_id.lower() or q in c.patient_id.lower() or q in c.name.lower()]
        self.stack.setCurrentIndex(0 if all_cases else 1)
        self.table.setRowCount(len(rows))
        cur = self.ctx.case.case_id if self.ctx.case else None
        for i, c in enumerate(rows):
            vals = [c.case_id, c.patient_id, c.name, c.specimen, c.created.replace("T", "  ")[:17],
                    ", ".join(c.results) or "none yet"]
            for j, t in enumerate(vals):
                it = QTableWidgetItem(t)
                if j == 0:
                    f = it.font()
                    f.setWeight(QFont.DemiBold)
                    it.setFont(f)
                if j == 5 and not c.results:
                    it.setForeground(QColor(C["ink3"]))
                self.table.setItem(i, j, it)
            state, tone = (("Signed", "good") if c.signed_by else
                           (("Open now", "accent") if c.case_id == cur else ("Draft", "warn")))
            it = QTableWidgetItem(icon("dot", TONE[tone][0], 12), state)
            it.setForeground(QColor(TONE[tone][0]))
            if c.signed_by:
                it.setToolTip(f"Signed by {c.signed_by}")
            self.table.setItem(i, 6, it)
        self.open_btn.setEnabled(self._selected_id() is not None)

    def new_case(self):
        d = CaseDialog(self)
        if d.exec() != QDialog.Accepted:
            return
        cid, patient, specimen = d.values()
        try:
            case = self.ctx.store.create(cid, patient, specimen, self.ctx.operator or None)
        except (ValueError, FileExistsError) as e:
            error_box(self, "Could not create the case", e)
            return
        self.ctx.set_case(case)
        self.refresh()

    def open_selected(self):
        cid = self._selected_id()
        if not cid:
            return
        try:
            case = self.ctx.store.open(cid, self.ctx.operator or None)
        except Exception as e:  # noqa: BLE001
            error_box(self, "Could not open the case", e)
            return
        self.ctx.set_case(case)

    def show_folder(self):
        cid = self._selected_id()
        p = self.ctx.store.case_dir(cid) if cid else self.ctx.store.root
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def show_frames(self):
        self.frames.clear()
        cid = self._selected_id()
        if not cid:
            return
        st = None if self.stain.currentIndex() == 0 else self.stain.currentText()
        frames = FrameLogger.list_frames(self.ctx.store.case_dir(cid), st)[-300:]
        if not frames:
            it = QListWidgetItem("No frames saved for this case yet.")
            it.setFlags(Qt.NoItemFlags)
            self.frames.addItem(it)
        for png, meta in frames:
            rgb = read_rgb(png)
            if rgb is None:
                continue
            it = QListWidgetItem(QIcon(to_pixmap(resize_w(rgb, 170))),
                                 f"{meta.get('stain', '')} {meta.get('objective', '')}  {png.stem[9:11]}:{png.stem[11:13]}")
            it.setData(Qt.UserRole, str(png))
            it.setToolTip(f"{png.name}\n{meta.get('module', '')}, {meta.get('um_per_px', '?')} µm/px")
            self.frames.addItem(it)

    def view_frame(self, it):
        if not it.data(Qt.UserRole):
            return
        p = Path(it.data(Qt.UserRole))
        rgb = read_rgb(p)
        if rgb is not None:
            ImageDialog(self, rgb, p.name).exec()
