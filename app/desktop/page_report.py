"""Report page (Feature 9): the open case at a glance - patient, Nottingham grade (components from the modules,
changeable by the pathologist with a reason), biomarkers, other findings, snapshots, audit log, draft PDF and
signing (which locks the case)."""
from __future__ import annotations

import html
import time

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog,
                               QListWidget, QListWidgetItem, QMessageBox, QScrollArea, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from core.grading.case import PATIENT_FIELDS
from core.grading.nottingham import COMPONENTS, LABELS

from .casestore import read_rgb
from .modules import resize_w
from .page_cases import CaseDialog, ImageDialog
from .theme import C, Chip, PageHeader, button, hline, label, page_layout, panel, section
from .ui_common import error_box, to_pixmap
from .widgets import Segmented

SOURCE = {"auto": "Assistant, accepted", "edited": "Changed by pathologist", "manual": "Entered by pathologist", "missing": ""}
MODULE_FOR = {"tubules": "Tubules", "pleomorphism": "Pleomorphism", "mitoses": "Mitoses"}


def _e(x):
    return html.escape(str(x))


class AuditDialog(QDialog):
    def __init__(self, parent, case):
        super().__init__(parent)
        self.setWindowTitle(f"Audit log of {case.case_id}")
        self.resize(980, 620)
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 18, 20, 18)
        ok, msg = case.audit.verify()
        h = QHBoxLayout()
        h.addWidget(label("Audit log", "h1"))
        h.addStretch(1)
        h.addWidget(Chip("Integrity verified" if ok else f"Integrity failed: {msg}", "good" if ok else "bad"))
        v.addLayout(h)
        v.addWidget(label(f"{len(case.audit.entries())} entries. Fingerprint {case.audit.last_hash[:16]} is printed on "
                          "every report so a report can be matched to this log.", "caption", wrap=True))
        t = QTableWidget(0, 4)
        t.setHorizontalHeaderLabels(["Time", "User", "Event", "Details"])
        t.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        t.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        t.verticalHeader().setVisible(False)
        t.setShowGrid(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        es = case.audit.entries()
        t.setRowCount(len(es))
        for i, e in enumerate(es):
            vals = [e["time"].replace("T", "  ")[:20], e["user"], e["event"].replace("_", " "),
                    ", ".join(f"{k} {v}" for k, v in e.get("details", {}).items())]
            for j, val in enumerate(vals):
                t.setItem(i, j, QTableWidgetItem(str(val)))
        t.resizeColumnsToContents()
        v.addWidget(t)


class GradeBadge(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.big = label("-")
        self.big.setAlignment(Qt.AlignCenter)
        self.small = label("", "caption")
        self.small.setAlignment(Qt.AlignCenter)
        v.addWidget(self.big)
        v.addWidget(self.small)
        self.setFixedWidth(120)

    def set(self, grade, total, assessed):
        if grade:
            col = {1: C["ok"], 2: C["dab"], 3: C["bad"]}[grade]
            self.big.setText(f"Grade {grade}")
            self.big.setStyleSheet(f"font-size:24px; font-weight:600; color:{col};")
            self.small.setText(f"total {total} of 9")
        else:
            self.big.setText("Not graded" if not assessed else "Incomplete")
            self.big.setStyleSheet(f"font-size:16px; font-weight:600; color:{C['ink3']};")
            self.small.setText("")


class ComponentRow(QWidget):
    def __init__(self, key, on_set, on_run):
        super().__init__()
        self.key = key
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 6, 0, 6)
        h.setSpacing(12)
        v = QVBoxLayout()
        v.setSpacing(1)
        t = label(LABELS[key])
        t.setStyleSheet("font-weight:600; font-size:14px;")
        v.addWidget(t)
        self.value = label("", "caption", wrap=True)
        v.addWidget(self.value)
        h.addLayout(v, 1)
        self.auto = Chip("")
        h.addWidget(self.auto)
        self.seg = Segmented(["1", "2", "3"])
        self.seg.changed.connect(lambda s: on_set(key, int(s)))
        h.addWidget(self.seg)
        self.source = label("", "caption")
        self.source.setFixedWidth(150)
        h.addWidget(self.source)
        self.run = button("Measure", "analyse", "ghost", f"Open the {MODULE_FOR[key]} analysis", lambda: on_run(key))
        h.addWidget(self.run)

    def set(self, comp, locked):
        if comp.value:
            self.value.setText(comp.value)
        elif comp.score:
            self.value.setText("Scored by the pathologist.")
        else:
            self.value.setText("Not scored yet. Measure it, or choose the score.")
        self.auto.set(f"Assistant {comp.auto_score}" if comp.auto_score is not None else "", "accent")
        self.seg.group.setExclusive(False)          # an unscored component shows no button pressed
        for b in self.seg.group.buttons():
            b.setChecked(b.text() == str(comp.score))
        self.seg.group.setExclusive(True)
        self.seg.setEnabled(not locked)
        self.run.setVisible(not locked)
        src = SOURCE[comp.source]
        if comp.source in ("auto", "edited") and not comp.validated:
            src += " (method not validated locally)"
        self.source.setText(src)
        self.source.setToolTip(src)


class MarkerTile(QWidget):
    """One biomarker: name, result, details, notes count, and a way to measure it."""

    def __init__(self, name, key, on_run):
        super().__init__()
        self.name = name
        f, lay = panel(margins=(14, 12, 14, 12), spacing=4)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(f)
        h = QHBoxLayout()
        h.addWidget(label(name, "h3"))
        h.addStretch(1)
        self.notes = Chip("")
        h.addWidget(self.notes)
        lay.addLayout(h)
        self.value = label("-")
        lay.addWidget(self.value)
        self.detail = label("", "caption", wrap=True)
        lay.addWidget(self.detail)
        lay.addStretch(1)
        self.run = button(f"Measure {name}", "analyse", "ghost", slot=lambda: on_run(key))
        lay.addWidget(self.run, 0, Qt.AlignLeft)

    def set(self, value, detail, tone, n_notes, notes_tip, assessed, locked):
        col = {"good": C["ok"], "warn": C["dab"], "bad": C["bad"]}.get(tone, C["ink"] if assessed else C["ink3"])
        self.value.setText(value)
        self.value.setStyleSheet(f"font-size:{22 if assessed else 15}px; font-weight:600; color:{col};")
        self.detail.setText(detail)
        self.notes.set(f"{n_notes} note{'s' * (n_notes > 1)}" if n_notes else "", "warn")
        self.notes.setToolTip(notes_tip)
        self.run.setVisible(not locked)
        self.run.setText("Measure again" if assessed else f"Measure {self.name}")


class ReportPage(QWidget):
    def __init__(self, ctx, goto_module=None, toast=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.goto_module = goto_module or (lambda k: None)
        self.toast = toast or (lambda *a, **k: None)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        v = page_layout(body)

        self.head = PageHeader("Report", "")
        self.audit_btn = self.head.add(button("Audit log", "lock", "ghost", slot=self.show_audit))
        self.folder_btn = self.head.add(button("Reports folder", "folder", "ghost", slot=self.open_reports))
        self.draft_btn = self.head.add(button("Draft PDF", "report", slot=lambda: self.write_pdf(final=False)))
        self.sign_btn = self.head.add(button("Sign report", "sign", "primary", "Sign and lock the case", self.sign))
        v.addWidget(self.head)
        self.empty = label("Open a case to see its report. Cases are on the Cases page.", "muted")
        v.addWidget(self.empty)
        self.signed_bar, sl = panel(margins=(14, 10, 14, 10), horizontal=True)
        self.signed_bar.setStyleSheet(f"QFrame {{ background:{C['ok_soft']}; border:1px solid #B9E0CC; border-radius:10px; }}")
        self.signed_lbl = label("")
        self.signed_lbl.setStyleSheet(f"color:{C['ok']}; font-weight:600; background: transparent; border: none;")
        sl.addWidget(self.signed_lbl)
        v.addWidget(self.signed_bar)

        self.content = QWidget()
        cv = QVBoxLayout(self.content)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(14)
        v.addWidget(self.content)
        v.addStretch(1)

        top = QHBoxLayout()
        top.setSpacing(14)
        p, lay = panel()
        h = QHBoxLayout()
        h.addWidget(section("Patient and request"))
        h.addStretch(1)
        self.edit_btn = button("Edit", "annotate", "ghost", slot=self.edit_details)
        h.addWidget(self.edit_btn)
        lay.addLayout(h)
        self.patient_grid = QGridLayout()
        self.patient_grid.setHorizontalSpacing(16)
        self.patient_grid.setVerticalSpacing(6)
        lay.addLayout(self.patient_grid)
        lay.addStretch(1)
        p.setFixedWidth(360)
        top.addWidget(p)

        p, lay = panel()
        h = QHBoxLayout()
        h.addWidget(section("Nottingham grade", "Each component comes from its analysis, or from you. "
                                                "Changing a score asks for a reason."), 1)
        self.badge = GradeBadge()
        h.addWidget(self.badge)
        lay.addLayout(h)
        self.rows = {}
        for i, k in enumerate(COMPONENTS):
            lay.addWidget(hline())
            r = ComponentRow(k, self.set_component, self.goto_module)
            self.rows[k] = r
            lay.addWidget(r)
        self.grade_note = label("", "caption", wrap=True)
        lay.addWidget(self.grade_note)
        top.addWidget(p, 1)
        cv.addLayout(top)

        cv.addWidget(section("Biomarkers"))
        tiles = QHBoxLayout()
        tiles.setSpacing(14)
        self.tiles = {}
        for name, key in (("ER", "ER"), ("PR", "PR"), ("HER2", "HER2"), ("Ki-67", "Ki67")):
            t = MarkerTile(name, key, self.goto_module)
            self.tiles[key] = t
            tiles.addWidget(t)
        cv.addLayout(tiles)

        bottom = QHBoxLayout()
        bottom.setSpacing(14)
        p, lay = panel()
        lay.addWidget(section("Other findings"))
        self.extras = label("", "muted", wrap=True)
        lay.addWidget(self.extras)
        lay.addStretch(1)
        p.setFixedWidth(360)
        bottom.addWidget(p)
        p, lay = panel()
        lay.addWidget(section("Images in the report", "Added when an analysis is saved. Double-click to enlarge."))
        self.snaps = QListWidget()
        self.snaps.setViewMode(QListWidget.IconMode)
        self.snaps.setIconSize(QSize(170, 128))
        self.snaps.setFixedHeight(180)
        self.snaps.setFlow(QListWidget.LeftToRight)
        self.snaps.setMovement(QListWidget.Static)
        self.snaps.setStyleSheet("QListWidget { border: none; }")
        self.snaps.itemDoubleClicked.connect(self._view_snap)
        lay.addWidget(self.snaps)
        bottom.addWidget(p, 1)
        cv.addLayout(bottom)

        ctx.case_changed.connect(lambda _: self.refresh())
        ctx.case_updated.connect(self.refresh)
        self.refresh()

    # ------------------------------------------------------------------ view
    def refresh(self):
        c = self.ctx.case
        has = c is not None
        self.content.setVisible(has)
        self.empty.setVisible(not has)
        for b in (self.audit_btn, self.folder_btn, self.draft_btn, self.sign_btn):
            b.setVisible(has)
        if not has:
            self.head.title.setText("Report")
            self.head.subtitle.setText("")
            self.signed_bar.setVisible(False)
            return
        locked = bool(c.signed_by)
        self.head.title.setText(f"Case {c.case_id}")
        self.head.subtitle.setText(f"Patient {c.patient.get('patient_id', '')}"
                                   + (f", {c.patient['name']}" if c.patient.get("name") else "") + f".  {c.specimen.capitalize()}.")
        self.signed_bar.setVisible(locked)
        self.signed_lbl.setText(f"Signed by {c.signed_by} on {(c.signed_at or '').replace('T', ' at ')[:19]}. "
                                "The case is locked." if locked else "")
        self.sign_btn.setVisible(not locked)
        self.edit_btn.setVisible(not locked)

        while self.patient_grid.count():
            w = self.patient_grid.takeAt(0).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        for i, (k, text) in enumerate(PATIENT_FIELDS):
            self.patient_grid.addWidget(label(text, "caption"), i, 0, Qt.AlignTop)
            val = label(c.patient.get(k, "") or "Not given", wrap=True)
            if not c.patient.get(k):
                val.setProperty("role", "caption")
            self.patient_grid.addWidget(val, i, 1)

        g = c.grade
        self.badge.set(g.grade, g.total, c.grade_assessed)
        for k, r in self.rows.items():
            r.set(g.c[k], locked)
        if g.complete:
            self.grade_note.setText("Core biopsy: the grade may differ on the excision specimen." if c.specimen == "core biopsy" else "")
        elif c.grade_assessed:
            miss = [LABELS[k].lower() for k in COMPONENTS if g.c[k].score is None]
            self.grade_note.setText(f"Still needed: {', '.join(miss)}.")
        else:
            self.grade_note.setText("Not graded. A biomarker-only report can be signed.")

        for key, t in self.tiles.items():
            t.set(*self._marker(c, key), locked)
        ex = [f"{k}: {e.get('summary', '')}" for k, e in c.extras.items()]
        self.extras.setText("\n".join(ex) if ex else "None yet. The tumour map and nuclei analyses add findings here.")

        self.snaps.clear()
        for s in c.snapshots:
            rgb = read_rgb(c.dir / s["file"])
            if rgb is None:
                continue
            it = QListWidgetItem(QIcon(to_pixmap(resize_w(rgb, 170))), s["caption"][:34])
            it.setData(Qt.UserRole, s["file"])
            it.setToolTip(s["caption"])
            self.snaps.addItem(it)
        if not c.snapshots:
            it = QListWidgetItem("No images yet.")
            it.setFlags(Qt.NoItemFlags)
            self.snaps.addItem(it)

    @staticmethod
    def _marker(c, name):
        b = c.biomarkers.get(name)
        if not b:
            return "Not assessed", "", "neutral", 0, "", False
        d = b["result"]
        if name in ("ER", "PR"):
            val = d["status"]
            det = f"{d['pct_positive']}% positive, Allred {d['allred_total']}, H-score {d['h_score']}, {d['n_tumour']} nuclei"
            tone = {"Positive": "good", "Low positive": "warn"}.get(val, "neutral")
        elif name == "HER2":
            val = d["score"] + (f" {d['subgroup']}" if d.get("subgroup") else "")
            det = d["her2_low"] + (", reflex ISH required" if d.get("reflex_ish") else "") + f", {d['n_cells']} cells"
            fc = d.get("field_classifier") or {}
            if fc.get("classifier_score"):
                det += f". Field classifier {fc['classifier_score']}."
            tone = {"3+": "bad", "2+": "warn"}.get(d["score"], "neutral")
        else:
            prim = d.get("primary", "global")
            m = d.get(prim) or d.get("global")
            val = f"{m['index_pct']}%"
            det = f"{prim.capitalize()} index, {m['n_positive']} of {m['n_counted']} nuclei. {m['band']}."
            tone = "warn" if m["band"].startswith("Intermediate") else "neutral"
        ws = d.get("warnings", []) + d.get("flags", [])
        for sub in ("global", "hotspot"):
            if isinstance(d.get(sub), dict):
                ws += d[sub].get("warnings", [])
        ws = list(dict.fromkeys(ws))
        return val, det, tone, len(ws), "\n".join(f"• {w}" for w in ws), True

    def _view_snap(self, it):
        c = self.ctx.case
        if not it.data(Qt.UserRole):
            return
        rgb = read_rgb(c.dir / it.data(Qt.UserRole))
        if rgb is not None:
            ImageDialog(self, rgb, it.toolTip()).exec()

    # ------------------------------------------------------------------ actions
    def show_audit(self):
        if self.ctx.case:
            AuditDialog(self, self.ctx.case).exec()

    def edit_details(self):
        c = self.ctx.case
        d = CaseDialog(self, c)
        if d.exec() != QDialog.Accepted:
            return
        _, p, specimen = d.values()
        try:
            changed = c.set_patient(specimen=specimen, **p)
        except (ValueError, PermissionError) as e:
            error_box(self, "Could not change the details", e)
            return
        self.ctx.save_case()
        if changed:
            self.toast("Details saved", "good")

    def set_component(self, k, new):
        c = self.ctx.case
        comp = c.grade.c[k]
        if new == comp.score:
            return
        reason, ok = QInputDialog.getText(self, f"{LABELS[k]}: score {new}",
                                          "Reason for this score. It is written to the audit log, and the report shows "
                                          "the change.")
        if not ok:
            self.refresh()
            return
        try:
            c.grade.set_manual(k, new, reason or "set by pathologist")
        except (ValueError, PermissionError) as e:
            error_box(self, "Score not changed", e)
        self.ctx.save_case()
        self.toast(f"{LABELS[k]} set to {new}", "good")

    def write_pdf(self, final=False):
        c = self.ctx.case
        if c is None:
            return None
        from core.grading.report import write_breast_report
        rep = c.dir / "reports"
        rep.mkdir(exist_ok=True)
        tag = "signed" if final else "draft_" + time.strftime("%Y%m%d_%H%M%S")
        path = rep / f"report_{c.case_id}_{tag}.pdf"
        lab = self.ctx.settings.value("lab_name", "", str)
        try:
            write_breast_report(c, path, lab=lab, microscope=self.ctx.profile.name)
        except Exception as e:  # noqa: BLE001
            error_box(self, "Report not written", e)
            return None
        self.ctx.save_case()
        self.toast(f"{'Signed report' if final else 'Draft'} saved: {path.name}", "good")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        return path

    def sign(self):
        c = self.ctx.case
        if c is None or c.signed_by:
            return
        g = c.grade
        if c.grade_assessed and not g.complete:
            miss = [LABELS[k].lower() for k in COMPONENTS if g.c[k].score is None]
            error_box(self, "Grade incomplete", f"Score {', '.join(miss)} first: measure it or choose the score.")
            return
        if not c.grade_assessed and not c.biomarkers:
            error_box(self, "Nothing to report", "Save at least one result to the case first.")
            return
        unvalidated = [LABELS[k] for k in COMPONENTS if g.c[k].source == "auto" and not g.c[k].validated]
        name, ok = QInputDialog.getText(self, "Sign report", "Reporting pathologist:", text=self.ctx.operator)
        if not ok or not name.strip():
            return
        text = (f"Sign case {c.case_id} as {name.strip()}?\n\n"
                + ("No grade: this is a biomarker-only report.\n" if not c.grade_assessed else f"Grade {g.grade} (total {g.total}).\n")
                + (f"Accepted from the assistant without change: {', '.join(unvalidated)}.\n" if unvalidated else "")
                + "\nAfter signing, nothing in the case can change.")
        if QMessageBox.question(self, "Sign report", text) != QMessageBox.Yes:
            return
        try:
            c.sign(name.strip(), allow_no_grade=True)
        except (ValueError, PermissionError) as e:
            error_box(self, "Not signed", e)
            return
        self.ctx.save_case()
        self.write_pdf(final=True)

    def open_reports(self):
        c = self.ctx.case
        if c is None:
            return
        p = c.dir / "reports"
        p.mkdir(exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
