"""Custom widgets: the case slide label, segmented control, module list rows, setup steps."""
from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from .theme import C, Chip, button, label, pixmap


class SlideLabel(QWidget):
    """The open case drawn as a glass slide: the frosted label end carries the case ID (written along it, as on a
    real slide), the glass carries the patient. Click to open the case overview."""
    clicked = Signal()

    def __init__(self):
        super().__init__()
        self.case = None
        self.setFixedHeight(104)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Open case overview")

    def set_case(self, case):
        self.case = case
        self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        c = self.case
        if c is None:
            pen = QPen(QColor(C["line2"]), 1.2, Qt.DashLine)
            p.setPen(pen)
            p.setBrush(QColor(C["glass"]))
            p.drawRoundedRect(r, 8, 8)
            p.setPen(QColor(C["ink2"]))
            f = QFont(self.font())
            f.setPixelSize(13)
            f.setWeight(QFont.DemiBold)
            p.setFont(f)
            p.drawText(r.adjusted(0, -10, 0, -10), Qt.AlignCenter, "No case open")
            f.setWeight(QFont.Normal)
            f.setPixelSize(12)
            p.setFont(f)
            p.setPen(QColor(C["ink3"]))
            p.drawText(r.adjusted(0, 22, 0, 22), Qt.AlignCenter, "Open or create one")
            return
        # glass
        p.setPen(QPen(QColor(C["line2"]), 1))
        p.setBrush(QColor("#F8FAFC"))
        p.drawRoundedRect(r, 8, 8)
        # frosted label end
        lw = 40
        path = QPainterPath()
        path.addRoundedRect(QRectF(r.left(), r.top(), lw + 8, r.height()), 8, 8)
        clip = QPainterPath()
        clip.addRect(QRectF(r.left(), r.top(), lw, r.height()))
        frost = path.intersected(clip)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["eosin"]))
        p.drawPath(frost)
        p.save()
        p.setClipPath(frost)
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        for k in range(-int(r.height()), lw + 10, 5):          # frosting hatch
            p.drawLine(int(r.left() + k), int(r.bottom()), int(r.left() + k + r.height()), int(r.top()))
        p.restore()
        # case ID along the label
        p.save()
        p.translate(r.left() + lw / 2 + 5, r.center().y())
        p.rotate(-90)
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor("#FFFFFF"))
        txt = p.fontMetrics().elidedText(c.case_id, Qt.ElideRight, int(r.height() - 12))
        p.drawText(QRectF(-r.height() / 2 + 6, -10, r.height() - 12, 20), Qt.AlignCenter, txt)
        p.restore()
        # patient on the glass
        x = r.left() + lw + 12
        w = r.right() - x - 8
        f = QFont(self.font())
        f.setPixelSize(11)
        p.setFont(f)
        p.setPen(QColor(C["ink3"]))
        p.drawText(QRectF(x, r.top() + 10, w, 14), Qt.AlignLeft, "Patient")
        f.setPixelSize(15)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        p.setPen(QColor(C["ink"]))
        pid = p.fontMetrics().elidedText(c.patient.get("patient_id", "-"), Qt.ElideRight, int(w))
        p.drawText(QRectF(x, r.top() + 24, w, 22), Qt.AlignLeft | Qt.AlignVCenter, pid)
        f.setPixelSize(12)
        f.setWeight(QFont.Normal)
        p.setFont(f)
        p.setPen(QColor(C["ink2"]))
        name = c.patient.get("name") or c.specimen
        p.drawText(QRectF(x, r.top() + 47, w, 16), Qt.AlignLeft, p.fontMetrics().elidedText(name, Qt.ElideRight, int(w)))
        # status
        signed = bool(c.signed_by)
        txt = "Signed" if signed else "Draft"
        fg, bg = (C["ok"], C["ok_soft"]) if signed else (C["dab"], C["dab_soft"])
        f.setPixelSize(11)
        f.setWeight(QFont.DemiBold)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(txt) + 16
        chip = QRectF(x, r.bottom() - 28, tw, 18)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(bg))
        p.drawRoundedRect(chip, 9, 9)
        p.setPen(QColor(fg))
        p.drawText(chip, Qt.AlignCenter, txt)


class Segmented(QWidget):
    """Row of checkable buttons acting as one control (objectives, scores)."""
    changed = Signal(str)

    def __init__(self, items=(), tips=None):
        super().__init__()
        self.h = QHBoxLayout(self)
        self.h.setContentsMargins(0, 0, 0, 0)
        self.h.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.set_items(items, tips)

    def set_items(self, items, tips=None, marks=None):
        for b in list(self.group.buttons()):
            self.group.removeButton(b)
            b.setParent(None)
            b.deleteLater()
        items = list(items)
        for i, t in enumerate(items):
            b = QPushButton(t)
            b.setProperty("kind", "seg")
            b.setProperty("pos", "first" if i == 0 else ("last" if i == len(items) - 1 else "mid"))
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumWidth(b.fontMetrics().horizontalAdvance(t) + 30)
            if tips and t in tips:
                b.setToolTip(tips[t])
            if marks and t in marks:
                b.setStyleSheet(f"QPushButton {{ color: {C['ink3']}; }} QPushButton:checked {{ color: white; }}")
            b.clicked.connect(lambda _=False, t=t: self.changed.emit(t))
            self.group.addButton(b)
            self.h.addWidget(b)

    def set_current(self, text):
        for b in self.group.buttons():
            b.setChecked(b.text() == text)

    def current(self):
        b = self.group.checkedButton()
        return b.text() if b else None


class ModuleRow(QWidget):
    """Row of the module list: title, one-line description, status chip."""

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 7, 8, 7)
        v.setSpacing(1)
        h = QHBoxLayout()
        t = label(cls.title)
        t.setStyleSheet("font-weight:600; font-size:14px;")
        h.addWidget(t)
        h.addStretch(1)
        self.chip = Chip("")
        h.addWidget(self.chip)
        v.addLayout(h)
        v.addWidget(label(cls.short, "caption"))

    def set_status(self, text, tone):
        self.chip.set(text, tone)

    def sizeHint(self):
        return QSize(220, 56)


class StepRow(QWidget):
    """Numbered setup step with a done state and an action."""

    def __init__(self, n, title, action_text, slot):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 6, 0, 6)
        h.setSpacing(12)
        self.badge = QLabel(str(n))
        self.badge.setFixedSize(28, 28)
        self.badge.setAlignment(Qt.AlignCenter)
        self.n = n
        h.addWidget(self.badge, 0, Qt.AlignTop)
        v = QVBoxLayout()
        v.setSpacing(1)
        self.title = label(title)
        self.title.setStyleSheet("font-weight:600; font-size:14px;")
        self.caption = label("", "caption", wrap=True)
        v.addWidget(self.title)
        v.addWidget(self.caption)
        h.addLayout(v, 1)
        self.btn = button(action_text, slot=slot)
        h.addWidget(self.btn, 0, Qt.AlignVCenter)
        self.set_done(False, "")

    def set_done(self, done, caption):
        self.caption.setText(caption)
        if done:
            self.badge.setPixmap(pixmap("check", "#FFFFFF", 16, 2.4))
            self.badge.setStyleSheet(f"background:{C['ok']}; border-radius:14px;")
        else:
            self.badge.setText(str(self.n))          # replaces the tick pixmap
            self.badge.setStyleSheet(f"background:{C['haem_soft']}; color:{C['haem']}; border-radius:14px; font-weight:600;")
