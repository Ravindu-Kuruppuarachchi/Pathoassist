"""Visual system: colour tokens (taken from the stains the app measures), IBM Plex Sans, the Qt stylesheet,
an inline icon set and small widget helpers.

Tokens
    glass  #EEF1F5   app background - cool slide-glass grey
    paper  #FFFFFF   panels
    ink    #1B2230   text;   ink2 #5A6475 secondary text;   line #DCE1E8 hairlines
    haem   #34318A   haematoxylin indigo - primary actions, selection, focus
    eosin  #C93C6E   eosin - used only for the case "slide label" and the active navigation marker
    dab    #A35F14   DAB brown-amber - warnings, things that need attention
    ok     #237A57   done / within range
    stage  #1F2228   the image stage: images are judged on a neutral dark surround, as in slide viewers
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

C = dict(glass="#EEF1F5", paper="#FFFFFF", ink="#1B2230", ink2="#5A6475", ink3="#8A93A3", line="#DCE1E8",
         line2="#C9D0DA", haem="#34318A", haem_hover="#2A2774", haem_soft="#E7E6F5", eosin="#C93C6E",
         eosin_soft="#F7E1EA", dab="#A35F14", dab_soft="#FBEFDF", ok="#237A57", ok_soft="#E1F2EA", bad="#B42318",
         bad_soft="#FCE8E6", stage="#1F2228", stage2="#2A2E36")
TONE = {"neutral": (C["ink2"], "#EDF0F4"), "good": (C["ok"], C["ok_soft"]), "warn": (C["dab"], C["dab_soft"]),
        "bad": (C["bad"], C["bad_soft"]), "accent": (C["haem"], C["haem_soft"])}
FONT = "IBM Plex Sans"

QSS = """
* {{ font-family: "{font}"; color: {ink}; }}
QMainWindow, QDialog {{ background: {glass}; }}
QWidget#page {{ background: {glass}; }}
QToolTip {{ background: {ink}; color: white; border: none; padding: 6px 8px; border-radius: 6px; }}

/* sidebar */
QFrame#sidebar {{ background: {paper}; border-right: 1px solid {line}; }}
QPushButton#nav {{ text-align: left; padding: 9px 14px 9px 12px; border: none; border-left: 3px solid transparent;
    border-radius: 0; background: transparent; color: {ink2}; font-size: 14px; font-weight: 500; }}
QPushButton#nav:hover {{ background: {glass}; color: {ink}; }}
QPushButton#nav:checked {{ background: {haem_soft}; color: {haem}; border-left: 3px solid {eosin}; font-weight: 600; }}

/* text roles */
QLabel[role="h1"] {{ font-size: 22px; font-weight: 600; }}
QLabel[role="h2"] {{ font-size: 15px; font-weight: 600; }}
QLabel[role="h3"] {{ font-size: 13px; font-weight: 600; color: {ink2}; }}
QLabel[role="muted"] {{ color: {ink2}; }}
QLabel[role="caption"] {{ color: {ink3}; font-size: 12px; }}
QLabel[role="value"] {{ font-size: 30px; font-weight: 600; }}
QLabel[role="sub"] {{ color: {ink2}; font-size: 13px; }}

/* panels */
QFrame[role="panel"] {{ background: {paper}; border: 1px solid {line}; border-radius: 10px; }}
QFrame[role="stage"] {{ background: {stage}; border-radius: 10px; }}
QFrame[role="bar"] {{ background: {paper}; border: 1px solid {line}; border-radius: 10px; }}
QFrame[role="hline"] {{ background: {line}; max-height: 1px; min-height: 1px; border: none; }}

/* buttons */
QPushButton {{ background: {paper}; border: 1px solid {line2}; border-radius: 7px; padding: 7px 14px;
    font-size: 13px; font-weight: 500; }}
QPushButton:hover {{ border-color: {haem}; color: {haem}; }}
QPushButton:pressed {{ background: {haem_soft}; }}
QPushButton:disabled {{ color: {ink3}; border-color: {line}; background: #F6F7F9; }}
QPushButton[kind="primary"] {{ background: {haem}; color: white; border: 1px solid {haem}; font-weight: 600; }}
QPushButton[kind="primary"]:hover {{ background: {haem_hover}; color: white; }}
QPushButton[kind="primary"]:disabled {{ background: #A9A8CC; border-color: #A9A8CC; color: #F2F2F8; }}
QPushButton[kind="ghost"] {{ background: transparent; border: 1px solid transparent; color: {ink2}; }}
QPushButton[kind="ghost"]:hover {{ background: {glass}; color: {haem}; }}
QPushButton[kind="danger"] {{ color: {bad}; }}
QPushButton[kind="danger"]:hover {{ border-color: {bad}; background: {bad_soft}; color: {bad}; }}
QPushButton[kind="seg"] {{ border-radius: 0; padding: 6px 12px; margin: 0; }}
QPushButton[kind="seg"]:checked {{ background: {haem}; color: white; border-color: {haem}; }}
QPushButton[kind="seg"][pos="first"] {{ border-top-left-radius: 7px; border-bottom-left-radius: 7px; }}
QPushButton[kind="seg"][pos="last"] {{ border-top-right-radius: 7px; border-bottom-right-radius: 7px; }}
QPushButton[kind="classchip"] {{ border-radius: 14px; padding: 4px 12px; }}

/* inputs */
QLineEdit, QComboBox, QPlainTextEdit, QDoubleSpinBox, QSpinBox, QTextEdit {{ background: {paper};
    border: 1px solid {line2}; border-radius: 7px; padding: 6px 8px; selection-background-color: {haem}; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QDoubleSpinBox:focus, QTextEdit:focus {{ border: 1px solid {haem}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {paper}; border: 1px solid {line2}; selection-background-color: {haem_soft};
    selection-color: {ink}; outline: none; padding: 4px; }}
QCheckBox, QRadioButton {{ spacing: 8px; font-size: 13px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
QCheckBox::indicator {{ border: 1px solid {line2}; border-radius: 4px; background: {paper}; }}
QCheckBox::indicator:checked {{ background: {haem}; border-color: {haem}; image: url({check}); }}
QCheckBox::indicator:disabled {{ background: #F1F3F6; }}

/* lists, tables */
QListWidget, QTableWidget, QTextBrowser {{ background: {paper}; border: 1px solid {line}; border-radius: 8px; }}
QListWidget::item {{ padding: 6px; border-radius: 6px; }}
QListWidget::item:selected {{ background: {haem_soft}; color: {ink}; }}
QListWidget::item:hover {{ background: {glass}; }}
QListWidget#modules {{ border: none; background: transparent; }}
QListWidget#modules::item {{ padding: 0; margin: 1px 0; }}
QTableWidget {{ gridline-color: transparent; alternate-background-color: #F8F9FB; }}
QTableWidget::item {{ padding: 6px 8px; border-bottom: 1px solid {line}; }}
QTableWidget::item:selected {{ background: {haem_soft}; color: {ink}; }}
QHeaderView::section {{ background: {paper}; color: {ink2}; font-weight: 600; font-size: 12px; border: none;
    border-bottom: 1px solid {line2}; padding: 8px; }}
QTextBrowser {{ padding: 6px; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {line2}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {ink3}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {line2}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; }}
QProgressBar {{ background: #E6E9EF; border: none; border-radius: 3px; max-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {haem}; border-radius: 3px; }}
QProgressBar[done="true"]::chunk {{ background: {ok}; }}
QStatusBar {{ background: {paper}; border-top: 1px solid {line}; color: {ink2}; }}
QStatusBar QLabel {{ color: {ink2}; padding: 0 8px; }}
QMenuBar {{ background: {paper}; border-bottom: 1px solid {line}; }}
QMenuBar::item {{ padding: 6px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {haem_soft}; border-radius: 6px; }}
QMenu {{ background: {paper}; border: 1px solid {line2}; padding: 6px; border-radius: 8px; }}
QMenu::item {{ padding: 7px 24px 7px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {haem_soft}; color: {ink}; }}
QMenu::separator {{ height: 1px; background: {line}; margin: 4px 6px; }}
QMessageBox QLabel {{ font-size: 13px; }}
QTabWidget::pane {{ border: none; }}
"""

# 24x24, stroke icons drawn for this app
_ICONS = {
    "home": '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6h-6v6H4a1 1 0 0 1-1-1z"/>',
    "cases": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "scope": '<path d="M9 3h4l1 3-3 7-3-1 2-6z"/><path d="M8.5 13.5l-1.5 3"/><path d="M15 9a6 6 0 0 1 1.5 10"/>'
             '<path d="M5 21h14"/><path d="M8 18h7"/>',
    "report": '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/><path d="M9 12h6"/><path d="M9 16h6"/>',
    "annotate": '<path d="M4 20l4-1 11-11-3-3L5 16z"/><path d="M14 6l3 3"/>',
    "settings": '<path d="M4 6h9"/><path d="M19 6h1"/><path d="M4 12h3"/><path d="M13 12h7"/><path d="M4 18h11"/>'
                '<path d="M19 18h1"/><circle cx="16" cy="6" r="2.2"/><circle cx="10" cy="12" r="2.2"/>'
                '<circle cx="17" cy="18" r="2.2"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    "play": '<path d="M8 5l11 7-11 7z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "analyse": '<circle cx="12" cy="12" r="6.5"/><path d="M12 2v4"/><path d="M12 18v4"/><path d="M2 12h4"/>'
               '<path d="M18 12h4"/><circle cx="12" cy="12" r="1.2"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17.2v.3"/>',
    "plus": '<path d="M12 5v14"/><path d="M5 12h14"/>',
    "save": '<path d="M12 4v11"/><path d="M7 10l5 5 5-5"/><path d="M5 20h14"/>',
    "refresh": '<path d="M20 12a8 8 0 1 1-2.3-5.7"/><path d="M20 4v4h-4"/>',
    "ruler": '<path d="M3 17L17 3l4 4L7 21z"/><path d="M7.5 12.5l2 2"/><path d="M10.5 9.5l2 2"/><path d="M13.5 6.5l2 2"/>',
    "light": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2.5"/><path d="M12 19.5V22"/><path d="M2 12h2.5"/>'
             '<path d="M19.5 12H22"/><path d="M4.9 4.9l1.8 1.8"/><path d="M17.3 17.3l1.8 1.8"/>'
             '<path d="M4.9 19.1l1.8-1.8"/><path d="M17.3 6.7l1.8-1.8"/>',
    "image": '<rect x="3" y="5" width="18" height="14" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-8 8"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
    "sign": '<path d="M3 17c3-4 5 2 8-2s4-6 6-2 1 3 4 1"/><path d="M3 21h18"/>',
    "live": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "trash": '<path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M6 7l1 13h10l1-13"/>',
    "back": '<path d="M15 5l-7 7 7 7"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.3a2.6 2.6 0 1 1 3.6 2.4c-.7.3-1.1.9-1.1 1.6v.6"/>'
            '<path d="M12 17v.4"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><path d="M3 10h18"/>',
    "files": '<path d="M8 3h7l4 4v11H8z"/><path d="M15 3v4h4"/><path d="M5 7v14h10"/>',
    "wand": '<path d="M4 20L15 9"/><path d="M15 4v3"/><path d="M15 11v3"/><path d="M11 7.5h-3"/><path d="M22 7.5h-3"/>'
            '<path d="M18.5 4.5l-1.5 1.5"/><path d="M18.5 10.5l-1.5-1.5"/>',
    "undo": '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
    "point": '<circle cx="12" cy="12" r="3.2"/><circle cx="12" cy="12" r="8"/>',
    "polygon": '<path d="M5 8l6-4 8 5-2 10H7z"/><circle cx="5" cy="8" r="1.3"/><circle cx="11" cy="4" r="1.3"/>'
               '<circle cx="19" cy="9" r="1.3"/><circle cx="17" cy="19" r="1.3"/><circle cx="7" cy="19" r="1.3"/>',
    "tag": '<path d="M3 12V4h8l10 10-8 8z"/><circle cx="7.5" cy="8.5" r="1.4"/>',
    "export": '<path d="M12 15V3"/><path d="M7 8l5-5 5 5"/><path d="M5 14v6h14v-6"/>',
    "tune": '<path d="M4 20l6-6"/><path d="M14 4l6 6-8 8-6-6z"/>',
    "dot": '<circle cx="12" cy="12" r="6" fill="{c}" stroke="none"/>',
    "grid": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/>'
            '<rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
}

_cache = {}


def icon(name: str, color: str = C["ink2"], size: int = 20) -> QIcon:
    key = (name, color, size)
    if key not in _cache:
        _cache[key] = QIcon(pixmap(name, color, size))
    return _cache[key]


def pixmap(name: str, color: str = C["ink2"], size: int = 20, stroke: float = 1.8) -> QPixmap:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" '
           f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">'
           f'{_ICONS[name].replace("{c}", color)}</svg>')
    r = QSvgRenderer(QByteArray(svg.encode()))
    dpr = 2
    pm = QPixmap(size * dpr, size * dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    r.render(p, QRectF(0, 0, size * dpr, size * dpr))
    p.end()
    pm.setDevicePixelRatio(dpr)
    return pm


def _check_png() -> str:
    """Checkbox tick image for the stylesheet (Qt needs a file URL)."""
    from .paths import user_data_dir
    p = user_data_dir() / ".ui" / "check.png"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        pixmap("check", "#FFFFFF", 14, 2.6).toImage().save(str(p))
    return p.as_posix()


def apply(app):
    fonts = Path(__file__).parent / "assets" / "fonts"
    for f in fonts.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(f))
    f = QFont(FONT)
    f.setPixelSize(13)
    f.setHintingPreference(QFont.PreferNoHinting)
    app.setFont(f)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS.format(font=FONT, check=_check_png(), **C))


# ------------------------------------------------------------------------------------------- helpers
def label(text="", role=None, wrap=False) -> QLabel:
    lb = QLabel(text)
    if role:
        lb.setProperty("role", role)
    lb.setWordWrap(wrap)
    return lb


def button(text="", icon_name=None, kind=None, tip=None, slot=None, icon_color=None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if icon_name:
        col = icon_color or ("#FFFFFF" if kind == "primary" else C["ink2"])
        b.setIcon(icon(icon_name, col, 18))
        b.setIconSize(QSize(18, 18))
    if tip:
        b.setToolTip(tip)
    if slot:
        b.clicked.connect(slot)
    b.setCursor(Qt.PointingHandCursor)
    return b


def panel(role="panel", margins=(16, 14, 16, 14), spacing=10, horizontal=False):
    f = QFrame()
    f.setProperty("role", role)
    lay = QHBoxLayout(f) if horizontal else QVBoxLayout(f)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return f, lay


def hline() -> QFrame:
    f = QFrame()
    f.setProperty("role", "hline")
    return f


def section(title, subtitle=None) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(2)
    v.addWidget(label(title, "h2"))
    if subtitle:
        v.addWidget(label(subtitle, "caption", wrap=True))
    return w


class Chip(QLabel):
    """Small rounded status label; tone in neutral / good / warn / bad / accent."""

    def __init__(self, text="", tone="neutral"):
        super().__init__(text)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.set(text, tone)

    def set(self, text, tone="neutral"):
        fg, bg = TONE.get(tone, TONE["neutral"])
        self.setText(text)
        self.setVisible(bool(text))
        self.setStyleSheet(f"QLabel {{ color:{fg}; background:{bg}; border-radius:10px; padding:2px 9px; "
                           "font-size:12px; font-weight:600; }")


class PageHeader(QWidget):
    """Page title on the left, page actions on the right."""

    def __init__(self, title, subtitle=""):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 4)
        v = QVBoxLayout()
        v.setSpacing(2)
        self.title = label(title, "h1")
        self.subtitle = label(subtitle, "sub", wrap=True)
        v.addWidget(self.title)
        v.addWidget(self.subtitle)
        h.addLayout(v, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        h.addLayout(self.actions)

    def add(self, w):
        self.actions.addWidget(w)
        return w


def page_layout(widget, margins=(28, 22, 28, 18), spacing=14) -> QVBoxLayout:
    widget.setObjectName("page")
    v = QVBoxLayout(widget)
    v.setContentsMargins(*margins)
    v.setSpacing(spacing)
    return v


def tone_color(tone):
    return QColor(TONE.get(tone, TONE["neutral"])[0])


def expanding(w):
    w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return w
