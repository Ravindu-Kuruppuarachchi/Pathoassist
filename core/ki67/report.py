"""Ki-67 overlay and PDF report (global and hotspot index, counts, field thumbnails)."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.ihc.pipeline import TUMOUR, FieldAnalysis
from core.ihc.report import DISCLAIMER, _thumb
from core.ki67.session import Ki67Session

POSITIVE, NEGATIVE, NON_TUMOUR = (230, 0, 0), (40, 110, 255), (170, 170, 170)   # RGB


def draw_ki67_overlay(frame_rgb: np.ndarray, a: FieldAnalysis, thickness: int = 2, corrected=()) -> np.ndarray:
    out = frame_rgb.copy()
    for i, c in enumerate(a.coords):
        pts = np.round(c[::-1].T).astype(np.int32)
        if a.cell_type[i] == TUMOUR:
            col, th = (POSITIVE if a.intensity[i] > 0 else NEGATIVE), thickness
        else:
            col, th = NON_TUMOUR, 1
        cv2.polylines(out, [pts], True, col, th, cv2.LINE_AA)
        if i in corrected:
            y, x = a.points[i]
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), -1)
    return out


def write_ki67_report(session: Ki67Session, path: str | Path, lab: str = "", microscope: str = "",
                      pathologist: str = "", primary: str = "global") -> Path:
    path = Path(path)
    st = getSampleStyleSheet()
    g, h = session.global_result(), session.hotspot_result()
    main_r = g if primary == "global" else h
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=f"Ki-67 report {session.case_id}")
    el = [Paragraph("Ki-67 proliferation index - assisted count", st["Title"]),
          Paragraph(f"Case <b>{session.case_id}</b> &nbsp; | &nbsp; {time.strftime('%Y-%m-%d %H:%M')}"
                    + (f" &nbsp; | &nbsp; {lab}" if lab else "") + (f" &nbsp; | &nbsp; {microscope}" if microscope else ""),
                    st["Normal"]),
          Spacer(1, 6 * mm)]
    rows = [["", "Global", "Hotspot"],
            ["Ki-67 index", f"{g.index_pct:.1f} %", f"{h.index_pct:.1f} %"],
            ["Positive / counted", f"{g.n_positive} / {g.n_counted}", f"{h.n_positive} / {h.n_counted}"],
            ["Fields", str(len(g.fields_used)), ", ".join(str(i + 1) for i in h.fields_used)],
            ["IKWG range", g.band, h.band]]
    t = Table(rows, colWidths=[50 * mm, 60 * mm, 60 * mm])
    hi = 1 if primary == "global" else 2
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                           ("FONTSIZE", (0, 0), (-1, -1), 11), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (hi, 0), (hi, -1), colors.HexColor("#eef3fb"))]))
    el += [Paragraph(f"Reported index (<b>{main_r.mode}</b>): <b>{main_r.index_pct:.1f} %</b>", st["Heading2"]),
           t, Spacer(1, 3 * mm),
           Paragraph("IKWG 2021: values of 5% or less and 30% or more are the reproducible ranges; values between "
                     "6% and 29% have poor inter-observer agreement and should be interpreted with caution.",
                     st["Normal"]),
           Spacer(1, 3 * mm)]
    for w in dict.fromkeys(g.warnings + h.warnings):
        el.append(Paragraph(f"<font color='#b35900'><b>Warning:</b> {w}</font>", st["Normal"]))
    n_corr = sum(len(f.corrected) for f in session.fields)
    if n_corr:
        el.append(Paragraph(f"{n_corr} nuclei were corrected manually.", st["Normal"]))
    el += [Spacer(1, 5 * mm), Paragraph("Counted fields", st["Heading2"]),
           Paragraph("Outlines: <font color='#e60000'>positive</font>, <font color='#2870ff'>negative</font> tumour "
                     "nuclei; grey = non-tumour; white dot = corrected. * = part of the hotspot.", st["Normal"]),
           Spacer(1, 2 * mm)]
    cells, row = [], []
    col_w = 88 * mm
    for f, c in zip(session.fields, session.field_counts()):
        rgb = cv2.cvtColor(cv2.imread(str(session.dir / f.frame_file)), cv2.COLOR_BGR2RGB)
        buf, ar = _thumb(draw_ki67_overlay(rgb, f.analysis, thickness=3, corrected=f.corrected))
        idx = 100 * c[1] / c.sum() if c.sum() else 0
        star = " *" if f.index in h.fields_used else ""
        cap = Paragraph(f"Field {f.index + 1}{star}: {c[1]} / {c.sum()} positive ({idx:.1f}%)", st["Normal"])
        row.append([Image(buf, width=col_w, height=col_w * ar), cap])
        if len(row) == 2:
            cells.append(row)
            row = []
    if row:
        cells.append(row + [""])
    if cells:
        gt = Table(cells, colWidths=[col_w + 2 * mm] * 2)
        gt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el.append(gt)
    el.append(Spacer(1, 6 * mm))
    if pathologist:
        el.append(Paragraph(f"Reviewed by: {pathologist}", st["Normal"]))
    el.append(Paragraph(f"<i>{DISCLAIMER}</i>", st["Normal"]))
    doc.build(el)
    return path
