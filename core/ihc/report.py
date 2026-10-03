"""ER/PR report v1 (PDF): score, counts, warnings and thumbnails of the counted fields."""
from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.ihc.case import CaseSession
from core.ihc.overlay import draw_erpr_overlay
from core.ihc.scoring import INTENSITY_NAMES

DISCLAIMER = ("Computer-assisted count for decision support. The reporting pathologist must review "
              "the counted fields and remains responsible for the final result.")


def _thumb(rgb, width_px=420):
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (width_px, round(h * width_px / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return io.BytesIO(buf.tobytes()), small.shape[0] / small.shape[1]


def write_report(session: CaseSession, path: str | Path, lab: str = "", microscope: str = "",
                 pathologist: str = "") -> Path:
    path = Path(path)
    st = getSampleStyleSheet()
    s = session.score()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=f"{session.stain} report {session.case_id}")
    el = [Paragraph(f"{session.stain} immunohistochemistry - assisted score", st["Title"]),
          Paragraph(f"Case <b>{session.case_id}</b> &nbsp; | &nbsp; {time.strftime('%Y-%m-%d %H:%M')}"
                    + (f" &nbsp; | &nbsp; {lab}" if lab else "") + (f" &nbsp; | &nbsp; {microscope}" if microscope else ""),
                    st["Normal"]),
          Spacer(1, 6 * mm)]

    main = [["Status", s.status],
            ["Tumour nuclei positive", f"{s.pct_positive:.1f} %"],
            ["H-score", f"{s.h_score:.0f} / 300"],
            ["Allred score", f"{s.allred_total} / 8  (proportion {s.allred_ps} + intensity {s.allred_is})"],
            ["Tumour nuclei counted", f"{s.n_tumour} in {len(session.fields)} fields"]]
    t = Table(main, colWidths=[60 * mm, 110 * mm])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 11),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb"))]))
    el += [t, Spacer(1, 5 * mm)]

    bins = [["Intensity"] + [n.capitalize() for n in INTENSITY_NAMES],
            ["Tumour nuclei"] + [str(c) for c in s.counts],
            ["% of tumour nuclei"] + [f"{p:.1f}" for p in s.pct_by_bin]]
    t2 = Table(bins, colWidths=[40 * mm] + [32 * mm] * 4)
    t2.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
    el += [t2, Spacer(1, 4 * mm)]
    for w in s.warnings:
        el.append(Paragraph(f"<font color='#b35900'><b>Warning:</b> {w}</font>", st["Normal"]))
    n_corr = sum(len(f.corrected) for f in session.fields)
    if n_corr:
        el.append(Paragraph(f"{n_corr} nuclei were corrected manually.", st["Normal"]))
    el += [Spacer(1, 5 * mm), Paragraph("Counted fields", st["Heading2"]),
           Paragraph("Outlines: tumour nuclei <font color='#2870ff'>negative</font>, "
                     "<font color='#c9b400'>weak</font>, <font color='#ff8c00'>moderate</font>, "
                     "<font color='#e60000'>strong</font>; grey = non-tumour; white dot = corrected.", st["Normal"]),
           Spacer(1, 2 * mm)]

    cells, row = [], []
    col_w = 88 * mm
    for f, fs in zip(session.fields, session.field_scores()):
        rgb = cv2.cvtColor(cv2.imread(str(session.dir / f.frame_file)), cv2.COLOR_BGR2RGB)
        buf, ar = _thumb(draw_erpr_overlay(rgb, f.analysis, thickness=3, corrected=f.corrected))
        cap = Paragraph(f"Field {f.index + 1}: {fs.n_tumour} tumour nuclei, {fs.pct_positive:.1f}% positive", st["Normal"])
        row.append([Image(buf, width=col_w, height=col_w * ar), cap])
        if len(row) == 2:
            cells.append(row)
            row = []
    if row:
        cells.append(row + [""])
    if cells:
        g = Table(cells, colWidths=[col_w + 2 * mm] * 2)
        g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el.append(g)
    el += [Spacer(1, 6 * mm)]
    if pathologist:
        el.append(Paragraph(f"Reviewed by: {pathologist}", st["Normal"]))
    el.append(Paragraph(f"<i>{DISCLAIMER}</i>", st["Normal"]))
    doc.build(el)
    return path
