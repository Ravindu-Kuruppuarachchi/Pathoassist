"""HER2 case session, overlay and PDF report.

Reuses core.ihc.case.CaseSession for de-duplication, field storage and click-to-correct:
`set_intensity(field, cell, category)` changes a cell's membrane category (logged as a label),
`toggle_tumour` includes / excludes a cell.
"""
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
from skimage.segmentation import find_boundaries

from core.her2.pipeline import HER2FieldAnalysis, field_counts
from core.her2.scoring import CELL_CATEGORIES, HER2Result, her2_score
from core.ihc.case import CaseSession
from core.ihc.pipeline import TUMOUR
from core.ihc.report import DISCLAIMER, _thumb

# RGB per category: none, faint/incomplete, weak-moderate complete, strong complete
CAT_COLORS = np.array([(60, 120, 255), (255, 220, 0), (255, 130, 0), (220, 0, 0)], np.uint8)
GREY = np.array((150, 150, 150), np.uint8)


class HER2Session(CaseSession):
    def __init__(self, case_dir, case_id, min_cells=500, max_overlap=0.5):
        super().__init__(case_dir, case_id, "HER2", min_cells=min_cells, max_overlap=max_overlap)

    def counts(self):
        c, ii = np.zeros(4, int), 0
        for f in self.fields:
            fc, fi = field_counts(f.analysis)
            c, ii = c + fc, ii + fi
        return c, ii

    def result(self) -> HER2Result:
        c, ii = self.counts()
        r = her2_score(c, ii, self.min_cells)
        for f in self.fields:
            for fl in f.analysis.artefact.get("flags", []):
                r.flags.append(f"Field {f.index + 1}: {fl}")
        return r

    def summary(self):
        res = self.result()
        return dict(case_id=self.case_id, stain="HER2", n_fields=len(self.fields), result=res.as_dict(),
                    fields=[dict(index=f.index, frame=f.frame_file, counts=field_counts(f.analysis)[0].tolist(),
                                 n_corrected=len(f.corrected), artefact=f.analysis.artefact) for f in self.fields])


def draw_her2_overlay(frame_rgb: np.ndarray, a: HER2FieldAnalysis, corrected=()) -> np.ndarray:
    out = frame_rgb.copy()
    b = find_boundaries(a.labels, mode="inner")
    lab = a.labels[b] - 1
    ok = lab >= 0
    col = np.where(((a.cell_type[lab[ok]] == TUMOUR) & a.measurable[lab[ok]])[:, None],
                   CAT_COLORS[a.intensity[lab[ok]]], GREY)
    ys, xs = np.nonzero(b)
    out[ys[ok], xs[ok]] = col
    for i in corrected:
        y, x = a.points[i]
        cv2.circle(out, (int(x), int(y)), 3, (255, 255, 255), -1)
    return out


def write_her2_report(session: HER2Session, path, lab="", microscope="", pathologist="") -> Path:
    path = Path(path)
    st = getSampleStyleSheet()
    r = session.result()
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=15 * mm,
                            bottomMargin=15 * mm, title=f"HER2 report {session.case_id}")
    score_txt = f"{r.score}" + (f" ({r.subgroup})" if r.subgroup else "")
    el = [Paragraph("HER2 immunohistochemistry - assisted score", st["Title"]),
          Paragraph(f"Case <b>{session.case_id}</b> &nbsp; | &nbsp; {time.strftime('%Y-%m-%d %H:%M')}"
                    + (f" &nbsp; | &nbsp; {lab}" if lab else "") + (f" &nbsp; | &nbsp; {microscope}" if microscope else ""),
                    st["Normal"]), Spacer(1, 6 * mm)]
    main = [["HER2 IHC score", score_txt], ["Category", r.her2_low],
            ["Reflex ISH", "REQUIRED" if r.reflex_ish else "not required"],
            ["Tumour cells assessed", f"{r.n_cells} in {len(session.fields)} fields"],
            ["Any membrane staining", f"{r.pct_any_staining:.1f} %   (0 / 1+ cut-off: 10%)"],
            ["Complete membrane staining", f"{r.pct_complete:.1f} %   (1+ / 2+ cut-off: 10%)"],
            ["Strong complete staining", f"{r.pct[3]:.1f} %   (2+ / 3+ cut-off: 10%)"]]
    t = Table(main, colWidths=[62 * mm, 108 * mm])
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 11),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb")),
                           ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#fff6e0"))]))
    el += [t, Spacer(1, 4 * mm)]
    cats = Table([["Cell category"] + CELL_CATEGORIES, ["% of tumour cells"] + [f"{p:.1f}" for p in r.pct]],
                 colWidths=[36 * mm, 30 * mm, 34 * mm, 38 * mm, 32 * mm])
    cats.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9),
                              ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ALIGN", (1, 0), (-1, -1), "CENTER")]))
    el += [cats, Spacer(1, 3 * mm)]
    for f in r.flags:
        el.append(Paragraph(f"<font color='#b35900'><b>Note:</b> {f}</font>", st["Normal"]))
    if not r.rules_verified:
        el.append(Paragraph(f"<font color='#b30000'>Scoring rules ({r.rules_version}) not yet verified by a pathologist "
                            "for this installation.</font>", st["Normal"]))
    el += [Spacer(1, 5 * mm), Paragraph("Counted fields", st["Heading2"]),
           Paragraph("Cell outlines: <font color='#3c78ff'>no staining</font>, <font color='#c9a800'>faint/incomplete"
                     "</font>, <font color='#ff8200'>weak-moderate complete</font>, <font color='#dc0000'>strong complete"
                     "</font>; grey = not counted.", st["Normal"]), Spacer(1, 2 * mm)]
    cells, row, col_w = [], [], 88 * mm
    for f in session.fields:
        rgb = cv2.cvtColor(cv2.imread(str(session.dir / f.frame_file)), cv2.COLOR_BGR2RGB)
        buf, ar = _thumb(draw_her2_overlay(rgb, f.analysis, f.corrected))
        fc, _ = field_counts(f.analysis)
        n = fc.sum()
        cap = Paragraph(f"Field {f.index + 1}: {n} cells; any staining {100 * fc[1:].sum() / max(n, 1):.0f}%, "
                        f"complete {100 * fc[2:].sum() / max(n, 1):.0f}%", st["Normal"])
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
    el.append(Spacer(1, 6 * mm))
    if pathologist:
        el.append(Paragraph(f"Reviewed by: {pathologist}", st["Normal"]))
    el.append(Paragraph(f"<i>{DISCLAIMER}</i>", st["Normal"]))
    doc.build(el)
    return path
