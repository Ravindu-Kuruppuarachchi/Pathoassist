"""Combined breast report (Feature 9): Nottingham grade with its components, ER, PR, HER2, Ki-67,
snapshot images, method / validation status of each automatic result, and the audit fingerprint."""
from __future__ import annotations

import time
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core.grading.case import PATIENT_FIELDS, BreastCase
from core.grading.nottingham import COMPONENTS, LABELS

CORE_BIOPSY_CAVEAT = ("Grade assessed on a core biopsy may differ from the grade of the excision specimen (in particular "
                      "the mitotic count may be under-estimated in limited material); the grade should be confirmed on "
                      "the excision specimen.")
DISCLAIMER = ("Computer-assisted assessment for decision support. All automatic results were reviewed and may have "
              "been edited by the reporting pathologist, who is responsible for the final report.")


def _tbl(rows, widths, header=True, shade=None):
    t = Table(rows, colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP")]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb"))]
    if shade:
        style += [("BACKGROUND", (0, shade), (-1, shade), colors.HexColor("#fff6e0"))]
    t.setStyle(TableStyle(style))
    return t


def write_breast_report(case: BreastCase, path, lab="", microscope="") -> Path:
    path = Path(path)
    st = getSampleStyleSheet()
    small = st["Normal"].clone("small", fontSize=8, leading=10)
    P = lambda t, s=st["Normal"]: Paragraph(t, s)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"Breast report {case.case_id}")
    g = case.grade
    el = [P("Breast carcinoma - assisted pathology report", st["Title"]),
          P(f"Case <b>{case.case_id}</b> &nbsp;|&nbsp; specimen: <b>{case.specimen}</b> &nbsp;|&nbsp; {time.strftime('%Y-%m-%d %H:%M')}"
            + (f" &nbsp;|&nbsp; {lab}" if lab else "") + (f" &nbsp;|&nbsp; {microscope}" if microscope else "")),
          Spacer(1, 3 * mm)]
    pat = [(label, case.patient[k]) for k, label in PATIENT_FIELDS if case.patient.get(k)]
    if pat:
        prow = [[P(f"<b>{l}</b>", small), P(escape(str(v)), small)] for l, v in pat]
        pt = Table(prow, colWidths=[45 * mm, 135 * mm])
        pt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f4f4f4"))]))
        el += [pt, Spacer(1, 3 * mm)]
    el += [P("Histological grade (Nottingham / Elston-Ellis)", st["Heading2"])]
    if not case.grade_assessed:
        el += [P("Not assessed in this report.", st["Normal"])]
    rows = [["Component", "Assistant", "Final score", "Source", "Measurement"]]
    for k in COMPONENTS:
        c = g.c[k]
        src = {"auto": "assistant (accepted)", "edited": "edited by pathologist", "manual": "pathologist", "missing": "-"}[c.source]
        if c.source in ("auto", "edited") and not c.validated:
            src += " *"
        rows.append([LABELS[k], c.auto_score if c.auto_score is not None else "-", c.score if c.score is not None else "-", src,
                     P(c.value or "-", small)])
    rows.append(["Total / grade", "", f"{g.total if g.total else '-'}", "", P(f"<b>Grade {g.grade}</b>" if g.grade else "incomplete")])
    if case.grade_assessed:
        el += [_tbl(rows, [38 * mm, 20 * mm, 20 * mm, 38 * mm, 64 * mm], shade=len(rows) - 1)]
    if any(g.c[k].source in ("auto", "edited") and not g.c[k].validated for k in COMPONENTS):
        el.append(P("* automatic method not yet validated against local pathologists.", small))
    if case.specimen == "core biopsy" and case.grade_assessed:
        el += [Spacer(1, 2 * mm), P(f"<font color='#b35900'><b>Core biopsy:</b> {CORE_BIOPSY_CAVEAT}</font>", small)]

    el += [Spacer(1, 5 * mm), P("Biomarkers", st["Heading2"])]
    rows = [["Marker", "Result", "Details", "Method"]]
    for name in ("ER", "PR", "HER2", "Ki67"):
        b = case.biomarkers.get(name)
        if not b:
            rows.append([name, "not assessed", "", ""])
            continue
        d = b["result"]
        if name in ("ER", "PR"):
            res = d["status"]
            det = (f"{d['pct_positive']}% tumour nuclei positive; Allred {d['allred_total']} (PS {d['allred_ps']} + IS {d['allred_is']}); "
                   f"H-score {d['h_score']}; {d['n_tumour']} tumour nuclei")
        elif name == "HER2":
            res = d["score"] + (f" ({d['subgroup']})" if d.get("subgroup") else "")
            det = (f"{d['her2_low']}; any staining {d['pct_any_staining']}%, complete {d['pct_complete']}%, strong {d['pct'][3]}%; "
                   f"{d['n_cells']} cells" + ("; <b>reflex ISH required</b>" if d.get("reflex_ish") else ""))
        else:
            gl, hs = d.get("global", d), d.get("hotspot")
            main = hs if d.get("primary") == "hotspot" and hs else gl
            res = f"{main['index_pct']}% ({main.get('mode', 'global')})"
            det = f"global {gl['index_pct']}% ({gl['n_positive']}/{gl['n_counted']})" + (f"; hotspot {hs['index_pct']}%" if hs else "") \
                  + f"; {gl['band']}"
        meth = b.get("method") or "assisted count"
        if not b.get("validated"):
            meth += " *"
        rows.append([name if name != "Ki67" else "Ki-67", P(f"<b>{res}</b>", small), P(det, small), P(meth, small)])
    el += [_tbl(rows, [18 * mm, 30 * mm, 100 * mm, 32 * mm])]
    if any(not b.get("validated") for b in case.biomarkers.values()):
        el.append(P("* thresholds / model not yet validated on local cases.", small))

    extras = [(k, e) for k, e in case.extras.items() if e.get("summary")]
    if extras:
        el += [Spacer(1, 4 * mm), P("Other assisted findings", st["Heading2"]),
               _tbl([["Module", "Result"]] + [[k, P(e["summary"], small)] for k, e in extras], [38 * mm, 142 * mm])]

    flags = []
    for name, b in case.biomarkers.items():
        d = b["result"]
        ws = d.get("warnings", []) + d.get("flags", [])
        for sub in ("global", "hotspot"):
            if isinstance(d.get(sub), dict):
                ws += d[sub].get("warnings", [])
        flags += [f"{name}: {w}" for w in dict.fromkeys(ws)]
    if flags:
        el += [Spacer(1, 3 * mm), P("Notes from the assistant", st["Heading3"])] + [P(f"- {f}", small) for f in flags[:20]]

    if case.snapshots:
        el += [PageBreak(), P("Snapshots", st["Heading2"])]
        cells, row = [], []
        for s in case.snapshots:
            img = Image(str(case.dir / s["file"]))
            ratio = img.imageHeight / img.imageWidth
            img.drawWidth, img.drawHeight = 85 * mm, 85 * mm * ratio
            row.append([img, P(s["caption"], small)])
            if len(row) == 2:
                cells.append(row)
                row = []
        if row:
            cells.append(row + [""])
        t = Table(cells, colWidths=[90 * mm] * 2)
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        el.append(t)

    ok, msg = case.audit.verify()
    el += [Spacer(1, 6 * mm),
           P(f"Signed by: <b>{case.signed_by}</b>" + (f" on {case.signed_at}" if case.signed_at else "") if case.signed_by else "<font color='#b30000'><b>NOT SIGNED - draft</b></font>"),
           P(f"Audit log: {len(case.audit.entries())} entries, integrity {'verified' if ok else 'FAILED - ' + msg}, "
             f"fingerprint {case.audit.last_hash[:16]}", small),
           Spacer(1, 2 * mm), P(f"<i>{DISCLAIMER}</i>", small)]
    doc.build(el)
    case.audit.log("report_written", file=path.name, signed=bool(case.signed_by))
    return path
