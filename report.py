"""PDF disease report (no TensorFlow needed)."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

GREEN = colors.HexColor("#2E7D32")
LIGHT = colors.HexColor("#E8F5E9")
AMBER = colors.HexColor("#F9A825")
RED = colors.HexColor("#C62828")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=GREEN, fontSize=13, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["Normal"], fontSize=10, leading=14)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, textColor=colors.grey, leading=11)
TITLE = ParagraphStyle("T", parent=ss["Title"], textColor=GREEN, fontSize=20, spaceAfter=2)


def _png(pil, width_cm):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "PNG")
    buf.seek(0)
    w = width_cm * cm
    return RLImage(buf, width=w, height=w * pil.height / pil.width)


def _bullets(items):
    return ListFlowable([ListItem(Paragraph(i, B), leftIndent=10) for i in items],
                        bulletType="bullet", start="\u2022", leftIndent=12)


def build_report(target, r, plant_note=""):
    """target: file path or BytesIO. r: result dict from infer.Predictor.analyse()."""
    doc = SimpleDocTemplate(target, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                            title="Plant Disease Report", author="RobustLeaf-Net")
    s = [Paragraph("Plant Disease Report", TITLE),
         Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}" + (f"  |  {plant_note}" if plant_note else ""), SM),
         Spacer(1, 8)]

    # images
    cols = [[_png(r["image"], 5.2), Paragraph("Uploaded leaf", SM)],
            [_png(r["gradcam"], 5.2), Paragraph("Grad-CAM: areas driving the diagnosis", SM)]]
    if r.get("attention") is not None:
        cols.append([_png(r["attention"], 5.2), Paragraph("Model attention map", SM)])
    t = Table([cols], colWidths=[5.7 * cm] * len(cols))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    s += [t, Spacer(1, 8)]

    # diagnosis summary
    status = "Healthy" if r["healthy"] else r["disease"]
    conf = f"{r['confidence'] * 100:.1f}%"
    sev = "None" if r["healthy"] else f"{r['severity_level']} (~{r['severity_pct']:.0f}% of leaf area)"
    rows = [["Crop", r["crop"]], ["Diagnosis", status], ["Confidence", conf],
            ["Estimated severity", sev], ["Cause", r["info"]["cause"] + f" ({r['info']['type']})"]]
    summ = Table([[Paragraph(f"<b>{a}</b>", B), Paragraph(b, B)] for a, b in rows], colWidths=[4.2 * cm, 13 * cm])
    summ.setStyle(TableStyle([("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
                              ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A5D6A7")),
                              ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    s.append(summ)
    if r["uncertain"]:
        s += [Spacer(1, 6), Paragraph(
            "<font color='#C62828'><b>Low confidence.</b></font> Retake the photo (single leaf, good light, "
            "plain background) and confirm with an expert before acting.", B)]

    s.append(Paragraph("Other possibilities", H1))
    top = Table([["Rank", "Crop", "Condition", "Probability"]] +
                [[str(i + 1), c, d, f"{p * 100:.1f}%"] for i, ((c, d), p) in enumerate(r["top3"])],
                colWidths=[1.5 * cm, 5 * cm, 7 * cm, 3.5 * cm])
    top.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#A5D6A7")), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    s.append(top)

    info = r["info"]
    s += [Paragraph("Symptoms", H1), Paragraph(info["symptoms"], B),
          Paragraph("Treatment", H1), _bullets(info["treatment"]),
          Paragraph("Prevention", H1), _bullets(info["prevention"]),
          Spacer(1, 10),
          Paragraph("<b>Disclaimer:</b> This AI-generated report is indicative only. " + r.get("disclaimer", ""), SM)]
    doc.build(s)
    return target
