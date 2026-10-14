"""Shared helpers for Project_Documentation.docx generation: styles, shaded tables,
hyperlinks, TOC field, hand-drawn diagram embedding."""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = os.environ.get("MSAE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
OUT = os.path.join(ROOT, "Project_Documentation.docx")
DIAG = os.path.join(ROOT, "docs", "diagrams")

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x59, 0x59, 0x59)
LINK = RGBColor(0x0B, 0x5E, 0xA8)

REFS = [
    ("Rombach et al. 2022",
     "R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, “High-Resolution Image "
     "Synthesis with Latent Diffusion Models”, Proc. IEEE/CVF CVPR 2022, pp. 10684–10695.",
     "https://doi.org/10.1109/CVPR52688.2022.01060"),
    ("Esser et al. 2021",
     "P. Esser, R. Rombach, B. Ommer, “Taming Transformers for High-Resolution Image "
     "Synthesis”, Proc. IEEE/CVF CVPR 2021, pp. 12868–12878.",
     "https://doi.org/10.1109/CVPR46437.2021.01268"),
    ("Seibert et al. 2022",
     "P. Seibert, J. Scholz, F. Welschof, K. Schladitz, S. O. Stephan, “Microstructure "
     "Characterization and Reconstruction in Python: MCRpy”, Integrating Materials and "
     "Manufacturing Innovation 11 (2022) 450–466.",
     "https://doi.org/10.1007/s40192-022-00273-4"),
    ("Wang et al. 2003",
     "Z. Wang, E. P. Simoncelli, A. C. Bovik, “Multiscale Structural Similarity for Image "
     "Quality Assessment”, Proc. IEEE Asilomar Conf. on Signals, Systems and Computers, "
     "2003, pp. 1398–1402.",
     "https://doi.org/10.1109/ACSSC.2003.1292216"),
    ("Zhang et al. 2018",
     "R. Zhang, P. Isola, A. A. Efros, E. Shechtman, O. Wang, “The Unreasonable "
     "Effectiveness of Deep Features as a Perceptual Metric”, Proc. IEEE/CVF CVPR 2018, "
     "pp. 586–595. (LPIPS)",
     "https://doi.org/10.1109/CVPR.2018.00068"),
]


def setup(doc):
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(6)
    for name, size, bold, color, before in [
        ("Heading 1", 17, True, ACCENT, 14),
        ("Heading 2", 13, True, ACCENT, 10),
        ("Heading 3", 11.5, True, RGBColor(0x2E, 0x5D, 0x9F), 8),
    ]:
        s = doc.styles[name]
        s.font.name = "Calibri"
        s.font.size = Pt(size)
        s.font.bold = bold
        s.font.color.rgb = color
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(4)
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Inches(0.8)
        sec.left_margin = sec.right_margin = Inches(0.9)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tcPr.append(shd)


def table(doc, headers, rows, widths=None, style="Light Grid Accent 1"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = style
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = h
        shade(c, "1F4E79")
        for p in c.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if widths:
        for i, w in enumerate(widths):
            for r in t.rows:
                r.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def mono(doc, text, size=8.5, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.left_indent = Inches(0.15)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor(0x20, 0x20, 0x20)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F2F2F2")
    pPr.append(shd)
    return p


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def hyperlink(paragraph, url, text):
    part = paragraph.part
    r_id = part.relate_to(
        url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True)
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), "0B5EA8")
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single")
    rPr.append(color); rPr.append(u)
    run.append(rPr)
    t = OxmlElement("w:t"); t.text = text
    run.append(t)
    hl.append(run)
    paragraph._p.append(hl)


def field(paragraph, instr):
    r = paragraph.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = "Right-click → Update Field"
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    for el in (f1, it, f2, t, f3):
        r._r.append(el)


def diagram(doc, name, width_in=6.7, caption=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(DIAG, name + ".png"), width=Inches(width_in))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        r.font.size = Pt(8.5)
        r.font.italic = True
        r.font.color.rgb = GREY
