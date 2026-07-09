"""
Quality-control PDF report generator for a single inspected fabric batch.
Consumes the report dict produced by batch_inspect.build_report (as stored in the
backend batch history) and renders a printable QC record with the pass/fail
verdict, defect-location map, per-zone table, and defect events.
"""
import io
import base64

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable)

GREEN = colors.HexColor('#15803d')
RED = colors.HexColor('#b91c1c')
DARK = colors.HexColor('#0f172a')
GREY = colors.HexColor('#64748b')
LIGHT = colors.HexColor('#e2e8f0')


def _decode_map(defect_map):
    if not defect_map:
        return None
    try:
        b64 = defect_map.split(',', 1)[-1]
        return io.BytesIO(base64.b64decode(b64))
    except Exception:
        return None


def generate_qc_pdf(report, out_path, timestamp="", operator="__________________"):
    """Render a QC PDF for one batch. `report` is the batch report/history dict."""
    passed = report.get('passed', report.get('defect_frames', 0) == 0)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('t', parent=styles['Title'], fontSize=18, textColor=DARK, spaceAfter=2)
    sub = ParagraphStyle('s', parent=styles['Normal'], fontSize=10, textColor=GREY, alignment=TA_LEFT)
    h2 = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=12, textColor=DARK, spaceBefore=10, spaceAfter=4)
    small = ParagraphStyle('sm', parent=styles['Normal'], fontSize=9, textColor=GREY)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"QC Report {report.get('batch','')}")
    story = []
    story.append(Paragraph("UltraFabric-Vision &mdash; Fabric Quality Control Report", title))
    story.append(Paragraph("Automated textile defect inspection record", sub))
    story.append(HRFlowable(width='100%', thickness=1, color=LIGHT, spaceBefore=6, spaceAfter=8))

    # Verdict banner
    verdict = "PASS &mdash; No defects detected" if passed else "REJECT &mdash; Defects detected"
    vstyle = ParagraphStyle('v', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER,
                            textColor=colors.white, leading=22)
    banner = Table([[Paragraph(f"<b>BATCH {report.get('batch','')} : {verdict}</b>", vstyle)]],
                   colWidths=[doc.width])
    banner.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), GREEN if passed else RED),
                                ('TOPPADDING', (0, 0), (-1, -1), 8),
                                ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
    story.append(banner)
    story.append(Spacer(1, 10))

    # Metadata
    rate = report.get('defect_rate')
    if rate is None:
        rate = report.get('defect_frames', 0) / max(1, report.get('processed_frames', 1))
    meta = [
        ['Batch number', str(report.get('batch', '')), 'Inspected', timestamp or '—'],
        ['Source', str(report.get('source', '—'))[:40], 'Mode', str(report.get('mode', '—'))],
        ['Batch length', f"{report.get('batch_length_m','—')} m", 'Device', str(report.get('device', '—'))],
        ['Frames processed', str(report.get('processed_frames', '—')), 'Defect frames',
         f"{report.get('defect_frames', 0)} ({rate*100:.0f}%)"],
        ['Decision threshold', str(report.get('threshold', '—')), 'Min defect size',
         f"{report.get('min_defect_area_frac', 0)*100:.2f}% of frame"],
    ]
    mt = Table(meta, colWidths=[doc.width * 0.22, doc.width * 0.28, doc.width * 0.22, doc.width * 0.28])
    mt.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (0, -1), GREY), ('TEXTCOLOR', (2, 0), (2, -1), GREY),
        ('TEXTCOLOR', (1, 0), (1, -1), DARK), ('TEXTCOLOR', (3, 0), (3, -1), DARK),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3), ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(mt)

    # Defect-location map
    map_buf = _decode_map(report.get('defect_map'))
    if map_buf is not None:
        story.append(Paragraph("Defect location along batch", h2))
        try:
            story.append(Image(map_buf, width=doc.width, height=doc.width * 0.14))
        except Exception:
            pass

    # Defect events
    events = report.get('defect_events', [])
    story.append(Paragraph("Defect events", h2))
    if events:
        rows = [['#', 'Position (m)', 'Zones', 'Peak score']]
        for i, e in enumerate(events, 1):
            rows.append([str(i), f"{e['start_m']} – {e['end_m']}",
                         ", ".join(map(str, e['zones'])), f"{e['max_score']}"])
        et = Table(rows, colWidths=[doc.width * 0.1, doc.width * 0.35, doc.width * 0.3, doc.width * 0.25])
        et.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), DARK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9), ('GRID', (0, 0), (-1, -1), 0.4, LIGHT),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('TEXTCOLOR', (0, 1), (-1, -1), RED),
        ]))
        story.append(et)
    else:
        story.append(Paragraph("None &mdash; batch is within quality tolerance.", small))

    # Per-zone table
    story.append(Paragraph("Per-segment report", h2))
    zrows = [['Zone', 'Position (m)', 'Status', 'Defect frames', 'Max score']]
    for z in report.get('segment_summary', []):
        zrows.append([str(z['zone']), z['position_m'], z['status'],
                      str(z['defect_frames']), str(z['max_score'])])
    zt = Table(zrows, colWidths=[doc.width * 0.12, doc.width * 0.3, doc.width * 0.2,
                                 doc.width * 0.2, doc.width * 0.18])
    style = [('BACKGROUND', (0, 0), (-1, 0), DARK), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
             ('FONTSIZE', (0, 0), (-1, -1), 9), ('GRID', (0, 0), (-1, -1), 0.4, LIGHT)]
    for i, z in enumerate(report.get('segment_summary', []), 1):
        if z['status'] == 'DEFECT':
            style.append(('TEXTCOLOR', (0, i), (-1, i), RED))
            style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#fef2f2')))
    zt.setStyle(TableStyle(style))
    story.append(zt)

    # Footer / signature
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width='100%', thickness=0.5, color=LIGHT, spaceAfter=6))
    sig = Table([[Paragraph(f"QC Inspector: {operator}", small),
                  Paragraph("Signature: __________________", small)]],
                colWidths=[doc.width * 0.5, doc.width * 0.5])
    story.append(sig)
    story.append(Spacer(1, 4))
    story.append(Paragraph("Generated by UltraFabric-Vision. Automated decision support; "
                           "final disposition subject to QC review.", small))
    doc.build(story)
    return out_path
