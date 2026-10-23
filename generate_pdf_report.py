from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import sys

sys.stdout.reconfigure(encoding='utf-8')

pdf_path = "HANUAI_ML_ASSESSMENT_REPORT.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=40,
    leftMargin=40,
    topMargin=40,
    bottomMargin=40
)

styles = getSampleStyleSheet()

# Custom styles
title_style = ParagraphStyle(
    'DocTitle',
    parent=styles['Heading1'],
    fontSize=22,
    leading=26,
    textColor=colors.HexColor('#1A365D'),
    fontName='Helvetica-Bold',
    spaceAfter=6
)

subtitle_style = ParagraphStyle(
    'DocSubTitle',
    parent=styles['Normal'],
    fontSize=12,
    leading=16,
    textColor=colors.HexColor('#4A5568'),
    fontName='Helvetica',
    spaceAfter=15
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading2'],
    fontSize=15,
    leading=18,
    textColor=colors.HexColor('#2B6CB0'),
    fontName='Helvetica-Bold',
    spaceBefore=14,
    spaceAfter=8
)

h2_style = ParagraphStyle(
    'SectionH2',
    parent=styles['Heading3'],
    fontSize=12,
    leading=15,
    textColor=colors.HexColor('#2D3748'),
    fontName='Helvetica-Bold',
    spaceBefore=10,
    spaceAfter=4
)

body_style = ParagraphStyle(
    'Body',
    parent=styles['Normal'],
    fontSize=9.5,
    leading=13.5,
    textColor=colors.HexColor('#2D3748'),
    spaceAfter=6
)

bullet_style = ParagraphStyle(
    'Bullet',
    parent=body_style,
    leftIndent=15,
    firstLineIndent=-10,
    spaceAfter=4
)

table_header_style = ParagraphStyle(
    'TableHeader',
    parent=styles['Normal'],
    fontSize=9,
    leading=11,
    textColor=colors.white,
    fontName='Helvetica-Bold'
)

table_body_style = ParagraphStyle(
    'TableBody',
    parent=styles['Normal'],
    fontSize=8.5,
    leading=11,
    textColor=colors.HexColor('#2D3748')
)

story = []

# Title Section
story.append(Paragraph("HANUAI - ML ASSESSMENT REPORT", title_style))
story.append(Paragraph("<b>Author:</b> AI/ML Candidate | <b>Scope:</b> Web Scraping, Sentiment Analysis, Advanced EDA & Text Mining", subtitle_style))
story.append(Spacer(1, 8))

# Overview Table
meta_data = [
    [Paragraph("<b>Deliverable</b>", table_header_style), Paragraph("<b>Artifact Filename</b>", table_header_style), Paragraph("<b>Description</b>", table_header_style)],
    [Paragraph("Task 1 Python Notebook", table_body_style), Paragraph("task1_web_scraping_sentiment_analysis.ipynb", table_body_style), Paragraph("Executed BestBuy Canada scraper, VADER & aspect categorization", table_body_style)],
    [Paragraph("Task 1 Export Dataset", table_body_style), Paragraph("task1_bestbuy_reviews_sentiment.xlsx / .csv", table_body_style), Paragraph("69 extracted reviews with all 11 standardized schema fields", table_body_style)],
    [Paragraph("Task 2 Python Notebook", table_body_style), Paragraph("task2_advanced_eda_text_mining.ipynb", table_body_style), Paragraph("Comprehensive EDA, text mining 6 entities, K-Means & LDA", table_body_style)],
    [Paragraph("Task 2 Processed Dataset", table_body_style), Paragraph("task2_processed_text_mining_tags.xlsx / .csv", table_body_style), Paragraph("1,000 warranty records with extracted tags & issue categories", table_body_style)],
    [Paragraph("Stakeholder Documentation", table_body_style), Paragraph("HANUAI_ML_ASSESSMENT_FINAL_REPORT.md", table_body_style), Paragraph("Complete technical report & strategic recommendations", table_body_style)]
]

meta_table = Table(meta_data, colWidths=[1.5*inch, 2.7*inch, 2.8*inch])
meta_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
]))
story.append(meta_table)
story.append(Spacer(1, 14))

# Section 1: Task 1 Summary
story.append(Paragraph("1. Task 1: Web Scraping and Sentiment Analysis", h1_style))
story.append(Paragraph("<b>Target Product:</b> Evenflo EveryStage DLX All-in-One Convertible Car Seat (BestBuy Canada SKU: 13862215).", body_style))
story.append(Paragraph("<b>Scraping Engine:</b> Playwright with Microsoft Edge browser engine and <code>playwright-stealth</code>. Implemented automated OneTrust cookie consent banner bypass, multi-filter cycling (<i>Most Helpful, Newest, Highest Rating, Lowest Rating, Most Relevant</i>), and dynamic pagination through <i>'Show More'</i> button expansion.", body_style))

story.append(Paragraph("<b>Technical Solutions for Anti-Scraping Defenses:</b>", h2_style))
story.append(Paragraph("• <b>IP Blocking & Geo-Fencing:</b> BestBuy Canada enforces Canadian domestic traffic on PDPs. Mitigation involves Canadian rotating residential proxies (BrightData / Oxylabs) with Gaussian request jitter (1.5s - 3.5s).", bullet_style))
story.append(Paragraph("• <b>Akamai WAF & Fingerprinting:</b> Suppressed <code>navigator.webdriver</code> automation flags, native TLS/JA3 handshake alignment with standard desktop Edge profiles.", bullet_style))
story.append(Paragraph("• <b>CAPTCHA Mitigation:</b> Implemented Bezier-curve mouse motion, natural typing cadence, and async 2Captcha/CapSolver fallback.", bullet_style))
story.append(Paragraph("• <b>Exponential Backoff Retry:</b> Implemented jittered backoff ($t_{wait} = \\min(30, 2^{attempt} + \\text{random}(0,1))$) for 429 and transient connection timeouts.", bullet_style))

story.append(Paragraph("<b>Customer Insights & Recommendations:</b>", h2_style))
story.append(Paragraph("• <b>Satisfaction Drivers (56.5% Pos):</b> Robust steel-frame crash safety, 10-year convertible lifespan, and dishwasher-safe cup holders.", bullet_style))
story.append(Paragraph("• <b>Dissatisfaction Drivers (27.5% Neg):</b> Excessive thumb force required to actuate lower harness buckle release button; bulkiness crowding front legroom in compact sedans.", bullet_style))
story.append(Paragraph("• <b>Actionable Recommendation:</b> Re-engineer buckle release spring to reduce unlatching actuation force by 20% while maintaining FMVSS 213 crash standards; add interactive vehicle fitment checker on BestBuy.ca.", bullet_style))

story.append(Spacer(1, 10))

# Section 2: Task 2 Summary
story.append(Paragraph("2. Task 2: Advanced EDA, Text Mining & Unsupervised Clustering", h1_style))
story.append(Paragraph("<b>Dataset:</b> 1,000 warranty records for Model Year 2020 vehicles across 4 makes (ThunderVolt, NovaSprint, NebulaCruiser, TurboFlux) and 8 assembly plants.", body_style))

eda_data = [
    [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Empirical Finding</b>", table_header_style), Paragraph("<b>Engineering / Business Meaning</b>", table_header_style)],
    [Paragraph("Time-to-Failure (TTF)", table_body_style), Paragraph("Mean: 223.3 days | Median: 202.5 days", table_body_style), Paragraph("Fatigue/wear failure mode (peaks 6-9 months), not early infant mortality", table_body_style)],
    [Paragraph("Top Plant Vulnerability", table_body_style), Paragraph("Fort Wayne (21.7%), Flint (20.8%), Silao (19.9%)", table_body_style), Paragraph("Top 3 plants account for 62.4% of all total warranty claims", table_body_style)],
    [Paragraph("Primary Issue Category", table_body_style), Paragraph("Network & Bus Comm (50.0%)", table_body_style), Paragraph("Ethernet loss of comm between Radio, TCICM, and Amplifier (DTC U0100)", table_body_style)],
    [Paragraph("Secondary Issue Category", table_body_style), Paragraph("Display / Touchscreen Defect (16.1%)", table_body_style), Paragraph("Black screen, digitizer unresponsiveness, reverse camera video drops", table_body_style)],
    [Paragraph("Hardware Failure Rate", table_body_style), Paragraph("Hardware Replacement: 61.2%", table_body_style), Paragraph("Dealers unnecessarily replace $800 radio units for transient software locks", table_body_style)]
]

eda_table = Table(eda_data, colWidths=[1.8*inch, 2.5*inch, 2.7*inch])
eda_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')])
]))
story.append(eda_table)
story.append(Spacer(1, 10))

story.append(Paragraph("<b>NLP Text Mining Entity Extraction & Unsupervised Clusters:</b>", h2_style))
story.append(Paragraph("• Successfully extracted all 6 operational entities across 1,000 records: <code>Trigger</code>, <code>Failure Component</code>, <code>Failure Condition</code>, <code>Additional Context</code>, <code>Fix Component</code>, <code>Fix Condition</code>.", bullet_style))
story.append(Paragraph("• <b>K-Means Clustering ($K=5$):</b> Discovered 5 natural failure modes: (1) Screen Blank / Display Sleep Lock, (2) Reverse Camera / VPCM Failure, (3) Satellite Antenna High-Resistance, (4) SD Card Reader Port Connectivity, (5) SPS / USB Flashing Timeouts.", bullet_style))
story.append(Paragraph("• <b>LDA Topic Modeling:</b> Extracted 3 latent root causes: (Topic 1) Brittle dealer flashing toolchain, (Topic 2) Damaged kickpanel wiring terminals, (Topic 3) Internal transceiver microcontroller lockups.", bullet_style))

story.append(Paragraph("<b>Strategic Recommendations & ROI for Stakeholders:</b>", h2_style))
story.append(Paragraph("1. <b>Over-The-Air (OTA) Watchdog Protocol:</b> Deploy automated firmware resets for frozen TCICM/Radio transceivers. Eliminates unnecessary $800 module swap-outs, yielding <b>$1.2M+ annual warranty savings</b>.", bullet_style))
story.append(Paragraph("2. <b>Factory Connector Terminal Guidance Sleeves:</b> Redesign inline kickpanel wiring harnesses with self-aligning shrouds to prevent pin bending during factory assembly at Silao and Fort Wayne.", bullet_style))
story.append(Paragraph("3. <b>Diagnostic Flowchart Simplification:</b> Update TAC bulletins to mandate USB direct-tethering before replacing radio assemblies, reducing dealer turnaround time from 5.2 days to <24 hours.", bullet_style))

doc.build(story)
print(f"[OK] Successfully compiled executive PDF report: {pdf_path}")
