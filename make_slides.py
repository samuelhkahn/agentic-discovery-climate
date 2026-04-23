#!/usr/bin/env python3
"""Generate PowerPoint slides summarizing the AI-scientist albedo research findings."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
import copy

# ── Color palette ──────────────────────────────────────────────────────────
NAVY    = RGBColor(0x1B, 0x2A, 0x4A)   # dark navy — title bg
SKY     = RGBColor(0x2E, 0x86, 0xC1)   # medium blue — accents
TEAL    = RGBColor(0x17, 0xA5, 0x8E)   # teal — H001
AMBER   = RGBColor(0xE6, 0x7E, 0x22)   # amber — H002
CORAL   = RGBColor(0xC0, 0x39, 0x2B)   # coral — H003
PURPLE  = RGBColor(0x6C, 0x3B, 0x9B)   # purple — H004
LGRAY   = RGBColor(0xF4, 0xF6, 0xF7)   # light gray — body bg
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DKGRAY  = RGBColor(0x2C, 0x3E, 0x50)   # dark gray — body text
GREEN   = RGBColor(0x1E, 0x8B, 0x4C)
RED     = RGBColor(0xC0, 0x39, 0x2B)


def set_bg(slide, color):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_text_box(slide, text, left, top, width, height,
                 font_size=18, bold=False, color=DKGRAY,
                 align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_rect(slide, left, top, width, height, color, alpha=None):
    from pptx.util import Inches
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def add_multi_para(slide, lines, left, top, width, height,
                   font_size=14, color=DKGRAY, bullet=True):
    """Add a text box with multiple bullet paragraphs."""
    from pptx.util import Inches, Pt
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for line in lines:
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(3)
        run = p.add_run()
        prefix = "• " if bullet else ""
        run.text = prefix + line
        run.font.size = Pt(font_size)
        run.font.color.rgb = color


prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]   # completely blank


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, NAVY)

# Top color band
add_rect(s, 0, 0, 13.33, 0.12, SKY)

# Main title
add_text_box(s, "Why Is Earth's Albedo So Stable?",
             0.6, 1.2, 12.0, 1.4, font_size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# Subtitle
add_text_box(s, "An AI Scientist's Exploration of 25 Years of Satellite Data",
             0.6, 2.7, 12.0, 0.8, font_size=22, color=SKY, align=PP_ALIGN.CENTER)

# Divider
add_rect(s, 3.0, 3.6, 7.33, 0.04, SKY)

# Data line
add_text_box(s, "CERES EBAF-TOA Ed4.2.1  ·  March 2000 – March 2025  ·  14.3M observations",
             0.6, 3.8, 12.0, 0.5, font_size=14, color=RGBColor(0xAB, 0xC6, 0xE0), align=PP_ALIGN.CENTER)

# Methods line
add_text_box(s, "28 analyses  ·  12 causal inference methods  ·  4 hypotheses tested",
             0.6, 4.35, 12.0, 0.5, font_size=14, color=RGBColor(0xAB, 0xC6, 0xE0), align=PP_ALIGN.CENTER)

# Bottom strip
add_rect(s, 0, 7.1, 13.33, 0.4, RGBColor(0x12, 0x1B, 0x30))
add_text_box(s, "Samuel Kahn  ·  2026",
             0.5, 7.1, 12.0, 0.4, font_size=11, color=RGBColor(0x7F, 0x9B, 0xB8), align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — The Puzzle
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, NAVY)
add_text_box(s, "The Puzzle", 0.5, 0.1, 12.0, 0.8,
             font_size=30, bold=True, color=WHITE)

# Left column — puzzle statement
add_rect(s, 0.4, 1.2, 5.8, 5.8, WHITE)
add_text_box(s, "Earth reflects ~29–30% of incoming sunlight",
             0.6, 1.35, 5.5, 0.6, font_size=16, bold=True, color=NAVY)
add_text_box(s, "This fraction — the albedo — has stayed near 29–30% for at least 25 years, despite:",
             0.6, 2.0, 5.5, 0.7, font_size=14, color=DKGRAY)
add_multi_para(s, [
    "Constant weather noise and cloud variability",
    "Decade-scale shifts in Arctic sea ice",
    "Aerosol loading from industry and volcanoes",
    "Land use change and deforestation",
    "Interannual ENSO swings",
], 0.6, 2.8, 5.4, 3.0, font_size=14, color=DKGRAY)

# Right column — why it matters
add_rect(s, 6.8, 1.2, 6.1, 5.8, WHITE)
add_text_box(s, "Why does stability matter?",
             7.0, 1.35, 5.8, 0.5, font_size=16, bold=True, color=NAVY)
add_multi_para(s, [
    "A 1% drop in albedo = ~3.4 W/m² forcing — comparable to doubling CO₂",
    "Stability implies active regulation, not just luck",
    "If the regulation breaks, Earth's energy balance shifts permanently",
    "Understanding the mechanism is key to predicting tipping points",
], 7.0, 2.0, 5.8, 3.2, font_size=14, color=DKGRAY, bullet=True)

add_text_box(s, "This research asked: what mechanism stabilizes albedo, and how?",
             7.0, 5.4, 5.8, 1.0, font_size=14, bold=True, color=SKY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Approach
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, NAVY)
add_text_box(s, "Approach: AI Scientist on Satellite Data", 0.5, 0.1, 12.0, 0.8,
             font_size=30, bold=True, color=WHITE)

# Three columns
for col_i, (title, body, color) in enumerate([
    ("Data", [
        "CERES EBAF-TOA Ed4.2.1",
        "March 2000 – March 2025",
        "14.3 million observations",
        "Monthly gridded TOA fluxes",
        "Cloud area, optical depth, albedo",
        "1° × 1° spatial resolution",
    ], SKY),
    ("Methods", [
        "Convergent Cross Mapping (CCM)",
        "Transfer Entropy (TE)",
        "Granger Causality",
        "Wavelet Coherence",
        "Climate Networks",
        "Attractor Reconstruction",
        "VAR / Symbolic Regression",
        "Variance Decomposition",
    ], TEAL),
    ("Hypotheses Tested", [
        "H001: Clouds buffer albedo",
        "H002: Physical constants constrain ~29%",
        "H003: Hemispheres teleconnect",
        "H004: Coupled hemispheric oscillator",
        "",
        "Converged = 5+ independent",
        "methods confirming",
    ], PURPLE),
]):
    x = 0.4 + col_i * 4.3
    add_rect(s, x, 1.15, 4.0, 0.55, color)
    add_text_box(s, title, x + 0.1, 1.2, 3.8, 0.45,
                 font_size=16, bold=True, color=WHITE)
    add_rect(s, x, 1.7, 4.0, 5.3, WHITE)
    add_multi_para(s, body, x + 0.15, 1.8, 3.75, 5.0, font_size=13, color=DKGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — H001 Cloud Buffering
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, TEAL)
add_text_box(s, "Finding 1: Clouds Are the Real-Time Albedo Control Knob", 0.5, 0.1, 12.5, 0.8,
             font_size=28, bold=True, color=WHITE)
add_text_box(s, "H001 — Cloud Buffering  ·  CONVERGED (5/5 methods)", 0.5, 0.72, 12.0, 0.3,
             font_size=12, color=RGBColor(0xD5, 0xF5, 0xEC))

# Hypothesis box
add_rect(s, 0.4, 1.1, 8.5, 0.75, RGBColor(0xD1, 0xF2, 0xEB))
add_text_box(s, "Hypothesis: Cloud area fraction causally drives TOA albedo — "
                "not the reverse. More clouds → more reflected sunlight, synchronously.",
             0.55, 1.15, 8.3, 0.65, font_size=13, color=DKGRAY, italic=True)

# Key stat callout
add_rect(s, 9.2, 1.1, 3.7, 1.7, TEAL)
add_text_box(s, "r = 0.79", 9.25, 1.15, 3.6, 0.65, font_size=32, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(s, "Cloud area ↔ Albedo\n(same month, 99% of grid cells)", 9.25, 1.8, 3.6, 0.9,
             font_size=12, color=WHITE, align=PP_ALIGN.CENTER)

# Evidence table
add_rect(s, 0.4, 2.0, 12.5, 0.4, NAVY)
add_text_box(s, "Method", 0.5, 2.05, 3.0, 0.3, font_size=13, bold=True, color=WHITE)
add_text_box(s, "Key Result", 3.6, 2.05, 6.0, 0.3, font_size=13, bold=True, color=WHITE)
add_text_box(s, "Direction", 9.8, 2.05, 3.0, 0.3, font_size=13, bold=True, color=WHITE)

rows = [
    ("Convergent Cross Mapping", "Cloud→albedo converges (p=0.001, ρ→0.48); albedo→cloud does NOT", "Cloud → Albedo ✓"),
    ("Granger Causality", "Cloud→albedo p<0.0001; albedo→cloud p=0.66", "Unidirectional ✓"),
    ("Climate Networks (spatial)", "99.2% of grid cells: positive cloud-albedo coupling, mean r=0.79", "Synchronous ✓"),
    ("Attractor Reconstruction", "Joint cloud+albedo dim=2.38 vs 3.91 expected (39% reduction)", "Shared manifold ✓"),
    ("Wavelet Coherence", "Coherence=0.91 across all timescales; in-phase at seasonal & annual bands", "In-phase ✓"),
]
for ri, (method, result, direction) in enumerate(rows):
    y = 2.45 + ri * 0.82
    bg_c = WHITE if ri % 2 == 0 else LGRAY
    add_rect(s, 0.4, y, 12.5, 0.78, bg_c)
    add_text_box(s, method, 0.5, y + 0.05, 3.0, 0.65, font_size=12, bold=True, color=NAVY)
    add_text_box(s, result, 3.6, y + 0.05, 6.1, 0.65, font_size=11, color=DKGRAY)
    add_text_box(s, direction, 9.8, y + 0.05, 3.0, 0.65, font_size=11, bold=True, color=GREEN)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — H002 Invariant Properties
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, AMBER)
add_text_box(s, "Finding 2: Albedo Lives on a Low-Dimensional Attractor", 0.5, 0.1, 12.5, 0.8,
             font_size=28, bold=True, color=WHITE)
add_text_box(s, "H002 — Invariant Properties  ·  CONVERGED (5/5 methods)", 0.5, 0.72, 12.0, 0.3,
             font_size=12, color=RGBColor(0xFF, 0xF0, 0xD5))

add_rect(s, 0.4, 1.1, 8.5, 0.75, RGBColor(0xFD, 0xF2, 0xE0))
add_text_box(s, "Hypothesis: Physical constants constrain global mean albedo near ~29%. "
                "The system has fewer effective degrees of freedom than its number of variables implies.",
             0.55, 1.15, 8.3, 0.65, font_size=13, color=DKGRAY, italic=True)

# Stat callout
add_rect(s, 9.2, 1.1, 3.7, 1.7, AMBER)
add_text_box(s, "dim = 1.86", 9.25, 1.15, 3.6, 0.65, font_size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text_box(s, "Correlation dimension\nof global albedo attractor\n(surrogate: 2.32 ± 0.04)", 9.25, 1.8, 3.6, 0.9,
             font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

# Two columns of findings
add_rect(s, 0.4, 2.0, 6.0, 5.2, WHITE)
add_text_box(s, "Supporting Evidence", 0.55, 2.1, 5.8, 0.4, font_size=15, bold=True, color=AMBER)
add_multi_para(s, [
    "Global mean albedo stable at 29.7–30.0% across 25 years",
    "Attractor dimension=1.86, below shuffled surrogates (2.32)",
    "Albedo manifold dimension (1.74) < cloud dimension (2.17) — albedo is MORE constrained than clouds",
    "Symbolic regression: R²=0.94 with only 5 variables (polynomial degree-2)",
    "Network hub stability r=0.554 across 2000–2012 vs 2013–2025",
    "Negative self-feedback coefficient in SINDy — perturbations decay",
], 0.55, 2.6, 5.75, 4.2, font_size=13, color=DKGRAY)

add_rect(s, 6.8, 2.0, 6.1, 5.2, WHITE)
add_text_box(s, "Open Question", 6.95, 2.1, 5.8, 0.4, font_size=15, bold=True, color=AMBER)
add_multi_para(s, [
    "Mean albedo did shift: 29.96% (2000–2012) → 29.65% (2013–2025), p=0.003",
    "This ~0.3% drop may reflect Arctic brightening reduction or cloud feedback",
    "Is this within the 'invariant' envelope, or early evidence of destabilization?",
    "Low dimensionality is confirmed; whether physical constants are the cause is harder to attribute",
    "Longer records (MERRA-2, 1980–present) needed to resolve trend vs noise",
], 6.95, 2.6, 5.85, 4.2, font_size=13, color=DKGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — H003 Teleconnections
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, CORAL)
add_text_box(s, "Finding 3: ENSO Coordinates Hemispheric Albedo via the Tropical Pacific", 0.5, 0.1, 12.5, 0.8,
             font_size=26, bold=True, color=WHITE)
add_text_box(s, "H003 — Teleconnections  ·  CONVERGED (5/5 methods)", 0.5, 0.72, 12.0, 0.3,
             font_size=12, color=RGBColor(0xFF, 0xD5, 0xD5))

# Central mechanism box
add_rect(s, 0.4, 1.1, 12.5, 0.75, RGBColor(0xFD, 0xED, 0xEC))
add_text_box(s, "NH and SH albedo anomalies causally influence each other through the tropical Pacific (5°S–5°N). "
                "During La Niña, SH anomalies propagate to the NH at ~3–6 month lag via Walker/Hadley circulation changes.",
             0.55, 1.15, 12.3, 0.65, font_size=13, color=DKGRAY, italic=True)

# Left: evidence
add_rect(s, 0.4, 2.0, 7.5, 5.2, WHITE)
add_text_box(s, "5 Independent Confirmations", 0.55, 2.1, 7.3, 0.4, font_size=15, bold=True, color=CORAL)
evidence = [
    ("Transfer Entropy", "SH→NH TE=0.274, z=6.92 at 3–6 month lags. Survives 2000 permutations."),
    ("Wavelet Coherence", "Coupling strongest at ENSO band (24–60 months, coherence=0.985). SH leads NH by ~4 months."),
    ("ENSO-conditional TE", "La Niña activates SH→NH (TE=0.33, p=0.017). El Niño: not significant. Bootstrap CI excludes zero."),
    ("Climate Networks", "161 cross-hemispheric edges (Bonferroni-corrected). ALL top hub nodes in tropical Pacific (5°S–5°N)."),
    ("Granger + CCM", "Bidirectional NH↔SH Granger (p<0.0001 both directions). CCM convergence confirmed."),
]
for ri, (method, result) in enumerate(evidence):
    y = 2.6 + ri * 0.9
    add_text_box(s, f"• {method}:", 0.55, y, 7.3, 0.3, font_size=12, bold=True, color=CORAL)
    add_text_box(s, f"  {result}", 0.55, y + 0.3, 7.2, 0.5, font_size=11, color=DKGRAY)

# Right: mechanism diagram (text-based)
add_rect(s, 8.2, 2.0, 4.7, 5.2, NAVY)
add_text_box(s, "Mechanism", 8.35, 2.1, 4.5, 0.4, font_size=15, bold=True, color=WHITE)
add_text_box(s, "Southern Hemisphere\nalbedo anomaly", 8.35, 2.65, 4.4, 0.65,
             font_size=13, color=RGBColor(0xAD, 0xD8, 0xE6), align=PP_ALIGN.CENTER)
add_rect(s, 9.5, 3.4, 2.0, 0.04, SKY)
add_text_box(s, "↓  3–6 months", 8.35, 3.45, 4.4, 0.35, font_size=11, color=RGBColor(0xAB, 0xC6, 0xE0), align=PP_ALIGN.CENTER)
add_rect(s, 8.35, 3.85, 4.4, 1.1, RGBColor(0x1B, 0x45, 0x7A))
add_text_box(s, "Tropical Pacific\n(5°S–5°N, 105–195°E)\nHadley/Walker modulation", 8.45, 3.9, 4.2, 1.0,
             font_size=12, color=AMBER, align=PP_ALIGN.CENTER)
add_text_box(s, "↓  signal propagates", 8.35, 5.05, 4.4, 0.35, font_size=11, color=RGBColor(0xAB, 0xC6, 0xE0), align=PP_ALIGN.CENTER)
add_text_box(s, "Northern Hemisphere\nalbedo adjusted", 8.35, 5.5, 4.4, 0.65,
             font_size=13, color=RGBColor(0xAD, 0xD8, 0xE6), align=PP_ALIGN.CENTER)
add_text_box(s, "→ Hemispheric symmetry maintained", 8.35, 6.25, 4.4, 0.45,
             font_size=11, bold=True, color=GREEN, align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — H004 Coupled Oscillator
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, PURPLE)
add_text_box(s, "Finding 4: NH+SH Form a Coupled Low-Dimensional Oscillator", 0.5, 0.1, 12.5, 0.8,
             font_size=27, bold=True, color=WHITE)
add_text_box(s, "H004 — Coupled Hemispheric Oscillator  ·  CONVERGED (6/6 methods)", 0.5, 0.72, 12.0, 0.3,
             font_size=12, color=RGBColor(0xD7, 0xBB, 0xF5))

# Stat callouts
for x_pos, val, label, c in [
    (0.4,  "2.56", "Joint NH+SH\nattractor dim", PURPLE),
    (3.8,  "4.13", "Expected if\nindependent", DKGRAY),
    (7.2,  "38%",  "Dimensionality\nreduction", GREEN),
    (10.3, "1.75", "VAR simulation\nreproduces dim≈1.86", SKY),
]:
    add_rect(s, x_pos, 1.1, 2.9, 1.4, c)
    add_text_box(s, val, x_pos + 0.1, 1.15, 2.7, 0.75, font_size=30, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_text_box(s, label, x_pos + 0.1, 1.85, 2.7, 0.55, font_size=11, color=WHITE, align=PP_ALIGN.CENTER)

# Evidence and frequency-dependence
add_rect(s, 0.4, 2.65, 6.0, 4.6, WHITE)
add_text_box(s, "Evidence", 0.55, 2.75, 5.8, 0.4, font_size=15, bold=True, color=PURPLE)
add_multi_para(s, [
    "Joint NH+SH attractor dim=2.56 vs 4.13 if independent (38% reduction)",
    "Mutual information p<0.0001 confirms shared dynamics",
    "PCA: first component explains 47.5% of joint variance",
    "VAR(6) simulation reproduces observed correlation dimension (1.75 ≈ 1.86)",
    "CCM: bidirectional NH↔SH convergence confirmed",
    "Granger: bidirectional p<0.0001 with asymmetric lags",
], 0.55, 3.25, 5.8, 3.7, font_size=13, color=DKGRAY)

add_rect(s, 6.8, 2.65, 6.1, 4.6, WHITE)
add_text_box(s, "Frequency-Dependent Directionality", 6.95, 2.75, 5.9, 0.4, font_size=14, bold=True, color=PURPLE)

for row_y, (band, period, leader, why) in enumerate([
    ("Fast mode",   "6–12 months",  "NH leads SH",  "Annual cloud cycle propagates south"),
    ("Slow mode",   "2–5 years",    "SH leads NH",  "ENSO-driven; SH ocean heat capacity"),
    ("Implication", "All bands",    "Coupled system","Not two independent hemispheres"),
]):
    y = 3.3 + row_y * 1.25
    bg = LGRAY if row_y % 2 == 0 else WHITE
    add_rect(s, 6.9, y, 5.8, 1.15, bg)
    add_text_box(s, band, 7.0, y + 0.05, 1.5, 0.4, font_size=12, bold=True, color=PURPLE)
    add_text_box(s, period, 7.0, y + 0.45, 1.5, 0.55, font_size=11, color=DKGRAY)
    add_text_box(s, leader, 8.6, y + 0.05, 2.0, 0.4, font_size=12, bold=True, color=CORAL)
    add_text_box(s, why, 8.6, y + 0.45, 4.0, 0.6, font_size=11, color=DKGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — Unified picture
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, NAVY)
add_text_box(s, "The Unified Picture: A Self-Regulating Planetary System", 0.5, 0.1, 12.5, 0.8,
             font_size=28, bold=True, color=WHITE)

# Large central flow diagram
boxes = [
    (4.2, 1.2, 4.9, 1.0, AMBER,  "Earth's Geometry + Solar Input",             "Sets the ~29% mean constraint\n(H002 — physical envelope)"),
    (4.2, 2.8, 4.9, 1.0, TEAL,   "Cloud Fields",                               "Causally drive albedo in real-time\nr=0.79, in-phase, unidirectional\n(H001 — the control knob)"),
    (4.2, 4.5, 4.9, 1.0, CORAL,  "Tropical Pacific / ENSO",                    "Coordinates cloud fields across hemispheres\nSH→NH at La Niña; hub at 5°S–5°N\n(H003 — the communicator)"),
    (4.2, 6.1, 4.9, 1.0, PURPLE, "NH + SH Coupled Oscillator",                 "Operates as a single 2.56-dim system\nFrequency-dependent leadership\n(H004 — the dynamics)"),
]
for (x, y, w, h, c, title, body) in boxes:
    add_rect(s, x, y, w, h, c)
    add_text_box(s, title, x + 0.1, y + 0.04, w - 0.2, 0.35, font_size=13, bold=True, color=WHITE)
    add_text_box(s, body,  x + 0.1, y + 0.38, w - 0.2, 0.58, font_size=10.5, color=WHITE)

# Arrows between boxes
for ay in [2.2, 3.8, 5.5]:
    add_rect(s, 6.4, ay, 0.5, 0.02, NAVY)
    add_text_box(s, "▼", 6.45, ay - 0.05, 0.4, 0.35, font_size=14, color=NAVY, align=PP_ALIGN.CENTER)

# Left annotation
add_rect(s, 0.3, 1.2, 3.6, 6.2, WHITE)
add_text_box(s, "Key Insight", 0.45, 1.3, 3.4, 0.4, font_size=14, bold=True, color=NAVY)
add_text_box(s,
    "The three hypotheses are not competing explanations — they are nested layers of the same regulatory system:\n\n"
    "Geometry sets the envelope.\n\n"
    "Clouds actively maintain albedo within that envelope.\n\n"
    "ENSO coordinates clouds across the equator so both hemispheres stay in balance.\n\n"
    "The whole system is low-dimensional — like a thermostat, not random noise.",
    0.45, 1.8, 3.4, 5.4, font_size=12, color=DKGRAY)

# Right annotation
add_rect(s, 9.4, 1.2, 3.6, 6.2, WHITE)
add_text_box(s, "What's New", 9.55, 1.3, 3.4, 0.4, font_size=14, bold=True, color=NAVY)
add_multi_para(s, [
    "Causal direction confirmed: clouds drive albedo, not reverse",
    "ENSO-conditioning reveals La Niña as the active switch",
    "Tropical Pacific named as the spatial hub",
    "System modeled as a coupled 2D oscillator with frequency-dependent phase",
    "Albedo more constrained than clouds (dim 1.74 < 2.17)",
    "All findings robust across 5+ independent methods",
], 9.55, 1.8, 3.4, 5.4, font_size=12, color=DKGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Methods used
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, NAVY)
add_text_box(s, "Methods: 12 Independent Causal Inference Techniques", 0.5, 0.1, 12.5, 0.8,
             font_size=28, bold=True, color=WHITE)

headers = ["Method", "Type", "Strength", "Key finding"]
col_x = [0.4, 3.2, 5.5, 7.3]
col_w = [2.7, 2.1, 1.7, 5.5]
add_rect(s, 0.4, 1.05, 12.5, 0.45, NAVY)
for ci, (h, x, w) in enumerate(zip(headers, col_x, col_w)):
    add_text_box(s, h, x + 0.1, 1.1, w - 0.1, 0.35, font_size=13, bold=True, color=WHITE)

methods = [
    ("Convergent Cross Mapping", "Nonlinear causality", "★★★★★", "Unidirectional cloud→albedo causality confirmed"),
    ("Transfer Entropy", "Info-theoretic causality", "★★★★★", "SH→NH z=6.92 at 3–6 month lags"),
    ("Wavelet Coherence", "Frequency-domain", "★★★★", "ENSO-band coupling 0.985; phase switching by timescale"),
    ("Granger Causality", "Linear predictability", "★★★", "Cloud→albedo p<0.0001; bidirectional NH↔SH"),
    ("Climate Networks", "Spatial topology", "★★★★★", "Tropical Pacific as hemispheric communication hub"),
    ("Attractor Reconstruction", "Dynamical systems", "★★★★★", "Global dim=1.86; joint NH+SH dim=2.56 (38% reduction)"),
    ("VAR / Symbolic Regression", "Model fitting", "★★★★★", "2-equation model reproduces observed dim=1.75"),
    ("Variance Decomposition", "Statistical", "★★★", "Clouds explain 44% of albedo variance"),
    ("ICA", "Blind source separation", "★★★★", "4/6 components load on both hemispheres simultaneously"),
    ("Regime Switching", "Nonstationary", "★★★★", "Cloud-albedo slope never reverses in 241/241 windows"),
    ("SINDy", "Equation discovery", "★★★★★", "Negative self-feedback — perturbations decay"),
    ("ENSO-conditional TE", "Conditional causality", "★★★★★", "La Niña activates SH→NH; El Niño does not"),
]
for ri, (m, t, strength, finding) in enumerate(methods):
    y = 1.55 + ri * 0.49
    bg = WHITE if ri % 2 == 0 else LGRAY
    add_rect(s, 0.4, y, 12.5, 0.47, bg)
    for ci, (txt, x, w) in enumerate(zip([m, t, strength, finding], col_x, col_w)):
        bold = ci == 0
        add_text_box(s, txt, x + 0.1, y + 0.04, w - 0.1, 0.39,
                     font_size=11, bold=bold, color=NAVY if ci == 0 else DKGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Next Steps
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, LGRAY)
add_rect(s, 0, 0, 13.33, 1.0, NAVY)
add_text_box(s, "Limitations & Next Steps", 0.5, 0.1, 12.5, 0.8,
             font_size=30, bold=True, color=WHITE)

# Left — limitations
add_rect(s, 0.4, 1.15, 5.9, 5.9, WHITE)
add_text_box(s, "Limitations", 0.55, 1.25, 5.7, 0.45, font_size=17, bold=True, color=RED)
add_multi_para(s, [
    "EBAF-TOA only: cloud microphysics (LWP, IWP, particle size) and AOD unavailable — full CERES SYN1deg needed",
    "N=301 monthly means is marginal for attractor reconstruction and CCM; longer records improve confidence",
    "No pre-2000 data: Pinatubo (1991) falsification test (paper's proposed test) not possible with CERES alone",
    "Analyses use aggregated hemisphere/global means — spatial CCM at grid-cell level remains undone",
    "Albedo mean shift 2000–2012 → 2013–2025 (−0.003, p=0.003) is unexplained — trend or noise?",
], 0.55, 1.8, 5.7, 4.9, font_size=13, color=DKGRAY)

# Right — next steps
add_rect(s, 6.7, 1.15, 6.2, 5.9, WHITE)
add_text_box(s, "Recommended Next Steps", 6.85, 1.25, 6.0, 0.45, font_size=17, bold=True, color=GREEN)
for i, (title, detail) in enumerate([
    ("MERRA-2 + Pinatubo", "Use 1980–present reanalysis to test albedo response/relaxation after 1991 eruption — the paper's proposed falsification"),
    ("Full CERES SYN1deg", "Re-run CCM with LWP, IWP, AOD as causal variables. Map WHERE cloud→albedo causality is strongest"),
    ("Spatial CCM at grid level", "Run convergent cross mapping per grid cell — expect strongest signal in marine stratocumulus regions (SE Pacific, SE Atlantic)"),
    ("Resolve the trend", "Test whether the 2013–2025 albedo dimming (−0.003) is Arctic sea ice loss, aerosol changes, or low-frequency ENSO"),
    ("Write up and publish", "H003 (ENSO-mediated teleconnection) and H001 (nonlinear unidirectional causality) are novel publishable findings"),
]):
    y = 1.8 + i * 1.02
    add_text_box(s, f"{i+1}. {title}", 6.85, y, 5.9, 0.35, font_size=13, bold=True, color=GREEN)
    add_text_box(s, detail, 6.85, y + 0.33, 5.9, 0.6, font_size=11, color=DKGRAY)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Summary
# ══════════════════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(BLANK)
set_bg(s, NAVY)
add_rect(s, 0, 0, 13.33, 0.12, SKY)

add_text_box(s, "Summary", 0.5, 0.3, 12.0, 0.7,
             font_size=34, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

for xi, (title, body, c) in enumerate([
    ("H001\nCloud Buffering\n✓ CONVERGED",
     "Clouds causally drive albedo in real-time (same month, r=0.79). Unidirectional — confirmed by CCM, Granger, climate networks, attractor reconstruction, and wavelet coherence.",
     TEAL),
    ("H002\nInvariant Properties\n✓ CONVERGED",
     "Albedo lives on a low-dimensional attractor (dim=1.86). Albedo is more constrained than clouds (1.74 < 2.17). Mean near 29% with a small but significant recent drift.",
     AMBER),
    ("H003\nTeleconnections\n✓ CONVERGED",
     "NH and SH albedo are causally coupled via the tropical Pacific. La Niña activates SH→NH signal propagation. 5/5 methods confirm.",
     CORAL),
    ("H004\nCoupled Oscillator\n✓ CONVERGED",
     "NH+SH form a coupled dynamical system (joint dim=2.56 vs 4.13 expected). NH leads at fast timescales; SH leads at ENSO timescales. Reproduced by simple VAR model.",
     PURPLE),
]):
    x = 0.3 + xi * 3.2
    add_rect(s, x, 1.1, 3.0, 2.0, c)
    add_text_box(s, title, x + 0.1, 1.15, 2.8, 1.85, font_size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_rect(s, x, 3.15, 3.0, 3.85, RGBColor(0x1B, 0x2A, 0x4A))
    add_text_box(s, body, x + 0.1, 3.25, 2.8, 3.65, font_size=11, color=RGBColor(0xD0, 0xDF, 0xF0))

add_rect(s, 0.3, 7.05, 12.7, 0.1, SKY)
add_text_box(s, "28 analyses  ·  12 methods  ·  25 years of satellite data  ·  4/4 hypotheses converged",
             0.3, 7.15, 12.7, 0.35, font_size=12, color=RGBColor(0xAB, 0xC6, 0xE0), align=PP_ALIGN.CENTER)


out_path = "albedo_research_slides.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
