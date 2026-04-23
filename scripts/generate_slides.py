#!/usr/bin/env python3
"""
Generate a PPTX presentation from the world model state.

Reads hypotheses.json, findings, cycle summaries, and generation log
to produce a slide deck summarizing the agentic discovery results.

The output format matches the "Agentic Discovery: AI Climate Scientist"
presentation style: dark navy title slides, light content slides,
teal/orange accents, two-column cards, status badges.

Usage:
    python scripts/generate_slides.py
    python scripts/generate_slides.py --output results.pptx
"""

import json
import sys
from pathlib import Path
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

sys.path.insert(0, str(Path(__file__).parent))

from utils import WORLD_MODEL_DIR, load_hypotheses, load_exploration_map, _load_meta
from generators import load_all_findings, load_generation_log

# ═══════════════════════════════════════════════════════════════════════
# Color Palette (matching the reference slides)
# ═══════════════════════════════════════════════════════════════════════

DARK_NAVY = RGBColor(0x1B, 0x2A, 0x4A)
TEAL = RGBColor(0x2E, 0xC4, 0xB6)
ORANGE = RGBColor(0xE8, 0x91, 0x3A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
MID_GRAY = RGBColor(0x99, 0x99, 0x99)
DARK_TEXT = RGBColor(0x1B, 0x2A, 0x4A)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xE7, 0x4C, 0x3C)

SLIDE_WIDTH = Inches(10)
SLIDE_HEIGHT = Inches(5.625)  # 16:9


# ═══════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════

def add_text_box(slide, left, top, width, height, text, font_size=14,
                 bold=False, italic=False, color=DARK_TEXT, alignment=PP_ALIGN.LEFT,
                 font_name="Calibri"):
    """Add a text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.italic = italic
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    return txBox


def add_dark_slide(prs):
    """Add a slide with dark navy background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = DARK_NAVY

    # Teal left accent bar
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                                    Inches(0.15), SLIDE_HEIGHT)
    shape.fill.solid()
    shape.fill.fore_color.rgb = TEAL
    shape.line.fill.background()

    return slide


def add_light_slide(prs):
    """Add a slide with light background."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    return slide


def add_card(slide, left, top, width, height, accent_color=TEAL, bg_color=None):
    """Add a card shape with colored top border."""
    # Background
    card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    if bg_color:
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
    else:
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = RGBColor(0xE0, 0xE0, 0xE0)
    card.line.width = Pt(0.5)

    # Top accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top,
                                  width, Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent_color
    bar.line.fill.background()

    return card


def add_badge(slide, left, top, text, color=TEAL):
    """Add a status badge (rounded rectangle with text)."""
    badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                    left, top, Inches(2.2), Inches(0.35))
    badge.fill.solid()
    badge.fill.fore_color.rgb = color
    badge.line.fill.background()

    tf = badge.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = "Calibri"
    p.alignment = PP_ALIGN.CENTER

    return badge


def status_color(status):
    """Get color for hypothesis status."""
    s = status.lower()
    if s in ("supported", "converged"):
        return TEAL
    elif s in ("refuted", "invalidated"):
        return RED
    elif s == "inconclusive":
        return ORANGE
    elif s == "testing":
        return MID_GRAY
    return DARK_TEXT


def status_label(hyp):
    """Get display label for hypothesis status."""
    s = hyp.get("status", "testing").lower()
    conf = hyp.get("confidence", 0)
    cm = hyp.get("convergence_metrics", {})
    confirming = cm.get("methods_confirming", 0)

    if s in ("supported", "converged"):
        return f"SUPPORTED  {conf:.2f}"
    elif s == "refuted":
        return f"REFUTED  {conf:.2f}"
    elif s == "testing":
        return f"TESTING  {confirming}/{cm.get('min_methods_for_convergence', 5)}"
    return f"INCONCLUSIVE  {conf:.2f}"


# ═══════════════════════════════════════════════════════════════════════
# Slide Builders (each reads from world model dynamically)
# ═══════════════════════════════════════════════════════════════════════

def build_title_slide(prs, meta):
    """Slide 1: Title slide."""
    slide = add_dark_slide(prs)

    add_text_box(slide, Inches(0.7), Inches(0.8), Inches(8), Inches(1.2),
                 "Agentic Discovery", font_size=54, bold=True, color=WHITE,
                 font_name="Georgia")

    add_text_box(slide, Inches(0.7), Inches(2.0), Inches(6), Inches(0.5),
                 "An AI Climate Scientist", font_size=22, color=TEAL,
                 font_name="Calibri")

    # Teal divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                   Inches(0.7), Inches(2.8), Inches(2), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()

    cycle_count = meta.get("current_cycle", 0)
    findings_count = meta.get("total_findings", 0)

    add_text_box(slide, Inches(0.7), Inches(3.2), Inches(6), Inches(0.4),
                 f"Proof-of-Concept: Hemispheric Symmetry Experiment",
                 font_size=14, color=WHITE)

    add_text_box(slide, Inches(0.7), Inches(3.6), Inches(6), Inches(0.3),
                 "Feldman, Gristey, Hakuba, Hellinger, Kahn",
                 font_size=12, italic=True, color=MID_GRAY)


def build_motivation_slide(prs, hypotheses):
    """Slide 2: Why an AI Climate Scientist."""
    slide = add_light_slide(prs)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(9), Inches(0.8),
                 "Why an AI Climate Scientist?", font_size=40, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    # Left card: What was found
    add_card(slide, Inches(0.5), Inches(1.3), Inches(4.2), Inches(3.2), accent_color=TEAL)
    add_text_box(slide, Inches(0.7), Inches(1.5), Inches(3.8), Inches(0.4),
                 "Feldman et al. 2026 found", font_size=18, bold=True, color=DARK_NAVY)

    found_items = [
        "Earth's albedo is remarkably stable (~29%)",
        "NH ~ SH despite radically different surfaces",
        "Cloud properties dominate (SHAP analysis)",
        "Used: FFNN, U-Net, Prophet, ExtraTrees",
    ]
    y = 2.1
    for item in found_items:
        add_text_box(slide, Inches(0.9), Inches(y), Inches(3.5), Inches(0.35),
                     f"  {item}", font_size=11, color=DARK_TEXT)
        y += 0.35

    # Right card: What was missing
    add_card(slide, Inches(5.3), Inches(1.3), Inches(4.2), Inches(3.2), accent_color=ORANGE)
    add_text_box(slide, Inches(5.5), Inches(1.5), Inches(3.8), Inches(0.4),
                 "What Was Missing", font_size=18, bold=True, color=ORANGE)

    missing_items = [
        "No causal inference (only correlational)",
        "No information-theoretic analysis",
        "No spectral / frequency-domain analysis",
        "No dynamical systems or equation discovery",
        "No graph / network approaches",
    ]
    y = 2.1
    for item in missing_items:
        add_text_box(slide, Inches(5.7), Inches(y), Inches(3.5), Inches(0.35),
                     f"  {item}", font_size=11, color=DARK_TEXT)
        y += 0.35

    # Bottom goal text
    add_text_box(slide, Inches(0.5), Inches(4.8), Inches(9), Inches(0.4),
                 "Goal: Autonomously discover novel methodologies the research team hadn't considered.",
                 font_size=12, italic=True, color=TEAL)


def build_architecture_slide(prs, meta):
    """Slide 3: System Architecture."""
    slide = add_light_slide(prs)

    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(9), Inches(0.7),
                 "System Architecture", font_size=40, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    add_text_box(slide, Inches(0.5), Inches(0.85), Inches(8), Inches(0.3),
                 "Inspired by Kosmos (Lu et al. 2025) + MCTS planning",
                 font_size=11, italic=True, color=MID_GRAY)

    # Architecture boxes
    boxes = [
        ("World Model", "Hypotheses, exploration\nmap, literature, prior work"),
        ("MCTS\nSelection", "UCB1 + hypothesis\nbalance scoring"),
        ("Analysis\nGeneration", "Novel script + execution\non CERES data"),
        ("Synthesis &\nConvergence", "Update beliefs, check\nmulti-method convergence"),
    ]

    x = 0.3
    for title, desc in boxes:
        card = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                       Inches(x), Inches(1.5), Inches(2.1), Inches(1.3))
        card.fill.solid()
        card.fill.fore_color.rgb = DARK_NAVY
        card.line.fill.background()

        add_text_box(slide, Inches(x + 0.1), Inches(1.55), Inches(1.9), Inches(0.5),
                     title, font_size=13, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER)
        add_text_box(slide, Inches(x + 0.1), Inches(2.1), Inches(1.9), Inches(0.6),
                     desc, font_size=9, color=RGBColor(0xCC, 0xCC, 0xCC),
                     alignment=PP_ALIGN.CENTER)

        # Arrow between boxes (except last)
        if title != boxes[-1][0]:
            arrow_x = x + 2.15
            add_text_box(slide, Inches(arrow_x), Inches(1.9), Inches(0.3), Inches(0.4),
                         "→", font_size=20, color=ORANGE, alignment=PP_ALIGN.CENTER)
        x += 2.4

    # Stats bar
    cycle_count = meta.get("current_cycle", 0)
    findings = load_all_findings()
    methods_used = set()
    for f in findings:
        m = f.get("method", "")
        if m and "adversarial" not in m:
            methods_used.add(m)

    gen_log = load_generation_log()
    adversarial_count = len(gen_log.get("adversarial_events", []))

    stats = [
        (str(cycle_count), "Cycles"),
        (str(len(findings)), "Analyses"),
        (str(len(methods_used)), "Novel Methods"),
        (str(adversarial_count), "Adversarial\nChallenges"),
    ]

    x = 0.5
    for number, label in stats:
        add_text_box(slide, Inches(x), Inches(3.6), Inches(2), Inches(0.7),
                     number, font_size=42, bold=True, color=TEAL,
                     alignment=PP_ALIGN.CENTER, font_name="Georgia")
        add_text_box(slide, Inches(x), Inches(4.3), Inches(2), Inches(0.5),
                     label, font_size=11, color=MID_GRAY,
                     alignment=PP_ALIGN.CENTER)
        x += 2.3


def build_data_slide(prs, meta):
    """Slide 4: Data Sources."""
    slide = add_light_slide(prs)

    add_text_box(slide, Inches(0.5), Inches(0.3), Inches(9), Inches(0.7),
                 "Data Sources", font_size=40, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    # Primary data card
    add_card(slide, Inches(0.5), Inches(1.2), Inches(9), Inches(2.5))

    add_badge(slide, Inches(0.7), Inches(1.35), "PRIMARY", TEAL)

    add_text_box(slide, Inches(0.7), Inches(1.85), Inches(5), Inches(0.5),
                 "CERES EBAF-TOA Edition 4.2.1", font_size=24, bold=True,
                 color=DARK_NAVY, font_name="Georgia")
    add_text_box(slide, Inches(0.7), Inches(2.35), Inches(5), Inches(0.3),
                 "NASA's Clouds and Earth's Radiant Energy System",
                 font_size=10, italic=True, color=MID_GRAY)

    details = [
        ("Coverage:", "March 2000 - March 2025 (301 months)"),
        ("Resolution:", "1 x 1 latitude-longitude grid"),
        ("Records:", "14.3 million rows"),
        ("Filtering:", "Terminator filter (|lat| < 66)"),
    ]
    y = 2.7
    for label, value in details:
        add_text_box(slide, Inches(0.7), Inches(y), Inches(1.2), Inches(0.25),
                     label, font_size=10, bold=True, color=TEAL)
        add_text_box(slide, Inches(1.9), Inches(y), Inches(3), Inches(0.25),
                     value, font_size=10, color=DARK_TEXT)
        y += 0.25

    # Key variables on right side
    add_text_box(slide, Inches(5.5), Inches(1.85), Inches(3.5), Inches(0.3),
                 "Key variables used across analyses:", font_size=11, bold=True, color=DARK_TEXT)
    add_text_box(slide, Inches(5.5), Inches(2.2), Inches(3.8), Inches(1.2),
                 "TOA shortwave albedo, cloud area fraction (CAF), cloud optical depth (tau), "
                 "TOA SW/LW fluxes (clear-sky and all-sky), solar irradiance, latitude, longitude, month",
                 font_size=10, color=DARK_TEXT)


def build_methodology_slide(prs, hypotheses, findings):
    """Slide 5: Novel Methodologies table."""
    slide = add_light_slide(prs)

    hyp_list = hypotheses.get("hypotheses", [])
    n_methods = len(set(f.get("method", "") for f in findings if f.get("method")))

    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(6), Inches(0.7),
                 f"{n_methods} Novel Methodologies", font_size=40, bold=True,
                 color=DARK_NAVY, font_name="Georgia")
    add_text_box(slide, Inches(0.5), Inches(0.85), Inches(7), Inches(0.3),
                 "None used in the original Nature paper", font_size=11,
                 italic=True, color=MID_GRAY)

    # Build methods table from findings
    methods_seen = {}
    for f in findings:
        method = f.get("method", "")
        if not method or "adversarial" in method:
            continue
        if method not in methods_seen:
            hyp_id = f.get("hypothesis_id", "")
            methods_seen[method] = hyp_id

    # Table
    y = 1.3
    # Header
    add_text_box(slide, Inches(0.5), Inches(y), Inches(2.2), Inches(0.3),
                 "Method", font_size=11, bold=True, color=WHITE)
    add_text_box(slide, Inches(2.7), Inches(y), Inches(1.5), Inches(0.3),
                 "Hypothesis", font_size=11, bold=True, color=WHITE)

    header_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(0.5), Inches(y), Inches(3.7), Inches(0.3))
    header_bg.fill.solid()
    header_bg.fill.fore_color.rgb = TEAL
    header_bg.line.fill.background()
    # Re-add text on top
    add_text_box(slide, Inches(0.5), Inches(y), Inches(2.2), Inches(0.3),
                 "Method", font_size=11, bold=True, color=WHITE)
    add_text_box(slide, Inches(2.7), Inches(y), Inches(1.5), Inches(0.3),
                 "Hypothesis", font_size=11, bold=True, color=WHITE)

    y += 0.35
    for method, hyp_id in list(methods_seen.items())[:12]:
        display_name = method.replace("_", " ").title()
        add_text_box(slide, Inches(0.5), Inches(y), Inches(2.2), Inches(0.25),
                     display_name, font_size=10, color=DARK_TEXT)
        add_text_box(slide, Inches(2.7), Inches(y), Inches(1.5), Inches(0.25),
                     hyp_id or "—", font_size=10, bold=True, color=TEAL)
        y += 0.28

    # Hypothesis legend on right
    x_legend = 5.5
    y_legend = 1.3
    for hyp in hyp_list[:4]:
        add_card(slide, Inches(x_legend), Inches(y_legend), Inches(4), Inches(0.8),
                 accent_color=status_color(hyp.get("status", "testing")))
        add_text_box(slide, Inches(x_legend + 0.15), Inches(y_legend + 0.1),
                     Inches(3.7), Inches(0.3),
                     f"{hyp['id']}   {hyp['name']}", font_size=13, bold=True, color=DARK_NAVY)
        # Short description
        stmt = hyp.get("statement", "")[:80]
        add_text_box(slide, Inches(x_legend + 0.15), Inches(y_legend + 0.45),
                     Inches(3.7), Inches(0.3),
                     stmt, font_size=9, color=MID_GRAY)
        y_legend += 1.0


def build_discovery_slide(prs, hyp, findings, discovery_num):
    """Build a discovery slide for a hypothesis."""
    slide = add_light_slide(prs)

    h_id = hyp["id"]
    cm = hyp.get("convergence_metrics", {})
    confirming = cm.get("methods_confirming", 0)

    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(7), Inches(0.8),
                 f"Discovery {discovery_num}: {hyp['name']}", font_size=32, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    # Status badge
    add_badge(slide, Inches(7.3), Inches(0.3), status_label(hyp),
              status_color(hyp.get("status", "testing")))

    # Supporting evidence list
    supporting_ids = hyp.get("supporting_evidence", [])
    h_findings = [f for f in findings if f.get("finding_id") in supporting_ids]

    add_text_box(slide, Inches(0.5), Inches(1.1), Inches(5.5), Inches(0.3),
                 f"{len(h_findings)} independent methods confirm:", font_size=12, color=DARK_TEXT)

    y = 1.5
    for f in h_findings[:6]:
        method = f.get("method", "unknown").replace("_", " ").title()
        summary = f.get("summary", "")[:120]
        stats = f.get("statistics", {})
        p_val = stats.get("p_value", "")
        p_str = f" (p={p_val})" if p_val and isinstance(p_val, (int, float)) and p_val < 1 else ""

        add_text_box(slide, Inches(0.5), Inches(y), Inches(5.5), Inches(0.2),
                     f"{method}{p_str}", font_size=11, bold=True, color=TEAL)
        add_text_box(slide, Inches(0.5), Inches(y + 0.22), Inches(5.5), Inches(0.3),
                     summary, font_size=9, color=MID_GRAY)
        y += 0.55

    # Adversarial note at bottom
    gen_log = load_generation_log()
    adv_events = [e for e in gen_log.get("adversarial_events", []) if e.get("hypothesis_id") == h_id]
    if adv_events:
        survived = sum(1 for e in adv_events if e.get("verdict") == "survives")
        invalidated = sum(1 for e in adv_events if e.get("verdict") == "invalidated")
        add_text_box(slide, Inches(0.5), Inches(5.0), Inches(8), Inches(0.3),
                     f"Adversarial: {survived} survived, {invalidated} invalidated",
                     font_size=10, italic=True, color=MID_GRAY)


def build_adversarial_slide(prs, hypotheses, findings):
    """Build adversarial self-challenge slide."""
    slide = add_light_slide(prs)

    gen_log = load_generation_log()
    adv_events = gen_log.get("adversarial_events", [])

    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(9), Inches(0.7),
                 "Adversarial Self-Challenge", font_size=40, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    n_invalidated = sum(1 for e in adv_events if e.get("verdict") == "invalidated")
    add_text_box(slide, Inches(0.5), Inches(0.85), Inches(8), Inches(0.3),
                 f"{len(adv_events)} adversarial tests — {n_invalidated} methods invalidated",
                 font_size=11, italic=True, color=MID_GRAY)

    # Group by hypothesis
    by_hyp = {}
    for e in adv_events:
        by_hyp.setdefault(e.get("hypothesis_id", "?"), []).append(e)

    x = 0.3
    for h_id, events in list(by_hyp.items())[:3]:
        hyp = next((h for h in hypotheses.get("hypotheses", []) if h["id"] == h_id), {})
        h_name = hyp.get("name", h_id)

        survived = sum(1 for e in events if e.get("verdict") == "survives")
        invalidated = sum(1 for e in events if e.get("verdict") == "invalidated")
        weakened = sum(1 for e in events if e.get("verdict") == "weakened")

        if invalidated > 0:
            verdict_text = f"{invalidated} INVALIDATED"
            verdict_color = ORANGE
        elif weakened > 0:
            verdict_text = "WEAKENED"
            verdict_color = ORANGE
        else:
            verdict_text = "ROBUST"
            verdict_color = GREEN

        add_card(slide, Inches(x), Inches(1.4), Inches(3), Inches(3.5),
                 accent_color=status_color(hyp.get("status", "testing")))

        add_text_box(slide, Inches(x + 0.15), Inches(1.6), Inches(2.7), Inches(0.3),
                     f"{h_id} Challenge", font_size=14, bold=True, color=DARK_NAVY)

        add_badge(slide, Inches(x + 0.15), Inches(2.0), verdict_text, verdict_color)

        # Details
        y_detail = 2.6
        for e in events[:4]:
            reason = e.get("reason", "")[:80]
            verdict = e.get("verdict", "?")
            add_text_box(slide, Inches(x + 0.15), Inches(y_detail), Inches(2.7), Inches(0.5),
                         f"{verdict}: {reason}", font_size=9, color=DARK_TEXT)
            y_detail += 0.55

        x += 3.2


def build_unified_mechanism_slide(prs, hypotheses):
    """Build the unified mechanism slide (dark background, nested layers)."""
    slide = add_dark_slide(prs)

    add_text_box(slide, Inches(0.7), Inches(0.3), Inches(8), Inches(0.7),
                 "The Unified Mechanism", font_size=40, bold=True,
                 color=WHITE, font_name="Georgia")

    hyp_list = hypotheses.get("hypotheses", [])
    supported = [h for h in hyp_list if h.get("status") in ("supported", "converged")]

    # Layer boxes (nested)
    layers = []
    for i, hyp in enumerate(supported[:3]):
        layers.append((hyp["name"], hyp.get("statement", "")[:120]))

    y = 1.3
    for i, (name, desc) in enumerate(layers):
        indent = Inches(0.3 * i)
        width = Inches(8.5 - 0.3 * i)

        layer_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                           Inches(0.7) + indent, Inches(y),
                                           width, Inches(1.1))
        if i < 2:
            layer_bg.fill.solid()
            layer_bg.fill.fore_color.rgb = RGBColor(0x25, 0x3A, 0x5E)
        else:
            layer_bg.fill.solid()
            layer_bg.fill.fore_color.rgb = TEAL
        layer_bg.line.fill.background()

        add_text_box(slide, Inches(0.9) + indent, Inches(y + 0.1),
                     Inches(3), Inches(0.3),
                     f"Layer {i+1}: {name}", font_size=16, bold=True,
                     color=WHITE if i < 2 else DARK_NAVY)
        add_text_box(slide, Inches(4.5), Inches(y + 0.1),
                     Inches(4), Inches(0.8),
                     desc, font_size=10,
                     color=RGBColor(0xCC, 0xCC, 0xCC) if i < 2 else DARK_NAVY)

        y += 1.25

    # Bottom summary
    add_text_box(slide, Inches(0.7), Inches(4.8), Inches(8.5), Inches(0.5),
                 "Earth's albedo stays near 29% because physics sets the envelope, "
                 "clouds actively buffer within it, and ENSO keeps the hemispheres synchronized.",
                 font_size=11, italic=True, color=TEAL, alignment=PP_ALIGN.CENTER)


def build_takeaways_slide(prs, hypotheses):
    """Build key takeaways and open questions slide."""
    slide = add_light_slide(prs)

    add_text_box(slide, Inches(0.5), Inches(0.2), Inches(9), Inches(0.7),
                 "Key Takeaways & Open Questions", font_size=36, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    # Results table
    hyp_list = hypotheses.get("hypotheses", [])

    # Header
    header_bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                        Inches(0.5), Inches(1.1), Inches(9), Inches(0.35))
    header_bg.fill.solid()
    header_bg.fill.fore_color.rgb = TEAL
    header_bg.line.fill.background()

    for col, text, w in [(0.5, "Hypothesis", 3), (3.5, "Status", 2),
                          (5.5, "Conf.", 1.2), (6.7, "Methods", 2.5)]:
        add_text_box(slide, Inches(col), Inches(1.12), Inches(w), Inches(0.3),
                     text, font_size=11, bold=True, color=WHITE)

    y = 1.55
    for hyp in hyp_list:
        cm = hyp.get("convergence_metrics", {})
        status = hyp.get("status", "testing").upper()
        conf = hyp.get("confidence", 0)
        confirming = cm.get("methods_confirming", 0)

        add_text_box(slide, Inches(0.5), Inches(y), Inches(3), Inches(0.3),
                     f"{hyp['id']} {hyp['name']}", font_size=11, bold=True, color=DARK_TEXT)
        add_text_box(slide, Inches(3.5), Inches(y), Inches(2), Inches(0.3),
                     status, font_size=11, bold=True, color=status_color(hyp.get("status")))
        add_text_box(slide, Inches(5.5), Inches(y), Inches(1.2), Inches(0.3),
                     f"{conf:.2f}", font_size=11, color=DARK_TEXT)
        add_text_box(slide, Inches(6.7), Inches(y), Inches(2.5), Inches(0.3),
                     f"{confirming} methods", font_size=11, color=DARK_TEXT)
        y += 0.35

    # Open questions
    add_text_box(slide, Inches(0.5), Inches(y + 0.4), Inches(8), Inches(0.5),
                 "Open Questions for Discussion", font_size=22, bold=True,
                 color=DARK_NAVY, font_name="Georgia")

    questions = [
        "Can the ~2D attractor be attributed to specific physical invariants?",
        "What circulation mechanisms transmit SH->NH coupling at 3-month lag?",
        "Will cloud buffering persist under strong anthropogenic forcing?",
        "How does this AI scientist methodology generalize to other Earth system questions?",
    ]

    y_q = y + 1.0
    for q in questions:
        add_text_box(slide, Inches(0.7), Inches(y_q), Inches(8), Inches(0.3),
                     f"  {q}", font_size=11, color=DARK_TEXT)
        y_q += 0.3


def build_appendix_slide(prs):
    """Build appendix divider slide."""
    slide = add_dark_slide(prs)

    add_text_box(slide, Inches(0.7), Inches(1.5), Inches(8), Inches(1),
                 "Appendix", font_size=54, bold=True, color=WHITE,
                 font_name="Georgia")
    add_text_box(slide, Inches(0.7), Inches(2.7), Inches(6), Inches(0.5),
                 "Methodology Details", font_size=22, color=TEAL)
    add_text_box(slide, Inches(0.7), Inches(3.3), Inches(6), Inches(0.3),
                 "One slide per method — how it works, what it tests, key result",
                 font_size=12, color=MID_GRAY)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def generate_presentation(output_path=None):
    """Generate the full presentation from world model state."""
    if output_path is None:
        output_path = Path(__file__).parent.parent / "agentic_discovery_results.pptx"

    # Load world model
    meta = _load_meta()
    hypotheses = load_hypotheses()
    findings = load_all_findings()

    print(f"Generating slides from {meta['current_cycle']} cycles, "
          f"{len(findings)} findings, {len(hypotheses['hypotheses'])} hypotheses...")

    # Create presentation (widescreen 16:9)
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)

    # Build slides
    build_title_slide(prs, meta)
    build_motivation_slide(prs, hypotheses)
    build_architecture_slide(prs, meta)
    build_data_slide(prs, meta)
    build_methodology_slide(prs, hypotheses, findings)

    # Discovery slides — one per hypothesis (sorted by confidence desc)
    sorted_hyps = sorted(hypotheses["hypotheses"],
                         key=lambda h: h.get("confidence", 0), reverse=True)
    for i, hyp in enumerate(sorted_hyps, 1):
        build_discovery_slide(prs, hyp, findings, i)

    # Adversarial slide
    gen_log = load_generation_log()
    if gen_log.get("adversarial_events"):
        build_adversarial_slide(prs, hypotheses, findings)

    # Unified mechanism
    build_unified_mechanism_slide(prs, hypotheses)

    # Takeaways
    build_takeaways_slide(prs, hypotheses)

    # Appendix
    build_appendix_slide(prs)

    # Save
    prs.save(str(output_path))
    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    output = None
    for arg in sys.argv[1:]:
        if arg.startswith("--output"):
            output = sys.argv[sys.argv.index(arg) + 1]
    generate_presentation(output)
