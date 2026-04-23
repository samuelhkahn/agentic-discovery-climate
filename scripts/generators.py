#!/usr/bin/env python3
"""
Dynamic Hypothesis and Method Generation for Agentic Discovery.

Detects triggers from findings that suggest new hypotheses or methods,
builds proposals for the agent to review, and integrates accepted
proposals into the world model (hypotheses.json + exploration_map.json).

Trigger detection is deterministic (no LLM needed).
Generation proposals are structured prompts for the agent.
Integration writes to JSON files.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    WORLD_MODEL_DIR, load_hypotheses, save_hypotheses,
    load_exploration_map, save_exploration_map,
    _load_meta, save_meta, load_literature,
)

GENERATION_LOG_PATH = WORLD_MODEL_DIR / "generation_log.json"
GENERATION_RESPONSES_DIR = WORLD_MODEL_DIR / "generation_responses"

# Keys in exploration_map method entries that are metadata, not hypothesis cells
METADATA_KEYS = {"priority", "differentiation", "agent_generated", "generated_cycle",
                 "generating_trigger", "description"}


# ═══════════════════════════════════════════════════════════════════════
# Utility Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_all_findings() -> list:
    """Load all findings from the world model."""
    findings = []
    findings_dir = WORLD_MODEL_DIR / "findings"
    if not findings_dir.exists():
        return findings
    for cycle_dir in sorted(findings_dir.iterdir()):
        if not cycle_dir.is_dir():
            continue
        for task_dir in sorted(cycle_dir.iterdir()):
            fp = task_dir / "findings.json"
            if fp.exists():
                with open(fp) as f:
                    findings.append(json.load(f))
    return findings


def load_latest_finding():
    """Load the most recent finding."""
    findings = load_all_findings()
    if not findings:
        return None
    return max(findings, key=lambda f: f.get("cycle", 0) * 100 + f.get("task_id", 0))


def load_generation_log() -> dict:
    """Load the generation audit log."""
    if GENERATION_LOG_PATH.exists():
        with open(GENERATION_LOG_PATH) as f:
            return json.load(f)
    return {"hypothesis_generations": [], "method_generations": [], "pruning_events": []}


def save_generation_log(log: dict):
    with open(GENERATION_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def hypothesis_cell_key(hypothesis: dict) -> str:
    """Convert hypothesis dict to exploration_map cell key."""
    name_slug = hypothesis["name"].lower().replace(" ", "_").replace("-", "_")
    return f"{hypothesis['id']}_{name_slug}"


def extract_novel_concepts(text: str, existing_text: str) -> list:
    """Find domain concepts in text that don't appear in existing_text.

    Simple keyword extraction — not NLP-sophisticated, but sufficient
    as a trigger (the agent makes the final judgment).
    """
    # Domain-relevant terms (climate science vocabulary)
    domain_terms = {
        "enso", "la nina", "el nino", "nino", "walker", "hadley", "itcz",
        "amoc", "thermohaline", "pdo", "amo", "nao", "mjo", "monsoon",
        "stratospheric", "tropospheric", "polar vortex", "jet stream",
        "sea ice", "permafrost", "snow cover", "vegetation", "desertification",
        "ozone", "methane", "volcanic", "pinatubo", "aerosol injection",
        "marine stratocumulus", "deep convection", "cirrus", "anvil",
        "shortwave", "longwave", "radiative", "feedback", "sensitivity",
        "tipping point", "bifurcation", "regime shift", "hysteresis",
        "self-organization", "emergent", "criticality", "phase transition",
        "coupled", "decoupled", "nonlinear", "threshold", "saturation",
        "compensation", "overcompensation", "damping", "amplification",
    }

    text_lower = text.lower()
    existing_lower = existing_text.lower()

    novel = []
    for term in domain_terms:
        if term in text_lower and term not in existing_lower:
            novel.append(term)

    return novel


# ═══════════════════════════════════════════════════════════════════════
# Trigger Detection
# ═══════════════════════════════════════════════════════════════════════

def detect_hypothesis_triggers(latest_finding, all_findings, hypotheses, literature) -> list:
    """Detect conditions that warrant proposing a new hypothesis.

    Returns list of trigger dicts, each with: type, finding, detail.
    """
    if latest_finding is None:
        return []

    triggers = []
    hyp_list = hypotheses.get("hypotheses", [])

    # ── TRIGGER 1: Finding contradicts existing hypothesis, supports none ──
    if latest_finding.get("refutes_hypothesis"):
        refuted_id = latest_finding.get("hypothesis_id")
        supports_any = False
        for h in hyp_list:
            if h["id"] != refuted_id:
                if latest_finding["finding_id"] in h.get("supporting_evidence", []):
                    supports_any = True
                    break
        if not supports_any:
            triggers.append({
                "type": "contradiction",
                "finding": latest_finding,
                "detail": (f"Finding {latest_finding['finding_id']} refutes {refuted_id} "
                          f"and supports no other hypothesis — something unexplained is happening")
            })

    # ── TRIGGER 2: High-confidence finding with novel concepts ──
    if latest_finding.get("confidence", 0) >= 0.7:
        all_statements = " ".join(h["statement"].lower() for h in hyp_list)
        novel = extract_novel_concepts(latest_finding.get("summary", ""), all_statements)
        if novel:
            triggers.append({
                "type": "unexpected_pattern",
                "finding": latest_finding,
                "novel_concepts": novel,
                "detail": (f"Finding {latest_finding['finding_id']} mentions "
                          f"{', '.join(novel)} — not covered by any existing hypothesis")
            })

    # ── TRIGGER 3: Adversarial invalidation → refinement opportunity ──
    if "adversarial" in latest_finding.get("method", "").lower():
        if latest_finding.get("refutes_hypothesis") or "INVALIDATED" in latest_finding.get("summary", ""):
            triggers.append({
                "type": "adversarial_refinement",
                "finding": latest_finding,
                "detail": (f"Adversarial finding {latest_finding['finding_id']} invalidated "
                          f"aspect of {latest_finding.get('hypothesis_id', '?')} — "
                          f"a refined sub-hypothesis may be warranted")
            })

    # ── TRIGGER 4: Strong unexpected correlation ──
    stats = latest_finding.get("statistics", {})
    strong_corrs = {k: v for k, v in stats.items()
                    if "corr" in k.lower() and isinstance(v, (int, float)) and abs(v) > 0.5}
    if strong_corrs:
        triggers.append({
            "type": "novel_correlation",
            "finding": latest_finding,
            "correlations": {k: float(v) for k, v in strong_corrs.items()},
            "detail": (f"Strong correlations in {latest_finding['finding_id']}: "
                      f"{strong_corrs} — may indicate an unexplored relationship")
        })

    return triggers


def detect_method_triggers(hypothesis, exploration_map, all_findings) -> list:
    """Detect conditions that warrant proposing a new method for a hypothesis.

    Returns list of trigger dicts.
    """
    triggers = []
    h_id = hypothesis["id"]
    cm = hypothesis.get("convergence_metrics", {})
    h_key_prefix = h_id  # Match cells starting with this ID

    # Count remaining cells for this hypothesis
    remaining = 0
    total_tried = 0
    for tier_key in ["tier_1_methods", "tier_2_methods"]:
        tier = exploration_map.get(tier_key, {})
        for method, mdata in tier.items():
            for key, val in mdata.items():
                if key in METADATA_KEYS:
                    continue
                if not isinstance(val, dict):
                    continue
                if key.startswith(h_key_prefix) or h_id in key:
                    if val.get("status") == "not_started":
                        remaining += 1
                    elif val.get("status") in ("completed", "in_progress"):
                        total_tried += 1

    # ── TRIGGER 1: All methods exhausted, convergence not reached ──
    min_needed = cm.get("min_methods_for_convergence", 5)
    confirming = cm.get("methods_confirming", 0)
    if remaining == 0 and confirming < min_needed:
        triggers.append({
            "type": "coverage_exhausted",
            "hypothesis_id": h_id,
            "methods_tried": total_tried,
            "methods_needed": min_needed,
            "detail": (f"All {total_tried} existing methods tried for {h_id} but only "
                      f"{confirming}/{min_needed} confirming — need novel methods")
        })

    # ── TRIGGER 2: Untested prediction with no applicable method ──
    for pred in hypothesis.get("testable_predictions", []):
        tested = pred.get("tested_by", [])
        untested = pred.get("untested_methods", [])
        if not tested and not untested:
            triggers.append({
                "type": "prediction_gap",
                "hypothesis_id": h_id,
                "prediction": pred,
                "detail": (f"Prediction {pred.get('id', '?')} of {h_id} has no tested "
                          f"or untested methods — needs a custom method")
            })

    # ── TRIGGER 3: Adversarial gap ──
    for f in all_findings:
        if (f.get("hypothesis_id") == h_id
                and "adversarial" in f.get("method", "").lower()
                and f.get("refutes_hypothesis")):
            triggers.append({
                "type": "adversarial_gap",
                "hypothesis_id": h_id,
                "finding_id": f.get("finding_id"),
                "detail": (f"Adversarial finding {f.get('finding_id')} suggests "
                          f"a methodological gap for {h_id}")
            })

    return triggers


# ═══════════════════════════════════════════════════════════════════════
# Proposal Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_hypothesis_proposal(trigger, hypotheses, literature, findings) -> dict:
    """Build a structured hypothesis proposal for the agent to review."""
    existing_ids = [h["id"] for h in hypotheses.get("hypotheses", [])]
    next_num = max((int(h[1:]) for h in existing_ids if h[1:].isdigit()), default=3) + 1
    next_id = f"H{next_num:03d}"

    existing_names = [h["name"] for h in hypotheses.get("hypotheses", [])]

    base = {
        "action": "generate_hypothesis",
        "id": next_id,
        "trigger": {
            "type": trigger["type"],
            "detail": trigger["detail"],
            "finding_id": trigger.get("finding", {}).get("finding_id"),
        },
    }

    if trigger["type"] == "adversarial_refinement":
        parent_id = trigger["finding"].get("hypothesis_id", "?")
        parent = next((h for h in hypotheses["hypotheses"] if h["id"] == parent_id), None)
        base["template"] = "refinement"
        base["parent_hypothesis"] = parent_id
        base["prompt_for_agent"] = (
            f"ADVERSARIAL REFINEMENT needed for {parent_id} "
            f"({parent['name'] if parent else '?'}).\n"
            f"The adversarial finding showed: {trigger['detail']}\n"
            f"Finding summary: {trigger['finding'].get('summary', '')[:300]}\n\n"
            f"Existing hypotheses: {existing_names}\n\n"
            f"Propose a REFINED sub-hypothesis that addresses this gap.\n"
            f"Write to: world_model/generation_responses/{next_id}_proposal.json\n"
            f"Format: {{\"name\": \"...\", \"statement\": \"...\", "
            f"\"testable_predictions\": [{{\"id\": \"P001\", \"prediction\": \"...\"}}, ...]}}"
        )
    elif trigger["type"] == "unexpected_pattern":
        base["template"] = "novel"
        base["prompt_for_agent"] = (
            f"UNEXPECTED PATTERN detected: {trigger.get('novel_concepts', [])}\n"
            f"From finding: {trigger['finding'].get('summary', '')[:300]}\n\n"
            f"Existing hypotheses cover: {existing_names}\n\n"
            f"Propose a NEW hypothesis capturing this pattern.\n"
            f"Write to: world_model/generation_responses/{next_id}_proposal.json\n"
            f"Format: {{\"name\": \"...\", \"statement\": \"...\", "
            f"\"testable_predictions\": [{{\"id\": \"P001\", \"prediction\": \"...\"}}, ...]}}"
        )
    elif trigger["type"] == "contradiction":
        base["template"] = "explanatory"
        base["prompt_for_agent"] = (
            f"CONTRADICTION: {trigger['detail']}\n"
            f"Finding: {trigger['finding'].get('summary', '')[:300]}\n\n"
            f"Existing hypotheses: {existing_names}\n"
            f"None of them explain this finding. Propose a hypothesis that does.\n"
            f"Write to: world_model/generation_responses/{next_id}_proposal.json\n"
            f"Format: {{\"name\": \"...\", \"statement\": \"...\", "
            f"\"testable_predictions\": [{{\"id\": \"P001\", \"prediction\": \"...\"}}, ...]}}"
        )
    elif trigger["type"] == "novel_correlation":
        base["template"] = "correlational"
        base["prompt_for_agent"] = (
            f"STRONG CORRELATION detected: {trigger.get('correlations', {})}\n"
            f"From finding: {trigger['finding'].get('summary', '')[:300]}\n\n"
            f"Existing hypotheses: {existing_names}\n"
            f"Propose a hypothesis explaining this correlation.\n"
            f"Write to: world_model/generation_responses/{next_id}_proposal.json\n"
            f"Format: {{\"name\": \"...\", \"statement\": \"...\", "
            f"\"testable_predictions\": [{{\"id\": \"P001\", \"prediction\": \"...\"}}, ...]}}"
        )
    else:
        return None

    return base


def generate_method_proposal(trigger, exploration_map, hypotheses) -> dict:
    """Build a structured method proposal for the agent to review."""
    existing_methods = set()
    for tier_key in ["tier_1_methods", "tier_2_methods", "tier_3_methods"]:
        existing_methods.update(exploration_map.get(tier_key, {}).keys())

    hyp_names = {h["id"]: h["name"] for h in hypotheses.get("hypotheses", [])}

    return {
        "action": "generate_method",
        "trigger": {
            "type": trigger["type"],
            "hypothesis_id": trigger.get("hypothesis_id"),
            "detail": trigger["detail"],
        },
        "prompt_for_agent": (
            f"METHOD NEEDED for {trigger.get('hypothesis_id', '?')} "
            f"({hyp_names.get(trigger.get('hypothesis_id', ''), '?')}).\n"
            f"Reason: {trigger['detail']}\n"
            f"Existing methods ({len(existing_methods)}): {sorted(existing_methods)}\n\n"
            f"Propose a NEW method not in the existing set.\n"
            f"Write to: world_model/generation_responses/M_new_method_proposal.json\n"
            f"Format: {{\"name\": \"method_name_snake_case\", "
            f"\"description\": \"...\", \"priority\": 3, \"differentiation\": 4, "
            f"\"applicable_hypotheses\": [\"H001\", \"H002\", ...]}}"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# Integration (write to world model)
# ═══════════════════════════════════════════════════════════════════════

def add_generated_hypothesis(proposal_id, agent_response, cycle):
    """Add a new hypothesis to hypotheses.json and create exploration_map cells."""
    hypotheses = load_hypotheses()
    emap = load_exploration_map()

    # Build new hypothesis entry
    new_hyp = {
        "id": proposal_id,
        "name": agent_response["name"],
        "statement": agent_response["statement"],
        "status": "testing",
        "confidence": 0.0,
        "source": "agent_generated",
        "agent_generated": True,
        "generated_cycle": cycle,
        "prune_deadline_cycle": cycle + 3,
        "supporting_evidence": [],
        "refuting_evidence": [],
        "testable_predictions": agent_response.get("testable_predictions", []),
        "convergence_metrics": {
            "methods_confirming": 0,
            "methods_refuting": 0,
            "methods_null": 0,
            "min_methods_for_convergence": 3,
            "data_subsets_tested": 0,
            "min_subsets_for_convergence": 1,
        },
        "history": [{"cycle": cycle, "status": "testing",
                     "rationale": f"Agent-generated hypothesis"}],
    }

    hypotheses["hypotheses"].append(new_hyp)
    save_hypotheses(hypotheses)

    # Create cells in exploration_map for ALL existing methods
    cell_key = hypothesis_cell_key(new_hyp)
    for tier_key in ["tier_1_methods", "tier_2_methods"]:
        tier = emap.get(tier_key, {})
        for method_name, method_data in tier.items():
            method_data[cell_key] = {"status": "not_started"}

    recount_exploration_map(emap)
    save_exploration_map(emap)

    # Log
    log = load_generation_log()
    log["hypothesis_generations"].append({
        "id": proposal_id,
        "name": agent_response["name"],
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "status": "active",
    })
    save_generation_log(log)

    return new_hyp


def add_generated_method(agent_response, cycle):
    """Add a new method to exploration_map.json and create cells for ALL hypotheses."""
    hypotheses = load_hypotheses()
    emap = load_exploration_map()

    method_name = agent_response["name"]
    applicable = set(agent_response.get("applicable_hypotheses", []))

    method_entry = {
        "priority": agent_response.get("priority", 3),
        "differentiation": agent_response.get("differentiation", 4),
        "agent_generated": True,
        "generated_cycle": cycle,
        "description": agent_response.get("description", ""),
    }

    # Create cells for all existing hypotheses
    for h in hypotheses["hypotheses"]:
        cell_key = hypothesis_cell_key(h)
        if h["id"] in applicable or not applicable:
            method_entry[cell_key] = {"status": "not_started"}
        else:
            method_entry[cell_key] = {"status": "not_applicable"}

    emap.setdefault("tier_2_methods", {})[method_name] = method_entry
    recount_exploration_map(emap)
    save_exploration_map(emap)

    # Log
    log = load_generation_log()
    log["method_generations"].append({
        "name": method_name,
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "status": "active",
    })
    save_generation_log(log)

    return method_entry


def recount_exploration_map(emap):
    """Recount summary stats after structural changes."""
    total = 0
    explored = 0
    not_applicable = 0
    not_started = 0
    in_progress = 0

    for tier_key in ["tier_1_methods", "tier_2_methods", "tier_3_methods"]:
        tier = emap.get(tier_key, {})
        for method, mdata in tier.items():
            for key, val in mdata.items():
                if key in METADATA_KEYS:
                    continue
                if isinstance(val, dict):
                    total += 1
                    status = val.get("status", "")
                    if status in ("completed", "in_progress"):
                        explored += 1
                    elif status == "not_applicable":
                        not_applicable += 1
                    elif status == "not_started":
                        not_started += 1

    actionable = total - not_applicable
    emap["summary"] = {
        "total_cells": total,
        "explored": explored,
        "not_started": not_started,
        "not_applicable": not_applicable,
        "coverage": round(explored / max(actionable, 1), 3),
    }


# ═══════════════════════════════════════════════════════════════════════
# Pruning
# ═══════════════════════════════════════════════════════════════════════

def prune_stale_hypotheses(cycle):
    """Remove agent-generated hypotheses with no evidence past deadline."""
    hypotheses = load_hypotheses()
    emap = load_exploration_map()

    pruned = []
    remaining = []

    for h in hypotheses["hypotheses"]:
        if not h.get("agent_generated"):
            remaining.append(h)
            continue

        deadline = h.get("prune_deadline_cycle", cycle + 999)
        if cycle >= deadline:
            cm = h.get("convergence_metrics", {})
            total_evidence = cm.get("methods_confirming", 0) + cm.get("methods_refuting", 0)
            if total_evidence == 0:
                pruned.append(h)
                # Remove cells from exploration_map
                cell_key = hypothesis_cell_key(h)
                for tier_key in ["tier_1_methods", "tier_2_methods"]:
                    tier = emap.get(tier_key, {})
                    for method_data in tier.values():
                        method_data.pop(cell_key, None)
                continue

        remaining.append(h)

    if pruned:
        hypotheses["hypotheses"] = remaining
        recount_exploration_map(emap)
        save_hypotheses(hypotheses)
        save_exploration_map(emap)

        log = load_generation_log()
        for h in pruned:
            log["pruning_events"].append({
                "type": "hypothesis",
                "id": h["id"],
                "name": h["name"],
                "cycle": cycle,
                "reason": "No evidence after deadline",
            })
        save_generation_log(log)

    return pruned


def prune_stale_methods(cycle):
    """Remove agent-generated methods never used after 5 cycles."""
    emap = load_exploration_map()
    pruned = []

    for tier_key in ["tier_2_methods"]:  # Generated methods are always tier_2
        tier = emap.get(tier_key, {})
        to_remove = []

        for method_name, mdata in tier.items():
            if not mdata.get("agent_generated"):
                continue

            gen_cycle = mdata.get("generated_cycle", 0)
            if cycle - gen_cycle < 5:
                continue  # Not old enough

            # Check if any cell has been used
            any_used = False
            for key, val in mdata.items():
                if key in METADATA_KEYS:
                    continue
                if isinstance(val, dict) and val.get("status") in ("completed", "in_progress"):
                    any_used = True
                    break

            if not any_used:
                to_remove.append(method_name)

        for m in to_remove:
            del tier[m]
            pruned.append(m)

    if pruned:
        recount_exploration_map(emap)
        save_exploration_map(emap)

        log = load_generation_log()
        for m in pruned:
            log["pruning_events"].append({
                "type": "method", "name": m, "cycle": cycle,
                "reason": "Never used after 5 cycles",
            })
        save_generation_log(log)

    return pruned


# ═══════════════════════════════════════════════════════════════════════
# Finalization (read agent responses)
# ═══════════════════════════════════════════════════════════════════════

def finalize_generation_responses(cycle):
    """Read agent proposal responses and integrate into world model."""
    GENERATION_RESPONSES_DIR.mkdir(exist_ok=True)
    results = {"hypotheses_added": [], "methods_added": []}

    for response_file in sorted(GENERATION_RESPONSES_DIR.glob("*.json")):
        try:
            with open(response_file) as f:
                response = json.load(f)

            filename = response_file.stem

            if filename.startswith("H") and "_proposal" in filename:
                # Hypothesis proposal response
                proposal_id = filename.split("_")[0]  # e.g., "H004"
                hyp = add_generated_hypothesis(proposal_id, response, cycle)
                results["hypotheses_added"].append(hyp["id"])
                print(f"  Added hypothesis: {hyp['id']} — {hyp['name']}")

            elif filename.startswith("M_"):
                # Method proposal response
                method = add_generated_method(response, cycle)
                results["methods_added"].append(response["name"])
                print(f"  Added method: {response['name']}")

            # Remove processed file
            response_file.unlink()

        except Exception as e:
            print(f"  Error processing {response_file}: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════════

def test_generators():
    """Test trigger detection and proposal generation with mock data."""
    print("Generator Self-Test")
    print("=" * 60)

    # Mock hypotheses
    hypotheses = {
        "hypotheses": [
            {"id": "H001", "name": "Cloud Buffering",
             "statement": "Clouds compensate surface albedo changes to maintain stability",
             "status": "supported", "confidence": 0.85,
             "supporting_evidence": ["E001"], "refuting_evidence": [],
             "testable_predictions": [], "convergence_metrics": {"methods_confirming": 5}},
            {"id": "H002", "name": "Invariant Properties",
             "statement": "Physical constants and geometry constrain albedo",
             "status": "testing", "confidence": 0.5,
             "supporting_evidence": [], "refuting_evidence": ["E015"],
             "testable_predictions": [
                 {"id": "P005", "prediction": "Clausius-Clapeyron constrains albedo range",
                  "tested_by": [], "untested_methods": []}
             ],
             "convergence_metrics": {"methods_confirming": 3, "min_methods_for_convergence": 5}},
            {"id": "H003", "name": "Teleconnections",
             "statement": "Cross-hemispheric coupling maintains symmetry",
             "status": "supported", "confidence": 0.9,
             "supporting_evidence": ["E001", "E002"], "refuting_evidence": [],
             "testable_predictions": [], "convergence_metrics": {"methods_confirming": 5}},
        ]
    }

    # Mock findings
    mock_finding_contradiction = {
        "finding_id": "E099",
        "cycle": 20, "task_id": 1,
        "summary": "Analysis reveals that stratospheric aerosol layer dynamics and polar vortex interactions drive albedo changes independently of tropospheric clouds",
        "statistics": {"p_value": 0.001, "correlation_enso_stratosphere": 0.65},
        "hypothesis_id": "H002",
        "refutes_hypothesis": True,
        "confidence": 0.8,
        "method": "novel_analysis",
    }

    mock_finding_adversarial = {
        "finding_id": "E100",
        "cycle": 20, "task_id": 2,
        "summary": "Adversarial testing shows poly2 spatial confound INVALIDATED the Bayesian comparison",
        "statistics": {"p_value": 0.01},
        "hypothesis_id": "H002",
        "refutes_hypothesis": True,
        "confidence": 0.8,
        "method": "adversarial_testing",
    }

    literature = {"papers": [], "open_questions": []}
    all_findings = [mock_finding_contradiction, mock_finding_adversarial]

    # ── Test hypothesis triggers ──
    print("\n--- Hypothesis Triggers ---")

    triggers = detect_hypothesis_triggers(
        mock_finding_contradiction, all_findings, hypotheses, literature)
    print(f"Contradiction finding → {len(triggers)} triggers:")
    for t in triggers:
        print(f"  [{t['type']}] {t['detail'][:80]}")

    triggers2 = detect_hypothesis_triggers(
        mock_finding_adversarial, all_findings, hypotheses, literature)
    print(f"\nAdversarial finding → {len(triggers2)} triggers:")
    for t in triggers2:
        print(f"  [{t['type']}] {t['detail'][:80]}")

    # ── Test proposals ──
    print("\n--- Proposals ---")
    if triggers:
        proposal = generate_hypothesis_proposal(triggers[0], hypotheses, literature, all_findings)
        if proposal:
            print(f"Hypothesis proposal: {proposal['id']} ({proposal.get('template', '?')})")
            print(f"  Prompt: {proposal['prompt_for_agent'][:150]}...")

    # ── Test method triggers ──
    print("\n--- Method Triggers ---")
    mock_emap = {"tier_1_methods": {}, "tier_2_methods": {}, "summary": {"total_cells": 0}}
    method_triggers = detect_method_triggers(hypotheses["hypotheses"][1], mock_emap, all_findings)
    print(f"H002 method triggers → {len(method_triggers)}:")
    for t in method_triggers:
        print(f"  [{t['type']}] {t['detail'][:80]}")

    if method_triggers:
        m_proposal = generate_method_proposal(method_triggers[0], mock_emap, hypotheses)
        print(f"\nMethod proposal:")
        print(f"  Prompt: {m_proposal['prompt_for_agent'][:150]}...")

    # ── Test concept extraction ──
    print("\n--- Novel Concept Extraction ---")
    novel = extract_novel_concepts(
        "The ENSO-driven Walker circulation modulates polar vortex dynamics and creates a bifurcation in albedo response",
        "clouds compensate surface albedo through scattering and teleconnections maintain hemispheric symmetry"
    )
    print(f"Novel concepts: {novel}")

    print(f"\n{'='*60}")
    print("Generator self-test complete.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_generators()
    else:
        # Run on real world model
        latest = load_latest_finding()
        if latest:
            hypotheses = load_hypotheses()
            literature = load_literature()
            all_findings = load_all_findings()

            h_triggers = detect_hypothesis_triggers(latest, all_findings, hypotheses, literature)
            print(f"Hypothesis triggers: {len(h_triggers)}")
            for t in h_triggers:
                print(f"  [{t['type']}] {t['detail']}")

            for h in hypotheses["hypotheses"]:
                if h["status"] == "testing":
                    emap = load_exploration_map()
                    m_triggers = detect_method_triggers(h, emap, all_findings)
                    if m_triggers:
                        print(f"\nMethod triggers for {h['id']}:")
                        for t in m_triggers:
                            print(f"  [{t['type']}] {t['detail']}")
        else:
            print("No findings yet — no triggers to detect.")
