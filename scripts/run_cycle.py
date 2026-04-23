#!/usr/bin/env python3
"""
Agentic Discovery Climate — Research Loop Orchestrator.

Lifecycle per cycle:
  1. MCTS selects experiment
  2. Agent executes experiment, gets result
  3. ADVERSARIAL SELF-CHECK on the result (before recording)
  4. Record finding ONLY if it survives adversarial check
  5. Check convergence
  6. If converged → THEORY SYNTHESIS (reflective loop)
  7. If theory has no holes → Generate PPTX → COMPLETE

Phases:
  exploring    → run experiments with per-cycle adversarial validation
  synthesizing → propose and refine a unified scientific theory
  complete     → generate slides, done

Usage:
    python scripts/run_cycle.py
"""

import json
import sys
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    _load_meta, save_meta, load_hypotheses, load_exploration_map, load_literature,
    get_frontier, check_convergence, start_cycle, end_cycle, WORLD_MODEL_DIR
)
from mcts import run_mcts_search, MCTSConfig
from generators import (
    detect_hypothesis_triggers, detect_method_triggers,
    generate_hypothesis_proposal, generate_method_proposal,
    prune_stale_hypotheses, prune_stale_methods,
    finalize_generation_responses,
    load_latest_finding, load_all_findings,
    ADVERSARIAL_ATTACK_TYPES,
)


# ═══════════════════════════════════════════════════════════════════════
# Per-Cycle Adversarial Self-Check
# ═══════════════════════════════════════════════════════════════════════

def generate_adversarial_self_check(finding):
    """Generate adversarial checks for a SINGLE finding before it's recorded.

    Returns a list of checks the agent should run on its own result.
    The agent must run these checks and only record the finding if it survives.
    """
    if finding is None:
        return []

    method = finding.get("method", "")
    summary = finding.get("summary", "")[:300]
    stats = finding.get("statistics", {})
    p_value = stats.get("p_value", "N/A")

    checks = []

    # Check 1: Is this trivially true?
    checks.append({
        "type": "trivial_result",
        "instruction": (
            "SELF-CHECK 1 — Trivial Result:\n"
            "Would this result hold for RANDOM or SHUFFLED data?\n"
            "Test: run the same analysis on shuffled/permuted data.\n"
            "If the result persists on random data, it's trivially true → REJECT.\n"
            "If the result vanishes, it reflects genuine signal → PASS."
        )
    })

    # Check 2: Confound check (if correlational/causal method)
    if any(kw in method.lower() for kw in ["granger", "correlation", "regression",
                                              "ccm", "transfer_entropy", "ica",
                                              "wavelet", "climate_network"]):
        checks.append({
            "type": "confound_check",
            "instruction": (
                "SELF-CHECK 2 — Confounding Variable:\n"
                "Could ENSO, seasonal residuals, or spatial autocorrelation\n"
                "explain this result?\n"
                "Test: deseasonalize more aggressively, or residualize against\n"
                "Nino3.4 index, then re-run. If result disappears → REJECT.\n"
                "If it persists after deconfounding → PASS."
            )
        })

    # Check 3: Robustness (if has p-value)
    if isinstance(p_value, (int, float)) and p_value < 1:
        checks.append({
            "type": "robustness",
            "instruction": (
                "SELF-CHECK 3 — Robustness:\n"
                f"Current p-value: {p_value}\n"
                "Test at least ONE of:\n"
                "  a) Bootstrap: resample 500x, does 95% CI exclude zero?\n"
                "  b) Split-half: does result hold on 2000-2012 AND 2013-2025?\n"
                "  c) Parameter sensitivity: vary key params (bins, lag, threshold)\n"
                "If fragile under perturbation → REJECT or WEAKEN.\n"
                "If robust → PASS."
            )
        })

    return checks


# ═══════════════════════════════════════════════════════════════════════
# Theory Synthesis Node
# ═══════════════════════════════════════════════════════════════════════

THEORY_SYNTHESIS_PROMPT = """
═══════════════════════════════════════════════════════════════
THEORY SYNTHESIS — Propose a Unified Scientific Theory
═══════════════════════════════════════════════════════════════

All hypotheses have converged. Your task is to synthesize the evidence
into a coherent SCIENTIFIC THEORY that explains Earth's albedo stability
and hemispheric symmetry.

EVIDENCE SUMMARY:
{evidence_summary}

INSTRUCTIONS:
1. Write a unified theory (2-3 paragraphs) that explains HOW and WHY
   Earth's albedo is stable at ~29% and why NH ≈ SH.

2. The theory must:
   - Be consistent with ALL confirmed evidence
   - Explain the causal mechanisms (not just correlations)
   - Make at least 2 NEW testable predictions not yet tested
   - Address any inconclusive hypotheses honestly
   - Be falsifiable — state what observation would disprove it

3. After writing the theory, CRITIQUE it yourself:
   - List every logical gap or unsupported claim
   - For each gap, state what evidence would fill it
   - Rate the theory's strength (1-10)

4. If there are gaps (rating < 8), REVISE the theory to address them.
   Repeat the critique-revise loop until rating ≥ 8 or you've done 3 iterations.

5. Save the final theory to: world_model/theory.json
   Format:
   {{
     "theory_statement": "The unified theory...",
     "key_mechanisms": ["mechanism 1", "mechanism 2", ...],
     "new_predictions": [
       {{"prediction": "...", "test": "how to test it"}},
       ...
     ],
     "limitations": ["limitation 1", ...],
     "falsification_criteria": "What would disprove this theory",
     "confidence": 0.0-1.0,
     "critique_iterations": N,
     "final_rating": N,
     "cycle_generated": N
   }}

6. Once the theory is saved, the system will generate the final
   presentation slides and mark the research as COMPLETE.

REMEMBER: A theory is not a summary of results. It's a MECHANISM
that explains WHY the results are what they are, and PREDICTS what
would happen under conditions not yet observed.
"""


def build_evidence_summary(hypotheses, findings):
    """Build a structured evidence summary for theory synthesis."""
    lines = []

    for h in hypotheses.get("hypotheses", []):
        cm = h.get("convergence_metrics", {})
        status = h.get("status", "testing")
        conf = h.get("confidence", 0)

        lines.append(f"\n{h['id']} — {h['name']} [{status.upper()}, conf={conf:.2f}]")
        lines.append(f"  Statement: {h['statement'][:200]}")
        lines.append(f"  Methods confirming: {cm.get('methods_confirming', 0)}, "
                     f"refuting: {cm.get('methods_refuting', 0)}")

        # Key supporting findings
        supporting_ids = h.get("supporting_evidence", [])
        h_findings = [f for f in findings if f.get("finding_id") in supporting_ids]
        for f in h_findings[:3]:
            method = f.get("method", "?")
            summary = f.get("summary", "")[:100]
            lines.append(f"  • {method}: {summary}")

    return "\n".join(lines)


def check_theory_exists():
    """Check if a theory.json exists and is complete."""
    theory_path = WORLD_MODEL_DIR / "theory.json"
    if not theory_path.exists():
        return False, None
    with open(theory_path) as f:
        theory = json.load(f)
    rating = theory.get("final_rating", 0)
    return rating >= 8, theory


# ═══════════════════════════════════════════════════════════════════════
# Context Builder
# ═══════════════════════════════════════════════════════════════════════

def build_cycle_context():
    """Build the full context for the Claude Code agent."""
    meta = _load_meta()
    hypotheses = load_hypotheses()
    emap = load_exploration_map()
    literature = load_literature()
    convergence = check_convergence()
    frontier = get_frontier(n=10)

    # Determine phase
    phase = meta.get("convergence_state", "exploring")
    if convergence["converged"] and phase == "exploring":
        phase = "synthesizing"
    elif phase == "synthesizing":
        theory_ok, theory = check_theory_exists()
        if theory_ok:
            phase = "complete"

    # MCTS recommendation (only in exploring phase)
    mcts_recommendation = {}
    if phase == "exploring":
        try:
            mcts_result = run_mcts_search(config=MCTSConfig(num_rollouts=200))
            mcts_recommendation = {
                "action": mcts_result.recommended_action,
                "value": mcts_result.action_values.get(mcts_result.recommended_action, 0),
                "visits": mcts_result.action_visits.get(mcts_result.recommended_action, 0),
                "alternatives": [
                    {"action": a, "value": v, "visits": mcts_result.action_visits.get(a, 0)}
                    for a, v in sorted(mcts_result.action_values.items(), key=lambda x: -x[1])[:5]
                ],
                "reasoning": mcts_result.reasoning,
                "total_rollouts": mcts_result.total_rollouts,
                "search_time": mcts_result.search_time_seconds,
                "tree_depth": mcts_result.tree_depth,
            }
        except Exception as e:
            mcts_recommendation = {"error": str(e)}

    # Detect generation triggers (only in exploring phase)
    latest_finding = load_latest_finding()
    all_findings = load_all_findings()
    hypothesis_proposals = []
    method_proposals = []

    if phase == "exploring" and latest_finding:
        h_triggers = detect_hypothesis_triggers(
            latest_finding, all_findings, hypotheses, literature)
        for t in h_triggers[:2]:
            proposal = generate_hypothesis_proposal(t, hypotheses, literature, all_findings)
            if proposal:
                hypothesis_proposals.append(proposal)

        for h in hypotheses["hypotheses"]:
            if h["status"] == "testing":
                m_triggers = detect_method_triggers(h, emap, all_findings)
                for t in m_triggers[:1]:
                    proposal = generate_method_proposal(t, emap, hypotheses)
                    if proposal:
                        method_proposals.append(proposal)

    # Per-cycle adversarial check info (for the NEXT experiment)
    adversarial_checks = generate_adversarial_self_check(latest_finding) if latest_finding else []

    # Theory synthesis info
    theory_prompt = ""
    if phase == "synthesizing":
        theory_prompt = THEORY_SYNTHESIS_PROMPT.format(
            evidence_summary=build_evidence_summary(hypotheses, all_findings)
        )

    # Load recent cycle summaries
    recent_summaries = []
    summary_dir = WORLD_MODEL_DIR / "cycle_summaries"
    if summary_dir.exists():
        for i in range(max(1, meta["current_cycle"] - 2), meta["current_cycle"] + 1):
            sf = summary_dir / f"cycle_{i:02d}_summary.json"
            if sf.exists():
                with open(sf) as f:
                    recent_summaries.append(json.load(f))

    # Build hypothesis status table
    hyp_table = []
    for h in hypotheses["hypotheses"]:
        cm = h.get("convergence_metrics", {})
        hyp_table.append({
            "id": h["id"],
            "name": h["name"],
            "status": h["status"],
            "confidence": h["confidence"],
            "confirming_methods": cm.get("methods_confirming", 0),
            "refuting_methods": cm.get("methods_refuting", 0),
            "agent_generated": h.get("agent_generated", False),
        })

    context = {
        "cycle_number": meta["current_cycle"] + 1,
        "status": meta["status"],
        "phase": phase,
        "convergence": convergence,
        "hypothesis_status": hyp_table,
        "frontier": frontier,
        "mcts_recommendation": mcts_recommendation,
        "hypothesis_proposals": hypothesis_proposals,
        "method_proposals": method_proposals,
        "adversarial_self_checks": adversarial_checks,
        "theory_prompt": theory_prompt,
        "exploration_coverage": emap["summary"].get("coverage", 0),
        "total_findings": meta["total_findings"],
        "recent_summaries": recent_summaries,
    }

    return context


# ═══════════════════════════════════════════════════════════════════════
# Prompt Printer
# ═══════════════════════════════════════════════════════════════════════

def print_cycle_prompt(context):
    """Print the structured prompt for the Claude Code agent."""
    phase = context.get("phase", "exploring")

    print("=" * 80)
    if phase == "synthesizing":
        print(f"AGENTIC DISCOVERY CLIMATE — THEORY SYNTHESIS")
    elif phase == "complete":
        print(f"AGENTIC DISCOVERY CLIMATE — COMPLETE")
    else:
        print(f"AGENTIC DISCOVERY CLIMATE — CYCLE {context['cycle_number']}")
    print("=" * 80)
    print()

    # Phase indicator
    phase_descriptions = {
        "exploring": None,
        "synthesizing": (
            "*** THEORY SYNTHESIS PHASE ***\n"
            "All hypotheses converged with adversarially-validated evidence.\n"
            "Now: propose a unified scientific theory, critique it, refine it.\n"
            "Convergence: theory has no logical gaps (self-rated >= 8/10).\n"
        ),
        "complete": (
            "*** RESEARCH COMPLETE ***\n"
            "Theory synthesized and validated. Generating presentation.\n"
        ),
    }
    if phase_descriptions.get(phase):
        print(phase_descriptions[phase])

    # Convergence status
    conv = context["convergence"]
    print(f"CONVERGENCE: {conv['progress']*100:.0f}% of hypotheses resolved | Phase: {phase}")
    print(f"  Resolved: {conv['resolved']}")
    print(f"  Unresolved: {conv['unresolved']}")
    print()

    # Hypothesis status table
    print("HYPOTHESIS STATUS:")
    print(f"{'ID':<6} {'Name':<30} {'Status':<12} {'Conf':<6} {'OK':<5} {'Fail':<5} {'Gen?'}")
    print("-" * 78)
    for h in context["hypothesis_status"]:
        gen = "  *" if h.get("agent_generated") else ""
        print(f"{h['id']:<6} {h['name'][:29]:<30} {h['status']:<12} "
              f"{h['confidence']:<6.2f} {h['confirming_methods']:<5} "
              f"{h['refuting_methods']:<5}{gen}")
    print()

    # ── EXPLORING PHASE ──
    if phase == "exploring":
        # Frontier
        print(f"EXPLORATION COVERAGE: {context['exploration_coverage']*100:.1f}%")
        print()
        print("TOP FRONTIER (greedy):")
        for i, f in enumerate(context["frontier"][:8], 1):
            gen = " [gen]" if f.get("agent_generated") else ""
            print(f"  {i}. [{f['score']:.1f}] {f['method']} x {f['hypothesis']}{gen}")
        print()

        # MCTS
        mcts = context.get("mcts_recommendation", {})
        if "action" in mcts:
            m, h = mcts["action"]
            print(f"MCTS RECOMMENDATION ({mcts['total_rollouts']} rollouts, "
                  f"{mcts['search_time']:.2f}s):")
            print(f"  >>> {m} x {h}  (value={mcts['value']:.3f})")
            if mcts.get("alternatives"):
                for alt in mcts["alternatives"][1:4]:
                    am, ah = alt["action"]
                    print(f"      {am} x {ah}  (value={alt['value']:.3f})")
            print()

        # Generation proposals
        for i, p in enumerate(context.get("hypothesis_proposals", []), 1):
            print(f"HYPOTHESIS PROPOSAL {i} ({p['trigger']['type']}): {p['id']}")
            print(f"  {p['prompt_for_agent'][:200]}")
            print()
        for i, p in enumerate(context.get("method_proposals", []), 1):
            print(f"METHOD PROPOSAL {i} ({p['trigger']['type']}):")
            print(f"  {p['prompt_for_agent'][:200]}")
            print()

        # Per-cycle adversarial instructions
        print("=" * 60)
        print("PER-CYCLE ADVERSARIAL PROTOCOL")
        print("=" * 60)
        print("After running your experiment, you MUST self-check before recording:")
        print()
        for check in context.get("adversarial_self_checks", [])[:3]:
            print(check["instruction"])
            print()
        if not context.get("adversarial_self_checks"):
            print("Run these checks on every new finding:")
            print("  1. Shuffle/permute test: does result hold on random data? → trivial check")
            print("  2. Confound test: remove ENSO/seasonal signal, re-run → confound check")
            print("  3. Robustness test: bootstrap CI, split-half, or param sensitivity")
            print()
        print("ONLY record the finding (create_finding) if it PASSES all checks.")
        print("If a check FAILS, note why and try a different approach.")
        print()

        # Instructions
        print("CYCLE STEPS:")
        print("  1. Run MCTS-recommended experiment (or override with reasoning)")
        print("  2. Get result")
        print("  3. Run adversarial self-checks (above)")
        print("  4. If passes: record finding with create_finding()")
        print("  5. If fails: note failure, do NOT record, try different method")
        print("  6. Accept/reject any generation proposals")
        print()

    # ── SYNTHESIZING PHASE ──
    elif phase == "synthesizing":
        print(context.get("theory_prompt", ""))
        print()

    # ── COMPLETE PHASE ──
    elif phase == "complete":
        theory_ok, theory = check_theory_exists()
        if theory:
            print("THEORY:")
            print(f"  {theory.get('theory_statement', '')[:500]}")
            print(f"\n  Confidence: {theory.get('confidence', 0)}")
            print(f"  Rating: {theory.get('final_rating', 0)}/10")
            print(f"  Iterations: {theory.get('critique_iterations', 0)}")
        print()

    # Recent summaries
    if context["recent_summaries"]:
        print("RECENT CYCLES:")
        for s in context["recent_summaries"]:
            print(f"  Cycle {s['cycle']}: {s['summary'][:150]}")
        print()

    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    meta = _load_meta()

    # Phase 0: Prune stale generated hypotheses/methods
    pruned_h = prune_stale_hypotheses(meta["current_cycle"])
    pruned_m = prune_stale_methods(meta["current_cycle"])
    if pruned_h:
        print(f"PRUNED {len(pruned_h)} stale hypothesis(es): {[h['id'] for h in pruned_h]}")
    if pruned_m:
        print(f"PRUNED {len(pruned_m)} stale method(s): {pruned_m}")

    # Finalize any pending generation responses
    results = finalize_generation_responses(meta["current_cycle"])
    if results["hypotheses_added"] or results["methods_added"]:
        print(f"INTEGRATED: {results['hypotheses_added']} hypotheses, {results['methods_added']} methods")

    # Check phase transitions
    convergence = check_convergence()
    current_state = meta.get("convergence_state", "exploring")

    if convergence["converged"] and current_state == "exploring":
        meta["convergence_state"] = "synthesizing"
        save_meta(meta)
        print("\n>>> All hypotheses converged → entering THEORY SYNTHESIS <<<\n")

    elif current_state == "synthesizing":
        theory_ok, theory = check_theory_exists()
        if theory_ok:
            meta["convergence_state"] = "complete"
            save_meta(meta)
            print("\n>>> Theory validated → RESEARCH COMPLETE <<<\n")

            # Auto-generate presentation
            try:
                from generate_slides import generate_presentation
                output_path = generate_presentation()
                print(f">>> PRESENTATION GENERATED: {output_path} <<<\n")
            except Exception as e:
                print(f">>> Slide generation failed: {e} <<<\n")

    # Build context and print prompt
    context = build_cycle_context()
    print_cycle_prompt(context)

    # Save context
    context_path = WORLD_MODEL_DIR / "current_cycle_context.json"
    serializable = json.loads(json.dumps(context, default=str))
    with open(context_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nContext saved to: {context_path}")


if __name__ == "__main__":
    main()
