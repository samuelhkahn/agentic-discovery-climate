#!/usr/bin/env python3
"""
Agentic Discovery Climate — Research Loop Orchestrator.

Each run:
0. Prune stale generated hypotheses/methods
1. Detect triggers for new hypothesis/method generation
2. Build context (hypotheses, frontier, MCTS recommendation)
3. Print cycle prompt with proposals for agent
4. After agent executes: finalize any accepted generation proposals

Usage:
    python scripts/run_cycle.py
"""

import json
import sys
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    _load_meta, load_hypotheses, load_exploration_map, load_literature,
    get_frontier, check_convergence, start_cycle, end_cycle, WORLD_MODEL_DIR
)
from mcts import run_mcts_search, MCTSConfig
from generators import (
    detect_hypothesis_triggers, detect_method_triggers,
    generate_hypothesis_proposal, generate_method_proposal,
    prune_stale_hypotheses, prune_stale_methods,
    finalize_generation_responses,
    load_latest_finding, load_all_findings,
)


def build_cycle_context():
    """Build the full context for the Claude Code agent to plan the next cycle."""
    meta = _load_meta()
    hypotheses = load_hypotheses()
    emap = load_exploration_map()
    literature = load_literature()
    convergence = check_convergence()
    frontier = get_frontier(n=10)

    # MCTS-informed recommendation
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

    # Detect generation triggers
    latest_finding = load_latest_finding()
    all_findings = load_all_findings()
    hypothesis_proposals = []
    method_proposals = []

    if latest_finding:
        h_triggers = detect_hypothesis_triggers(
            latest_finding, all_findings, hypotheses, literature)
        for t in h_triggers[:2]:  # Max 2 proposals per cycle
            proposal = generate_hypothesis_proposal(t, hypotheses, literature, all_findings)
            if proposal:
                hypothesis_proposals.append(proposal)

        for h in hypotheses["hypotheses"]:
            if h["status"] == "testing":
                m_triggers = detect_method_triggers(h, emap, all_findings)
                for t in m_triggers[:1]:  # Max 1 method proposal per hypothesis
                    proposal = generate_method_proposal(t, emap, hypotheses)
                    if proposal:
                        method_proposals.append(proposal)

    # Load recent cycle summaries (last 3)
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
            "needs": h.get("testable_predictions", [{}])[0].get("needs", ""),
        })

    context = {
        "cycle_number": meta["current_cycle"] + 1,
        "status": meta["status"],
        "convergence": convergence,
        "hypothesis_status": hyp_table,
        "frontier": frontier,
        "mcts_recommendation": mcts_recommendation,
        "hypothesis_proposals": hypothesis_proposals,
        "method_proposals": method_proposals,
        "exploration_coverage": emap["summary"].get("coverage", 0),
        "total_findings": meta["total_findings"],
        "recent_summaries": recent_summaries,
        "open_questions": literature.get("open_questions", []),
    }

    return context


def print_cycle_prompt(context):
    """Print the structured prompt for the Claude Code agent."""
    print("=" * 80)
    print(f"AGENTIC DISCOVERY CLIMATE — CYCLE {context['cycle_number']}")
    print("=" * 80)
    print()

    # Convergence status
    conv = context["convergence"]
    print(f"CONVERGENCE: {conv['progress']*100:.0f}% of hypotheses resolved")
    if conv["converged"]:
        print("ALL HYPOTHESES RESOLVED — system can stop.")
        return
    print(f"  Resolved: {conv['resolved']}")
    print(f"  Unresolved: {conv['unresolved']}")
    print()

    # Hypothesis status table
    print("HYPOTHESIS STATUS:")
    print(f"{'ID':<6} {'Name':<25} {'Status':<12} {'Conf':<6} {'Confirm':<8} {'Refute':<8} {'Gen?'}")
    print("-" * 78)
    for h in context["hypothesis_status"]:
        gen_marker = "  *" if h.get("agent_generated") else ""
        print(f"{h['id']:<6} {h['name']:<25} {h['status']:<12} {h['confidence']:<6.1f} "
              f"{h['confirming_methods']:<8} {h['refuting_methods']:<8}{gen_marker}")
    print()

    # Exploration frontier
    print(f"EXPLORATION COVERAGE: {context['exploration_coverage']*100:.1f}%")
    print()
    print("TOP FRONTIER (greedy):")
    for i, f in enumerate(context["frontier"][:10], 1):
        gen = " [generated]" if f.get("agent_generated") else ""
        print(f"  {i}. [{f['score']:.1f}] {f['method']} × {f['hypothesis']}{gen}")
    print()

    # MCTS recommendation
    mcts = context.get("mcts_recommendation", {})
    if "action" in mcts:
        method, hyp = mcts["action"]
        print(f"MCTS RECOMMENDATION ({mcts['total_rollouts']} rollouts, {mcts['search_time']:.2f}s, depth={mcts['tree_depth']}):")
        print(f"  >>> {method} × {hyp}")
        print(f"      Value: {mcts['value']:.3f}, Visits: {mcts['visits']}")
        print()
        if mcts.get("alternatives"):
            print("  Alternatives (by value):")
            for alt in mcts["alternatives"][:5]:
                m, h = alt["action"]
                print(f"    {m:30s} × {h}  value={alt['value']:.3f}  visits={alt['visits']}")
        print()
        print(f"  Reasoning: {mcts['reasoning']}")
        print()
    elif "error" in mcts:
        print(f"MCTS: Error — {mcts['error']}")
        print()

    # ── Generation Proposals ──
    hyp_proposals = context.get("hypothesis_proposals", [])
    method_proposals = context.get("method_proposals", [])

    if hyp_proposals or method_proposals:
        print("=" * 60)
        print("DYNAMIC GENERATION PROPOSALS")
        print("=" * 60)
        print("Triggers detected from recent findings. You may ACCEPT or REJECT each.")
        print()

        for i, p in enumerate(hyp_proposals, 1):
            print(f"HYPOTHESIS PROPOSAL {i} (trigger: {p['trigger']['type']}):")
            print(f"  ID: {p['id']}")
            print(f"  {p['prompt_for_agent']}")
            print()

        for i, p in enumerate(method_proposals, 1):
            print(f"METHOD PROPOSAL {i} (trigger: {p['trigger']['type']}):")
            print(f"  {p['prompt_for_agent']}")
            print()

    # Recent findings
    if context["recent_summaries"]:
        print("RECENT CYCLE SUMMARIES:")
        for s in context["recent_summaries"]:
            print(f"  Cycle {s['cycle']}: {s['summary'][:200]}")
        print()

    # Instructions
    print("INSTRUCTIONS:")
    print("1. Consider the MCTS recommendation (optimized for convergence)")
    print("2. Review any GENERATION PROPOSALS — accept by writing response JSON")
    print("3. Write analysis script in world_model/analyses/")
    print("4. Execute, record findings, update hypotheses")
    print("5. You may also propose new hypotheses/methods at any time")
    print()
    print("You have FULL FREEDOM in choice of method, hypothesis, and approach.")
    print("=" * 80)


def main():
    meta = _load_meta()

    # Phase 0: Prune stale generated hypotheses/methods
    pruned_h = prune_stale_hypotheses(meta["current_cycle"])
    pruned_m = prune_stale_methods(meta["current_cycle"])
    if pruned_h:
        print(f"PRUNED {len(pruned_h)} stale hypothesis(es): {[h['id'] for h in pruned_h]}")
    if pruned_m:
        print(f"PRUNED {len(pruned_m)} stale method(s): {pruned_m}")

    # Phase 1: Finalize any pending generation responses from previous cycle
    results = finalize_generation_responses(meta["current_cycle"])
    if results["hypotheses_added"] or results["methods_added"]:
        print(f"INTEGRATED: {results['hypotheses_added']} hypotheses, {results['methods_added']} methods")

    # Phase 2-3: Build context and print prompt
    context = build_cycle_context()
    print_cycle_prompt(context)

    # Save context
    context_path = WORLD_MODEL_DIR / "current_cycle_context.json"
    # Convert tuples to lists for JSON serialization
    serializable = json.loads(json.dumps(context, default=str))
    with open(context_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull context saved to: {context_path}")


if __name__ == "__main__":
    main()
