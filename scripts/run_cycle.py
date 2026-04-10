#!/usr/bin/env python3
"""
Kosmos-Albedo Research Loop Orchestrator.

This script is designed to be executed by a Claude Code agent. Each run:
1. Reads the current world model state
2. Identifies the highest-priority unexplored (method × hypothesis) cell
3. Generates and executes an analysis script
4. Records findings and updates the world model
5. Checks convergence

The Claude Code agent provides the intelligence — this script provides the structure.

Usage:
    python scripts/run_cycle.py

Or invoke from Claude Code as the orchestration entry point.
"""

import json
import sys
from pathlib import Path

# Add scripts dir to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    _load_meta, load_hypotheses, load_exploration_map, load_literature,
    get_frontier, check_convergence, start_cycle, end_cycle, WORLD_MODEL_DIR
)


def build_cycle_context():
    """Build the full context for the Claude Code agent to plan the next cycle."""
    meta = _load_meta()
    hypotheses = load_hypotheses()
    emap = load_exploration_map()
    literature = load_literature()
    convergence = check_convergence()
    frontier = get_frontier(n=10)

    # Load recent cycle summaries (last 3)
    recent_summaries = []
    summary_dir = WORLD_MODEL_DIR / "cycle_summaries"
    if summary_dir.exists():
        for i in range(max(1, meta["current_cycle"] - 2), meta["current_cycle"] + 1):
            sf = summary_dir / f"cycle_{i:02d}_summary.json"
            if sf.exists():
                with open(sf) as f:
                    recent_summaries.append(json.load(f))

    # Load prior work
    prior_work_path = WORLD_MODEL_DIR / "prior_work.yaml"
    prior_work = ""
    if prior_work_path.exists():
        with open(prior_work_path) as f:
            prior_work = f.read()

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
            "needs": h.get("testable_predictions", [{}])[0].get("needs", "")
        })

    context = {
        "cycle_number": meta["current_cycle"] + 1,
        "status": meta["status"],
        "convergence": convergence,
        "hypothesis_status": hyp_table,
        "frontier": frontier,
        "exploration_coverage": emap["summary"]["coverage"],
        "total_findings": meta["total_findings"],
        "recent_summaries": recent_summaries,
        "open_questions": literature.get("open_questions", []),
    }

    return context


def print_cycle_prompt(context):
    """Print the structured prompt for the Claude Code agent."""
    print("=" * 80)
    print(f"KOSMOS-ALBEDO RESEARCH CYCLE {context['cycle_number']}")
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
    print(f"{'ID':<6} {'Name':<25} {'Status':<12} {'Conf':<6} {'Confirm':<8} {'Refute':<8}")
    print("-" * 70)
    for h in context["hypothesis_status"]:
        print(f"{h['id']:<6} {h['name']:<25} {h['status']:<12} {h['confidence']:<6.1f} {h['confirming_methods']:<8} {h['refuting_methods']:<8}")
    print()

    # Exploration frontier
    print(f"EXPLORATION COVERAGE: {context['exploration_coverage']*100:.1f}%")
    print()
    print("TOP FRONTIER (highest-priority unexplored cells):")
    for i, f in enumerate(context["frontier"][:10], 1):
        print(f"  {i}. [{f['score']:.1f}] {f['method']} × {f['hypothesis']} (P={f['priority']}, D={f['differentiation']})")
    print()

    # Recent findings
    if context["recent_summaries"]:
        print("RECENT CYCLE SUMMARIES:")
        for s in context["recent_summaries"]:
            print(f"  Cycle {s['cycle']}: {s['summary'][:200]}")
        print()

    # Instructions
    print("INSTRUCTIONS FOR AGENT:")
    print("1. Select the highest-priority frontier cell (or follow a promising lead)")
    print("2. Write a Python analysis script in world_model/analyses/")
    print("3. Execute the script and record findings")
    print("4. Search literature for relevant papers using WebSearch")
    print("5. Update hypothesis status if evidence warrants it")
    print("6. Check if any hypothesis has converged")
    print("7. Propose next cycle priorities")
    print()
    print("You have FULL FREEDOM to choose any method, library, or approach.")
    print("The frontier is a suggestion — follow surprising leads aggressively.")
    print("=" * 80)


def main():
    context = build_cycle_context()
    print_cycle_prompt(context)

    # Output context as JSON for programmatic consumption
    context_path = WORLD_MODEL_DIR / "current_cycle_context.json"
    with open(context_path, "w") as f:
        json.dump(context, f, indent=2)
    print(f"\nFull context saved to: {context_path}")


if __name__ == "__main__":
    main()
