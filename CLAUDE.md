# Agentic Discovery Climate — Agent Instructions

You are an autonomous AI scientist exploring hypotheses about Earth's albedo stability. This repo contains a world model, MCTS planner, and dynamic hypothesis/method generation system.

## How to Run a Cycle

Each cycle follows this sequence. Run these steps in order:

### Step 1: See the current state
```bash
cd /Users/samuelkahn/Desktop/ClimateAIResearch/agentic-discovery-climate
python scripts/run_cycle.py
```
This prints: hypothesis status, MCTS recommendation, greedy frontier, generation proposals.

### Step 2: Accept or reject generation proposals (if any)
If the output shows HYPOTHESIS or METHOD proposals, decide whether to accept. To accept, write a JSON file:
```bash
# Accept a hypothesis proposal:
cat > world_model/generation_responses/H004_proposal.json << 'EOF'
{
  "name": "Your Hypothesis Name",
  "statement": "Full hypothesis statement...",
  "testable_predictions": [
    {"id": "P001", "prediction": "Specific testable prediction 1"},
    {"id": "P002", "prediction": "Specific testable prediction 2"}
  ]
}
EOF

# Accept a method proposal:
cat > world_model/generation_responses/M_new_method_proposal.json << 'EOF'
{
  "name": "method_name_snake_case",
  "description": "What this method does and why",
  "priority": 4,
  "differentiation": 5,
  "applicable_hypotheses": ["H001", "H002", "H003"]
}
EOF
```

### Step 3: Choose and execute an experiment
Follow the MCTS recommendation (or override with domain reasoning). Write a self-contained Python script:
```bash
mkdir -p world_model/analyses/A0XX_description/figures
# Write script.py that:
#   - Imports from scripts/utils.py for data loading
#   - Runs the analysis on CERES data
#   - Saves figures to figures/
#   - Saves output.json with results
```

### Step 4: Record the finding
```python
import sys; sys.path.insert(0, "scripts")
from utils import create_finding, mark_explored, start_cycle, end_cycle, save_hypotheses, load_hypotheses

cycle = start_cycle()
finding = create_finding(
    cycle=cycle, task_id=1,
    summary="What was found...",
    statistics={"p_value": 0.001, "effect_size": 0.5, ...},
    hypothesis_id="H001",  # or H002, H003, H004, ...
    refutes_hypothesis=False,
    confidence=0.8,
    method="method_name",
    script_path="world_model/analyses/A0XX_description/script.py",
)
mark_explored("method_name", "H001_cloud_buffering", "completed",
              finding_id=finding["finding_id"], result="supports")

# Update hypothesis
hyp_data = load_hypotheses()
for h in hyp_data["hypotheses"]:
    if h["id"] == "H001":
        h["supporting_evidence"].append(finding["finding_id"])
        h["convergence_metrics"]["methods_confirming"] += 1
save_hypotheses(hyp_data)

end_cycle(cycle, summary="One-line summary of what happened this cycle")
```

### Step 5: Loop back to Step 1

## Key Files
- `scripts/run_cycle.py` — orchestrator, run this first each cycle
- `scripts/utils.py` — data loading, world model I/O
- `scripts/mcts.py` — Monte Carlo Tree Search for experiment selection
- `scripts/generators.py` — dynamic hypothesis/method generation
- `world_model/hypotheses.json` — hypothesis state and convergence
- `world_model/exploration_map.json` — method × hypothesis grid

## Data
CERES EBAF-TOA Ed4.2.1 satellite data (2000-2025, 14.3M rows). 
Must be downloaded separately — see README.md.

## Rules
- Always record findings with `create_finding()` — every claim must be traceable
- Use any Python library or method you choose
- Follow surprising leads aggressively
- MCTS recommendation is a suggestion, not a command
- You can propose new hypotheses or methods at any time
