# Agentic Discovery Climate — Agent Instructions

You are an autonomous AI scientist exploring hypotheses about Earth's albedo stability.

## Lifecycle

```
EXPLORING → (all hypotheses converge) → SYNTHESIZING → (theory rated ≥8) → COMPLETE + PPTX
```

## Each Cycle (Exploring Phase)

### Step 1: See the state
```bash
python scripts/run_cycle.py
```

### Step 2: Run the MCTS-recommended experiment
Write a self-contained Python script in `world_model/analyses/A0XX_description/script.py`.

### Step 3: ADVERSARIAL SELF-CHECK (mandatory before recording)
After getting your result, you MUST run these checks:

1. **Shuffle test**: Run the same analysis on permuted/shuffled data. If the result persists on random data, it's trivially true → DO NOT RECORD.
2. **Confound test**: Remove ENSO signal (residualize against Nino3.4) or remove seasonal cycle more aggressively. If the result disappears → DO NOT RECORD.
3. **Robustness test**: Bootstrap (500x, check 95% CI excludes zero), split-half (2000-2012 vs 2013-2025), or vary key parameters. If fragile → DO NOT RECORD.

### Step 4: Record ONLY if passes
```python
import sys; sys.path.insert(0, "scripts")
from utils import create_finding, mark_explored, start_cycle, end_cycle, save_hypotheses, load_hypotheses

cycle = start_cycle()
finding = create_finding(
    cycle=cycle, task_id=1,
    summary="What was found + adversarial checks passed: [list which checks and results]",
    statistics={"p_value": 0.001, "effect_size": 0.5, "bootstrap_ci": [0.3, 0.7], ...},
    hypothesis_id="H001",
    confidence=0.8,
    method="method_name",
    script_path="world_model/analyses/A0XX_description/script.py",
)
# Update hypothesis convergence metrics...
end_cycle(cycle, summary="...")
```

If the adversarial check FAILS, still call `end_cycle` but note the failure — do not call `create_finding`.

### Step 5: Accept/reject generation proposals
If the prompt shows HYPOTHESIS or METHOD proposals, write response JSON to `world_model/generation_responses/`.

### Step 6: Loop back to Step 1

## Theory Synthesis Phase

When all hypotheses converge, the system transitions to SYNTHESIZING. The prompt will show detailed instructions. You must:

1. **Propose a theory** — a unified mechanism explaining WHY albedo is stable, not just a summary of findings
2. **Critique it yourself** — list every logical gap, unsupported claim, or weakness
3. **Revise** — fix the gaps, re-critique, repeat until self-rated ≥ 8/10
4. **Save** to `world_model/theory.json`

The theory must:
- Be consistent with all confirmed evidence
- Explain causal mechanisms (not just correlations)
- Make 2+ new testable predictions
- State what would falsify it
- Address inconclusive hypotheses honestly

## Complete Phase

When the theory is saved with rating ≥ 8, the system auto-generates a PPTX presentation and marks research complete.

## Key Rules
- NEVER record a finding without running adversarial self-checks
- Every finding summary should mention which adversarial checks passed
- MCTS recommendation is a suggestion — override with reasoning if needed
- You can propose new hypotheses or methods at any time
- A theory is NOT a summary — it's a MECHANISM with PREDICTIONS
