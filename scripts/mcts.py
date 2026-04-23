#!/usr/bin/env python3
"""
Monte Carlo Tree Search for scientific experiment selection.

Replaces greedy frontier scoring with lookahead planning that considers:
- Sequential dependencies between experiments
- Information gain across all hypotheses simultaneously
- Exploration/exploitation balance via UCB1
- Expected trajectory to convergence

The LLM plays four roles (all via prompts or heuristics):
1. Action generator (policy): propose K plausible next experiments
2. Value estimator: rate convergence likelihood for a trajectory
3. World model (simulator): predict experiment outcome
4. Terminal check: are all hypotheses resolved?

Usage:
    # From run_cycle.py:
    from mcts import run_mcts_search, MCTSConfig
    result = run_mcts_search(config=MCTSConfig(num_rollouts=50))

    # Standalone test:
    python scripts/mcts.py --test
"""

import math
import random
import copy
import json
import time
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HypothesisStatus:
    """Convergence state for a single hypothesis."""
    id: str
    name: str
    methods_confirming: int = 0
    methods_refuting: int = 0
    methods_null: int = 0
    data_subsets_tested: int = 0
    confidence: float = 0.0
    status: str = "testing"
    min_methods: int = 5
    min_subsets: int = 2

    def is_resolved(self) -> bool:
        return (
            (self.methods_confirming >= self.min_methods and self.data_subsets_tested >= self.min_subsets)
            or (self.methods_refuting >= self.min_methods and self.data_subsets_tested >= self.min_subsets)
        )

    def progress(self) -> float:
        """0-1 progress toward convergence."""
        method_prog = min(self.methods_confirming, self.min_methods) / self.min_methods
        subset_prog = min(self.data_subsets_tested, self.min_subsets) / self.min_subsets
        return (method_prog + subset_prog) / 2


@dataclass
class MCTSState:
    """Snapshot of the world for MCTS simulation."""
    hypotheses: dict  # {id: HypothesisStatus}
    explored_cells: set  # {(method, hypothesis), ...}
    cycles_elapsed: int = 0
    trajectory: list = field(default_factory=list)  # [(action, outcome), ...]

    def copy(self):
        return MCTSState(
            hypotheses={k: copy.copy(v) for k, v in self.hypotheses.items()},
            explored_cells=self.explored_cells.copy(),
            cycles_elapsed=self.cycles_elapsed,
            trajectory=self.trajectory.copy(),
        )

    def is_terminal(self) -> bool:
        return all(h.is_resolved() for h in self.hypotheses.values())

    def overall_progress(self) -> float:
        if not self.hypotheses:
            return 0.0
        return sum(h.progress() for h in self.hypotheses.values()) / len(self.hypotheses)

    def apply_outcome(self, action: tuple, outcome: str):
        """Update state after an experiment."""
        method, hypothesis = action
        self.explored_cells.add((method, hypothesis))
        self.cycles_elapsed += 1
        self.trajectory.append((action, outcome))

        h = self.hypotheses.get(hypothesis)
        if h is None:
            # Try without prefix
            for hid, hobj in self.hypotheses.items():
                if hypothesis.startswith(hid) or hid.startswith(hypothesis.split("_")[0]):
                    h = hobj
                    break
        if h is None:
            return

        if outcome == "supports":
            h.methods_confirming += 1
            h.data_subsets_tested = max(h.data_subsets_tested, 1)
            h.confidence = min(1.0, h.confidence + 0.1)
        elif outcome == "refutes":
            h.methods_refuting += 1
            h.data_subsets_tested = max(h.data_subsets_tested, 1)
            h.confidence = max(0.0, h.confidence - 0.1)
        # inconclusive: no change to confirming/refuting


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: MCTSState
    action: Optional[tuple] = None  # (method, hypothesis) that led here
    outcome: Optional[str] = None   # outcome that led here (for chance nodes)
    parent: Optional['MCTSNode'] = None
    children: list = field(default_factory=list)

    # MCTS statistics
    visits: int = 0
    total_value: float = 0.0

    # Chance node metadata
    is_chance_node: bool = False
    outcome_probs: Optional[dict] = None  # {"supports": 0.5, ...}

    @property
    def value(self) -> float:
        return self.total_value / max(self.visits, 1)

    def is_fully_expanded(self, max_children: int = 5) -> bool:
        if self.is_chance_node:
            return len(self.children) >= 3
        return len(self.children) >= max_children

    def best_child_ucb(self, c: float = 1.414, balance_weight: float = 0.2) -> 'MCTSNode':
        """Select child with highest UCB1 score."""
        best = None
        best_score = -float('inf')

        for child in self.children:
            if child.visits == 0:
                return child  # Always visit unexplored nodes first

            exploitation = child.value
            exploration = c * math.sqrt(math.log(self.visits) / child.visits)

            # Hypothesis balance bonus: favor actions on least-progressed hypothesis
            balance = 0.0
            if child.action:
                _, hyp = child.action
                for hid, hobj in self.state.hypotheses.items():
                    if hyp.startswith(hid) or hid.startswith(hyp.split("_")[0]):
                        balance = 1.0 - hobj.progress()
                        break

            score = exploitation + exploration + balance_weight * balance

            if score > best_score:
                best_score = score
                best = child

        return best

    def sample_chance_child(self) -> 'MCTSNode':
        """Sample a chance child proportional to outcome probabilities."""
        if not self.children:
            return None
        weights = []
        for c in self.children:
            w = self.outcome_probs.get(c.outcome, 0.33) if self.outcome_probs else 0.33
            weights.append(w)
        return random.choices(self.children, weights=weights, k=1)[0]


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""
    num_rollouts: int = 50
    max_rollout_depth: int = 8
    llm_rollout_depth: int = 2      # Steps using LLM simulator (rest use heuristic)
    ucb_c: float = 1.414
    action_branching: int = 5       # Max actions expanded per node
    balance_weight: float = 0.2     # Weight for hypothesis balance bonus
    convergence_weight: float = 0.6 # Weight for convergence vs scientific value
    use_llm: bool = False           # If False, use pure heuristic (fast mode)


@dataclass
class MCTSResult:
    """Output of an MCTS search."""
    recommended_action: tuple
    action_values: dict    # {(method, hyp): value}
    action_visits: dict    # {(method, hyp): visits}
    total_rollouts: int
    search_time_seconds: float
    tree_depth: int
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════
# Scientific Oracle (LLM or Heuristic)
# ═══════════════════════════════════════════════════════════════════════

class HeuristicOracle:
    """Fast heuristic oracle for MCTS rollouts. No LLM calls.

    Learns base rates from historical findings and uses domain priors
    (priority, differentiation scores) to estimate outcomes.
    """

    def __init__(self, exploration_map: dict = None, findings_history: list = None):
        self.exploration_map = exploration_map or {}
        self.findings_history = findings_history or []

        # Learn base rates from history
        outcomes = [f.get("outcome", "inconclusive") for f in self.findings_history]
        n = max(len(outcomes), 1)
        self.base_rates = {
            "supports": max(outcomes.count("supports") / n, 0.3),
            "refutes": max(outcomes.count("refutes") / n, 0.1),
            "inconclusive": max(outcomes.count("inconclusive") / n, 0.2),
        }
        # Normalize
        total = sum(self.base_rates.values())
        self.base_rates = {k: v / total for k, v in self.base_rates.items()}

    def get_priority(self, method: str) -> int:
        """Look up method priority from exploration map."""
        for tier_key in ["tier_1_methods", "tier_2_methods", "tier_3_methods"]:
            tier = self.exploration_map.get(tier_key, {})
            if method in tier:
                return tier[method].get("priority", 3)
        return 3

    def get_differentiation(self, method: str) -> int:
        for tier_key in ["tier_1_methods", "tier_2_methods", "tier_3_methods"]:
            tier = self.exploration_map.get(tier_key, {})
            if method in tier:
                return tier[method].get("differentiation", 3)
        return 3

    def is_novel(self, method: str, hypothesis: str) -> bool:
        """Check if method or hypothesis is agent-generated."""
        for tier_key in ["tier_1_methods", "tier_2_methods"]:
            tier = self.exploration_map.get(tier_key, {})
            if method in tier and tier[method].get("agent_generated"):
                return True
        # Check hypothesis ID: H004+ are generated
        try:
            h_num = int(hypothesis[1:4]) if hypothesis[0] == "H" else 0
            if h_num > 3:
                return True
        except (ValueError, IndexError):
            pass
        return False

    def predict_outcome(self, state: MCTSState, action: tuple) -> dict:
        """Predict outcome probabilities for (method, hypothesis)."""
        method, hypothesis = action
        priority = self.get_priority(method)

        # Higher priority methods are more likely to produce signal
        p_supports = 0.15 + 0.08 * priority
        p_refutes = 0.05 + 0.02 * priority
        p_inconclusive = 1.0 - p_supports - p_refutes

        # Adjust based on hypothesis momentum
        for hid, h in state.hypotheses.items():
            if hypothesis.startswith(hid) or hid.startswith(hypothesis.split("_")[0]):
                if h.methods_confirming > h.methods_refuting:
                    p_supports *= 1.3  # Momentum: confirming begets confirming
                elif h.methods_refuting > h.methods_confirming:
                    p_refutes *= 1.3
                break

        # Novel entries: blend toward uniform (higher uncertainty)
        if self.is_novel(method, hypothesis):
            uncertainty_factor = 0.4
            uniform = 1.0 / 3.0
            p_supports = p_supports * (1 - uncertainty_factor) + uniform * uncertainty_factor
            p_refutes = p_refutes * (1 - uncertainty_factor) + uniform * uncertainty_factor
            p_inconclusive = p_inconclusive * (1 - uncertainty_factor) + uniform * uncertainty_factor

        # Normalize
        total = p_supports + p_refutes + p_inconclusive
        return {
            "supports": p_supports / total,
            "refutes": p_refutes / total,
            "inconclusive": p_inconclusive / total,
        }

    def sample_outcome(self, state: MCTSState, action: tuple) -> str:
        """Sample a single outcome from predicted distribution."""
        probs = self.predict_outcome(state, action)
        return random.choices(
            list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]

    def estimate_value(self, state: MCTSState) -> float:
        """Estimate value of a state (0-1). Higher = closer to convergence."""
        progress = state.overall_progress()

        # Bonus for balanced progress (all hypotheses advancing)
        progresses = [h.progress() for h in state.hypotheses.values()]
        if progresses:
            balance = 1.0 - (max(progresses) - min(progresses))
        else:
            balance = 0.0

        # Penalty for too many cycles
        efficiency = max(0.0, 1.0 - state.cycles_elapsed / 30.0)

        return 0.6 * progress + 0.2 * balance + 0.2 * efficiency

    def propose_actions(self, state: MCTSState, available: list, k: int = 5) -> list:
        """Rank top-K actions from available set."""
        scored = []
        for action in available:
            method, hypothesis = action
            priority = self.get_priority(method)
            diff = self.get_differentiation(method)

            # Score: priority + differentiation + hypothesis need
            hyp_need = 0.0
            for hid, h in state.hypotheses.items():
                if hypothesis.startswith(hid) or hid.startswith(hypothesis.split("_")[0]):
                    hyp_need = 1.0 - h.progress()
                    break

            score = 0.3 * priority / 5 + 0.2 * diff / 5 + 0.5 * hyp_need
            scored.append((action, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [a for a, s in scored[:k]]


# ═══════════════════════════════════════════════════════════════════════
# MCTS Engine
# ═══════════════════════════════════════════════════════════════════════

class MCTSEngine:
    """Monte Carlo Tree Search for scientific experiment planning."""

    def __init__(self, config: MCTSConfig, oracle: HeuristicOracle,
                 available_actions: list):
        self.config = config
        self.oracle = oracle
        self.available_actions = available_actions  # [(method, hypothesis), ...]

    def search(self, root_state: MCTSState) -> MCTSResult:
        """Run MCTS and return recommended action."""
        start_time = time.time()

        root = MCTSNode(state=root_state)
        max_depth = 0

        for iteration in range(self.config.num_rollouts):
            # 1. Selection: walk tree using UCB1
            node = self._select(root)

            # 2. Expansion: add a new child
            child = self._expand(node)
            if child is None:
                child = node

            # 3. Simulation: rollout to estimate value
            value = self._rollout(child)

            # 4. Backpropagation
            self._backpropagate(child, value)

            # Track depth
            depth = 0
            n = child
            while n.parent:
                n = n.parent
                depth += 1
            max_depth = max(max_depth, depth)

        # Select best action by visit count (more robust than value)
        if not root.children:
            # Fallback: return first available action
            return MCTSResult(
                recommended_action=self.available_actions[0] if self.available_actions else ("none", "none"),
                action_values={},
                action_visits={},
                total_rollouts=self.config.num_rollouts,
                search_time_seconds=time.time() - start_time,
                tree_depth=0,
                reasoning="No children expanded — using fallback"
            )

        # Collect action-level statistics (aggregate over chance children)
        action_stats = {}
        for child in root.children:
            a = child.action
            if a not in action_stats:
                action_stats[a] = {"visits": 0, "value": 0.0}
            action_stats[a]["visits"] += child.visits
            action_stats[a]["value"] += child.total_value

        action_values = {a: s["value"] / max(s["visits"], 1) for a, s in action_stats.items()}
        action_visits = {a: s["visits"] for a, s in action_stats.items()}

        best_action = max(action_stats, key=lambda a: action_stats[a]["visits"])

        # Generate reasoning
        top_3 = sorted(action_values.items(), key=lambda x: -x[1])[:3]
        reasoning_parts = []
        for action, val in top_3:
            method, hyp = action
            visits = action_visits[action]
            reasoning_parts.append(f"{method}×{hyp} (value={val:.3f}, visits={visits})")
        reasoning = f"MCTS selected {best_action[0]}×{best_action[1]} after {self.config.num_rollouts} rollouts. " \
                    f"Top alternatives: {'; '.join(reasoning_parts)}"

        return MCTSResult(
            recommended_action=best_action,
            action_values=action_values,
            action_visits=action_visits,
            total_rollouts=self.config.num_rollouts,
            search_time_seconds=time.time() - start_time,
            tree_depth=max_depth,
            reasoning=reasoning,
        )

    def _get_available(self, state: MCTSState) -> list:
        """Get actions not yet explored in this state."""
        return [a for a in self.available_actions if a not in state.explored_cells]

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk from root to a leaf using UCB1."""
        while not node.state.is_terminal():
            if node.is_chance_node:
                # Sample outcome proportional to probabilities
                child = node.sample_chance_child()
                if child is None:
                    return node
                node = child
            elif not node.is_fully_expanded(self.config.action_branching):
                return node  # Expand this node
            elif node.children:
                node = node.best_child_ucb(
                    c=self.config.ucb_c,
                    balance_weight=self.config.balance_weight
                )
            else:
                return node
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Add a new child to the node."""
        if node.state.is_terminal():
            return node

        available = self._get_available(node.state)
        if not available:
            return node

        # Get actions not already expanded as children
        expanded_actions = {c.action for c in node.children}
        unexpanded = [a for a in available if a not in expanded_actions]

        if not unexpanded:
            return node

        # Use oracle to rank and pick best unexpanded action
        ranked = self.oracle.propose_actions(node.state, unexpanded, k=1)
        if not ranked:
            return node
        action = ranked[0]

        # Predict outcome probabilities
        outcome_probs = self.oracle.predict_outcome(node.state, action)

        # Create action node (chance node with 3 outcome children)
        action_node = MCTSNode(
            state=node.state,
            action=action,
            parent=node,
            is_chance_node=True,
            outcome_probs=outcome_probs,
        )
        node.children.append(action_node)

        # Expand one outcome child (sample from distribution)
        outcome = random.choices(
            list(outcome_probs.keys()),
            weights=list(outcome_probs.values()),
            k=1
        )[0]

        new_state = node.state.copy()
        new_state.apply_outcome(action, outcome)

        outcome_child = MCTSNode(
            state=new_state,
            action=action,
            outcome=outcome,
            parent=action_node,
        )
        action_node.children.append(outcome_child)

        return outcome_child

    def _rollout(self, node: MCTSNode, max_depth: int = None) -> float:
        """Simulate from node to estimate value."""
        if max_depth is None:
            max_depth = self.config.max_rollout_depth

        state = node.state.copy()
        depth = 0

        while depth < max_depth and not state.is_terminal():
            available = self._get_available(state)
            if not available:
                break

            # Pick action: top-ranked by oracle
            ranked = self.oracle.propose_actions(state, available, k=3)
            if not ranked:
                break
            action = random.choice(ranked)  # Randomize among top-3 for diversity

            # Simulate outcome
            outcome = self.oracle.sample_outcome(state, action)
            state.apply_outcome(action, outcome)
            depth += 1

        # Terminal value estimation
        return self.oracle.estimate_value(state)

    def _backpropagate(self, node: MCTSNode, value: float):
        """Propagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent


# ═══════════════════════════════════════════════════════════════════════
# Integration Entry Point
# ═══════════════════════════════════════════════════════════════════════

def run_mcts_search(config: MCTSConfig = None) -> MCTSResult:
    """Called from run_cycle.py to get next experiment recommendation.

    Builds MCTS state from world model files, runs search, returns result.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import (
        load_hypotheses, load_exploration_map, _load_meta,
        WORLD_MODEL_DIR
    )

    if config is None:
        config = MCTSConfig()

    # Build state
    hypotheses_data = load_hypotheses()
    emap = load_exploration_map()
    meta = _load_meta()

    hyp_status = {}
    for h in hypotheses_data["hypotheses"]:
        cm = h.get("convergence_metrics", {})
        hyp_status[h["id"]] = HypothesisStatus(
            id=h["id"],
            name=h["name"],
            methods_confirming=cm.get("methods_confirming", 0),
            methods_refuting=cm.get("methods_refuting", 0),
            methods_null=cm.get("methods_null", 0),
            data_subsets_tested=cm.get("data_subsets_tested", 0),
            confidence=h.get("confidence", 0.0),
            status=h.get("status", "testing"),
            min_methods=cm.get("min_methods_for_convergence", 5),
            min_subsets=cm.get("min_subsets_for_convergence", 2),
        )

    explored = set()
    available = []
    for tier_key in ["tier_1_methods", "tier_2_methods"]:
        tier = emap.get(tier_key, {})
        for method, mdata in tier.items():
            for key, val in mdata.items():
                if key in ("priority", "differentiation", "agent_generated",
                          "generated_cycle", "generating_trigger", "description"):
                    continue
                if isinstance(val, dict):
                    if val.get("status") in ("completed", "in_progress"):
                        explored.add((method, key))
                    elif val.get("status") == "not_started":
                        available.append((method, key))

    state = MCTSState(
        hypotheses=hyp_status,
        explored_cells=explored,
        cycles_elapsed=meta.get("current_cycle", 0),
    )

    # Load historical findings for oracle calibration
    findings = []
    findings_dir = WORLD_MODEL_DIR / "findings"
    if findings_dir.exists():
        for cycle_dir in findings_dir.iterdir():
            if cycle_dir.is_dir():
                for task_dir in cycle_dir.iterdir():
                    fp = task_dir / "findings.json"
                    if fp.exists():
                        with open(fp) as f:
                            finding = json.load(f)
                            outcome = "supports"
                            if finding.get("refutes_hypothesis"):
                                outcome = "refutes"
                            elif finding.get("confidence", 0) < 0.3:
                                outcome = "inconclusive"
                            findings.append({"outcome": outcome})

    oracle = HeuristicOracle(exploration_map=emap, findings_history=findings)
    engine = MCTSEngine(config=config, oracle=oracle, available_actions=available)
    result = engine.search(state)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Self-Test
# ═══════════════════════════════════════════════════════════════════════

def test_mcts():
    """Run MCTS with mock state to verify tree construction."""
    print("MCTS Self-Test")
    print("=" * 60)

    # Create mock state
    state = MCTSState(
        hypotheses={
            "H001": HypothesisStatus("H001", "Cloud Buffering", methods_confirming=1, confidence=0.6),
            "H002": HypothesisStatus("H002", "Invariant Properties", methods_confirming=0, confidence=0.4),
            "H003": HypothesisStatus("H003", "Teleconnections", methods_confirming=0,
                                      methods_refuting=1, confidence=0.3),
        },
        explored_cells={("ffnn_perturbation", "H001")},
        cycles_elapsed=0,
    )

    # Mock available actions
    available = [
        ("transfer_entropy", "H003"), ("wavelet_coherence", "H003"),
        ("convergent_cross_mapping", "H001"), ("attractor_reconstruction", "H002"),
        ("symbolic_regression", "H002"), ("granger_causality", "H001"),
        ("climate_networks", "H003"), ("ica", "H001"),
        ("regime_switching", "H001"), ("sindy", "H002"),
        ("bayesian_comparison", "H002"), ("hidden_markov", "H002"),
    ]

    # Mock exploration map for priority lookup
    mock_emap = {
        "tier_1_methods": {
            "transfer_entropy": {"priority": 5, "differentiation": 5},
            "convergent_cross_mapping": {"priority": 5, "differentiation": 5},
            "attractor_reconstruction": {"priority": 5, "differentiation": 5},
            "wavelet_coherence": {"priority": 5, "differentiation": 4},
            "climate_networks": {"priority": 5, "differentiation": 5},
            "symbolic_regression": {"priority": 5, "differentiation": 5},
        },
        "tier_2_methods": {
            "granger_causality": {"priority": 3, "differentiation": 3},
            "ica": {"priority": 4, "differentiation": 4},
            "regime_switching": {"priority": 4, "differentiation": 4},
            "sindy": {"priority": 4, "differentiation": 5},
            "bayesian_comparison": {"priority": 5, "differentiation": 4},
            "hidden_markov": {"priority": 4, "differentiation": 4},
        },
    }

    oracle = HeuristicOracle(exploration_map=mock_emap)

    # Test with different rollout counts
    for num_rollouts in [10, 50, 200]:
        config = MCTSConfig(num_rollouts=num_rollouts, action_branching=5)
        engine = MCTSEngine(config=config, oracle=oracle, available_actions=available)

        result = engine.search(state)

        print(f"\n--- {num_rollouts} rollouts ({result.search_time_seconds:.2f}s) ---")
        print(f"Recommended: {result.recommended_action[0]} × {result.recommended_action[1]}")
        print(f"Tree depth: {result.tree_depth}")

        # Show top 5 actions by visits
        sorted_actions = sorted(result.action_visits.items(), key=lambda x: -x[1])
        print(f"Action ranking:")
        for action, visits in sorted_actions[:5]:
            value = result.action_values.get(action, 0)
            print(f"  {action[0]:30s} × {action[1]:6s}: visits={visits:4d}, value={value:.3f}")

    print(f"\n{'='*60}")
    print("MCTS self-test complete.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_mcts()
    else:
        result = run_mcts_search()
        print(f"MCTS Recommendation: {result.recommended_action}")
        print(f"Reasoning: {result.reasoning}")
