"""
Shared utilities for Kosmos-Albedo AI Scientist.
Data loading, world model I/O, statistical helpers.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent          # kosmos-albedo/
WORLD_MODEL_DIR = PROJECT_ROOT / "world_model"
DATA_ROOT = PROJECT_ROOT.parent                      # ClimateAIResearch/

# ── Data Loading ───────────────────────────────────────────────────────

def load_ceres_data(terminator_filter=True):
    """Load and concatenate CERES EBAF parquet data."""
    meta = _load_meta()
    p1 = Path(meta["data_paths"]["ceres_parquet_1"])
    p2 = Path(meta["data_paths"]["ceres_parquet_2"])

    dfs = []
    if p1.exists():
        dfs.append(pd.read_parquet(p1))
    if p2.exists():
        dfs.append(pd.read_parquet(p2))

    if not dfs:
        raise FileNotFoundError(f"CERES parquet files not found at {p1} or {p2}")

    df = pd.concat(dfs, ignore_index=True)

    if terminator_filter and "lat" in df.columns:
        df = df[(df["lat"] >= -66) & (df["lat"] <= 66)]

    return df


def load_zone_weights():
    """Load cosine-latitude area weights."""
    meta = _load_meta()
    path = Path(meta["data_paths"]["zone_weights"])
    return pd.read_csv(path)


def load_ffnn_model():
    """Load the trained FFNN PyTorch model."""
    import torch
    meta = _load_meta()
    path = Path(meta["data_paths"]["ffnn_checkpoint"])
    if not path.exists():
        raise FileNotFoundError(f"FFNN checkpoint not found at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def load_merra2_monthly(year, month):
    """Load a single MERRA-2 monthly radiation file."""
    import xarray as xr
    meta = _load_meta()
    merra2_dir = Path(meta["data_paths"]["merra2_dir"])

    # Stream number depends on year
    if year <= 1991:
        stream = 100
    elif year <= 2000:
        stream = 200
    elif year <= 2010:
        stream = 300
    else:
        stream = 400

    filename = f"MERRA2_{stream}.tavgM_2d_rad_Nx.{year}{month:02d}.nc4"
    path = merra2_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"MERRA-2 file not found: {path}")
    return xr.open_dataset(path)


def load_merra2_timeseries(start_year=1980, end_year=2025):
    """Load MERRA-2 monthly data as a concatenated time series."""
    import xarray as xr
    meta = _load_meta()
    merra2_dir = Path(meta["data_paths"]["merra2_dir"])

    files = sorted(merra2_dir.glob("MERRA2_*.nc4"))
    if not files:
        raise FileNotFoundError(f"No MERRA-2 files found in {merra2_dir}")

    ds = xr.open_mfdataset(files, combine="by_coords")
    return ds


def compute_weighted_hemispheric_means(df, value_col="toa_alb_all_mon"):
    """Compute area-weighted hemispheric means from gridded data."""
    weights = load_zone_weights()
    df = df.copy()
    weight_map = dict(zip(
        weights["lat"].astype(float).round(1),
        weights["weight"].astype(float)
    ))
    df["weight"] = df["lat"].round(1).map(weight_map)
    df["weighted_val"] = df[value_col] * df["weight"]

    nh = df[df["lat"] > 0]
    sh = df[df["lat"] < 0]

    nh_mean = nh["weighted_val"].sum() / nh["weight"].sum() if len(nh) else np.nan
    sh_mean = sh["weighted_val"].sum() / sh["weight"].sum() if len(sh) else np.nan

    return {"NH": nh_mean, "SH": sh_mean, "global": (nh_mean + sh_mean) / 2}


def compute_monthly_hemispheric_timeseries(value_col="toa_alb_all_mon"):
    """Compute area-weighted hemispheric mean albedo time series from CERES."""
    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    weight_map = dict(zip(
        weights["lat"].astype(float).round(1),
        weights["weight"].astype(float)
    ))
    df["weight"] = df["lat"].round(1).map(weight_map)
    df["weighted_val"] = df[value_col] * df["weight"]

    results = []
    for (year, month), grp in df.groupby(["year", "month"]):
        nh = grp[grp["lat"] > 0]
        sh = grp[grp["lat"] < 0]
        results.append({
            "year": year, "month": month,
            "NH": nh["weighted_val"].sum() / nh["weight"].sum() if len(nh) else np.nan,
            "SH": sh["weighted_val"].sum() / sh["weight"].sum() if len(sh) else np.nan,
            "global": grp["weighted_val"].sum() / grp["weight"].sum() if len(grp) else np.nan,
        })

    ts = pd.DataFrame(results).sort_values(["year", "month"]).reset_index(drop=True)
    ts["date"] = pd.to_datetime(ts[["year", "month"]].assign(day=15))
    return ts


# ── World Model I/O ───────────────────────────────────────────────────

def _load_meta():
    with open(WORLD_MODEL_DIR / "meta.json") as f:
        return json.load(f)


def save_meta(meta):
    with open(WORLD_MODEL_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_hypotheses():
    with open(WORLD_MODEL_DIR / "hypotheses.json") as f:
        return json.load(f)


def save_hypotheses(data):
    with open(WORLD_MODEL_DIR / "hypotheses.json", "w") as f:
        json.dump(data, f, indent=2)


def load_exploration_map():
    with open(WORLD_MODEL_DIR / "exploration_map.json") as f:
        return json.load(f)


def save_exploration_map(data):
    with open(WORLD_MODEL_DIR / "exploration_map.json", "w") as f:
        json.dump(data, f, indent=2)


def load_literature():
    with open(WORLD_MODEL_DIR / "literature.json") as f:
        return json.load(f)


# ── Finding Management ────────────────────────────────────────────────

def create_finding(
    cycle, task_id, summary, statistics, hypothesis_id=None,
    refutes_hypothesis=False, confidence=0.5, script_path=None,
    data_source="CERES_EBAF", method="", figure_paths=None
):
    """Create a structured finding record."""
    meta = _load_meta()
    finding_id = f"E{meta['total_findings'] + 1:03d}"

    finding = {
        "finding_id": finding_id,
        "cycle": cycle,
        "task_id": task_id,
        "summary": summary,
        "statistics": statistics,
        "hypothesis_id": hypothesis_id,
        "refutes_hypothesis": refutes_hypothesis,
        "confidence": confidence,
        "script_path": script_path,
        "data_source": data_source,
        "method": method,
        "figure_paths": figure_paths or [],
        "created_at": datetime.now().isoformat(),
        "scholar_eval_score": None,
        "reproducibility_hash": _compute_hash(script_path) if script_path else None
    }

    # Save finding
    finding_dir = WORLD_MODEL_DIR / "findings" / f"cycle_{cycle:02d}" / f"task_{task_id:02d}"
    finding_dir.mkdir(parents=True, exist_ok=True)
    with open(finding_dir / "findings.json", "w") as f:
        json.dump(finding, f, indent=2)

    # Update meta
    meta["total_findings"] += 1
    save_meta(meta)

    return finding


def _compute_hash(script_path):
    """SHA-256 hash of a script for reproducibility."""
    if script_path and os.path.exists(script_path):
        with open(script_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    return None


# ── Exploration Map Helpers ───────────────────────────────────────────

def get_frontier(n=5):
    """Get the highest-priority unexplored (method, hypothesis) cells."""
    emap = load_exploration_map()
    frontier = []

    for tier_key in ["tier_1_methods", "tier_2_methods"]:
        tier = emap.get(tier_key, {})
        for method_name, method_data in tier.items():
            priority = method_data.get("priority", 1)
            diff = method_data.get("differentiation", 1)
            for key, val in method_data.items():
                if key in ("priority", "differentiation"):
                    continue
                if isinstance(val, dict) and val.get("status") == "not_started":
                    frontier.append({
                        "method": method_name,
                        "hypothesis": key,
                        "priority": priority,
                        "differentiation": diff,
                        "score": priority * 0.6 + diff * 0.4
                    })

    frontier.sort(key=lambda x: x["score"], reverse=True)
    return frontier[:n]


def mark_explored(method, hypothesis, status, finding_id=None, result=None):
    """Mark a (method, hypothesis) cell as explored."""
    emap = load_exploration_map()
    for tier_key in ["tier_1_methods", "tier_2_methods", "tier_3_methods"]:
        tier = emap.get(tier_key, {})
        if method in tier and hypothesis in tier[method]:
            tier[method][hypothesis] = {
                "status": status,
                "finding_id": finding_id,
                "result": result
            }
            break

    # Update summary
    explored = sum(
        1 for tier_key in ["tier_1_methods", "tier_2_methods"]
        for m in emap.get(tier_key, {}).values()
        for k, v in m.items()
        if k not in ("priority", "differentiation") and isinstance(v, dict) and v.get("status") in ("completed", "in_progress")
    )
    total = emap["summary"]["total_cells"]
    emap["summary"]["explored"] = explored
    emap["summary"]["coverage"] = round(explored / max(total, 1), 3)

    save_exploration_map(emap)


# ── Cycle Management ─────────────────────────────────────────────────

def start_cycle():
    """Start a new cycle, return cycle number."""
    meta = _load_meta()
    meta["current_cycle"] += 1
    meta["status"] = "running"
    save_meta(meta)
    return meta["current_cycle"]


def end_cycle(cycle, summary, next_priorities=None):
    """End a cycle and save its summary."""
    summary_data = {
        "cycle": cycle,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "next_priorities": next_priorities or []
    }

    summary_dir = WORLD_MODEL_DIR / "cycle_summaries"
    summary_dir.mkdir(exist_ok=True)
    with open(summary_dir / f"cycle_{cycle:02d}_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    meta = _load_meta()
    meta["status"] = "idle"
    save_meta(meta)


# ── Scholar Eval (simplified) ────────────────────────────────────────

def scholar_eval(finding):
    """Heuristic quality score for a finding. Returns score dict."""
    stats = finding.get("statistics", {})
    scores = {}

    has_pval = "p_value" in stats
    has_effect = "effect_size" in stats or "correlation" in stats
    has_ci = "ci_lower" in stats or "confidence_interval" in stats
    scores["rigor"] = (0.4 * has_pval + 0.3 * has_effect + 0.3 * has_ci)
    scores["reproducibility"] = 1.0 if finding.get("script_path") else 0.0

    if has_pval:
        p = stats.get("p_value", 1.0)
        scores["impact"] = 1.0 if p < 0.01 else (0.5 if p < 0.05 else 0.1)
    else:
        scores["impact"] = 0.3

    scores["coherence"] = 1.0 if finding.get("hypothesis_id") else 0.5

    overall = (
        0.25 * scores["rigor"] +
        0.20 * scores["impact"] +
        0.15 * scores.get("novelty", 0.5) +
        0.15 * scores["reproducibility"] +
        0.10 * scores.get("clarity", 0.7) +
        0.10 * scores["coherence"] +
        0.03 * scores.get("limitations", 0.5) +
        0.02 * scores.get("ethics", 1.0)
    )
    scores["overall"] = round(overall, 3)
    scores["passes_threshold"] = overall >= 0.75

    return scores


# ── Convergence Check ────────────────────────────────────────────────

def check_convergence():
    """Check if all hypotheses have converged."""
    hyp_data = load_hypotheses()
    resolved = []
    unresolved = []

    for h in hyp_data["hypotheses"]:
        cm = h.get("convergence_metrics", {})
        confirming = cm.get("methods_confirming", 0)
        refuting = cm.get("methods_refuting", 0)
        min_methods = cm.get("min_methods_for_convergence", 3)
        subsets = cm.get("data_subsets_tested", 0)
        min_subsets = cm.get("min_subsets_for_convergence", 2)

        if confirming >= min_methods and subsets >= min_subsets:
            resolved.append({"id": h["id"], "result": "supported"})
        elif refuting >= min_methods and subsets >= min_subsets:
            resolved.append({"id": h["id"], "result": "refuted"})
        else:
            unresolved.append(h["id"])

    return {
        "converged": len(unresolved) == 0,
        "resolved": resolved,
        "unresolved": unresolved,
        "progress": len(resolved) / max(len(resolved) + len(unresolved), 1)
    }
