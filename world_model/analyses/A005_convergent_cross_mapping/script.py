#!/usr/bin/env python3
"""
A005: Convergent Cross Mapping — Is Cloud Buffering Causal?

Tests H001 (Cloud Buffering): Does cloud area fraction CAUSALLY influence
albedo stability, or is the SHAP-detected relationship merely correlational?

Convergent Cross Mapping (CCM, Sugihara et al. 2012) detects causality in
nonlinear deterministic systems where Granger causality fails. If X causes Y,
then Y's attractor contains information about X — so cross-mapping from Y's
shadow manifold to X should improve with library size (convergence).

We test:
1. CCM(cloud_area → albedo): Does the albedo manifold contain cloud info?
2. CCM(surface_albedo → cloud_area): Does cloud respond to surface changes?
3. CCM(albedo → cloud_area): Reverse direction
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import spatial, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def time_delay_embedding(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]
    return embedded


def ccm(source, target, dim=3, tau=1, lib_sizes=None, n_reps=50):
    """Convergent Cross Mapping: test if source causally influences target.

    If source → target (source causes target), then target's shadow manifold
    contains information about source. We cross-map FROM target TO source.

    Returns correlation (rho) vs library size. Convergence = causality.
    """
    n = len(source)
    if lib_sizes is None:
        lib_sizes = np.linspace(dim + 2, n - (dim-1)*tau, 15).astype(int)

    results = {"lib_sizes": lib_sizes.tolist(), "rho_mean": [], "rho_std": [],
               "rho_all": []}

    for L in lib_sizes:
        rhos = []
        for rep in range(n_reps):
            # Random subsample of library points
            max_idx = n - (dim - 1) * tau
            if L > max_idx:
                L = max_idx
            lib_idx = np.sort(np.random.choice(max_idx, min(L, max_idx), replace=False))

            # Embed the TARGET (we map FROM target's manifold TO source)
            target_emb = time_delay_embedding(target, dim, tau)

            # Use only library indices
            lib_emb = target_emb[lib_idx]

            # Find nearest neighbors in target's embedded space
            tree = spatial.KDTree(lib_emb)

            # Predict source values via simplex projection
            source_pred = np.zeros(len(lib_idx))
            source_actual = source[lib_idx + (dim-1)*tau]  # Align indices

            for i in range(len(lib_idx)):
                # Find dim+1 nearest neighbors (simplex projection)
                dists, idxs = tree.query(lib_emb[i], k=min(dim + 1, len(lib_emb)))

                if isinstance(dists, float):
                    dists = np.array([dists])
                    idxs = np.array([idxs])

                # Exponential weights
                min_dist = max(dists[0], 1e-10)
                weights = np.exp(-dists / min_dist)
                weights /= weights.sum()

                # Weighted prediction of source
                pred_val = 0.0
                for j, idx in enumerate(idxs):
                    if idx < len(source_actual):
                        pred_val += weights[j] * source_actual[idx]
                source_pred[i] = pred_val

            # Correlation between predicted and actual source
            if np.std(source_pred) > 1e-10 and np.std(source_actual) > 1e-10:
                rho = np.corrcoef(source_pred, source_actual)[0, 1]
                if not np.isnan(rho):
                    rhos.append(rho)

        results["rho_mean"].append(np.mean(rhos) if rhos else 0)
        results["rho_std"].append(np.std(rhos) if rhos else 0)
        results["rho_all"].append(rhos)

    return results


def test_convergence(results):
    """Test if CCM rho converges with library size (= causality)."""
    lib_sizes = np.array(results["lib_sizes"])
    rho_mean = np.array(results["rho_mean"])

    if len(lib_sizes) < 5:
        return {"converges": False, "slope": 0, "p_value": 1.0}

    # Test: does rho increase with library size?
    slope, intercept, r, p, se = stats.linregress(lib_sizes, rho_mean)

    # Also test: is rho at max library significantly > rho at min library?
    rho_first = results["rho_all"][0]
    rho_last = results["rho_all"][-1]
    if rho_first and rho_last:
        t_stat, p_diff = stats.ttest_ind(rho_last, rho_first, alternative='greater')
    else:
        t_stat, p_diff = 0, 1.0

    return {
        "converges": slope > 0 and p < 0.05,
        "slope": float(slope),
        "p_slope": float(p),
        "r_squared": float(r**2),
        "rho_min": float(rho_mean[0]) if len(rho_mean) > 0 else 0,
        "rho_max": float(rho_mean[-1]) if len(rho_mean) > 0 else 0,
        "p_diff": float(p_diff),
    }


def main():
    print("A005: Convergent Cross Mapping — Cloud Buffering Causality")
    print("=" * 60)

    # Compute monthly global means of key variables
    print("Loading CERES data and computing global monthly means...")
    df = load_ceres_data(terminator_filter=True)

    # Area-weighted global monthly means
    from utils import load_zone_weights
    weights = load_zone_weights()
    weight_map = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(weight_map)

    variables = {
        "albedo": "toa_alb_all_mon",
        "cloud_area": "cldarea_total_mon",
        "cloud_tau": "cldtau_total_mon",
    }

    monthly = {}
    for name, col in variables.items():
        if col in df.columns:
            weighted = df.groupby(["year", "month"]).apply(
                lambda g: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "weight"])
                if g[col].notna().any() else np.nan
            ).reset_index(name=name)
            monthly[name] = weighted

    # Merge all variables
    ts = monthly["albedo"]
    for name in list(monthly.keys())[1:]:
        ts = ts.merge(monthly[name], on=["year", "month"], how="inner")

    ts = ts.sort_values(["year", "month"]).reset_index(drop=True)
    ts = ts.dropna()
    print(f"Monthly time series: {len(ts)} months, variables: {list(ts.columns)}")

    # Deseasonalize
    for col in variables.keys():
        seasonal = ts.groupby("month")[col].transform("mean")
        ts[f"{col}_anom"] = ts[col] - seasonal
        ts[f"{col}_std"] = (ts[f"{col}_anom"] - ts[f"{col}_anom"].mean()) / ts[f"{col}_anom"].std()

    # CCM parameters (from attractor reconstruction: dim=3, tau=3)
    dim = 3
    tau = 3
    lib_sizes = np.linspace(dim + 2, len(ts) - (dim-1)*tau - 10, 12).astype(int)
    n_reps = 100

    # ── Test 1: cloud_area → albedo (does cloud cause albedo?) ──
    print(f"\n--- CCM: Cloud Area → Albedo ---")
    print(f"  (Testing if cloud area CAUSES albedo variations)")
    ccm_cloud_to_albedo = ccm(
        ts["cloud_area_std"].values, ts["albedo_std"].values,
        dim=dim, tau=tau, lib_sizes=lib_sizes, n_reps=n_reps
    )
    conv_cloud_alb = test_convergence(ccm_cloud_to_albedo)
    print(f"  Converges: {conv_cloud_alb['converges']} (slope={conv_cloud_alb['slope']:.4f}, p={conv_cloud_alb['p_slope']:.4f})")
    print(f"  rho: {conv_cloud_alb['rho_min']:.3f} → {conv_cloud_alb['rho_max']:.3f}")

    # ── Test 2: albedo → cloud_area (does albedo cause cloud changes?) ──
    print(f"\n--- CCM: Albedo → Cloud Area ---")
    print(f"  (Testing if albedo CAUSES cloud area changes — reverse direction)")
    ccm_albedo_to_cloud = ccm(
        ts["albedo_std"].values, ts["cloud_area_std"].values,
        dim=dim, tau=tau, lib_sizes=lib_sizes, n_reps=n_reps
    )
    conv_alb_cloud = test_convergence(ccm_albedo_to_cloud)
    print(f"  Converges: {conv_alb_cloud['converges']} (slope={conv_alb_cloud['slope']:.4f}, p={conv_alb_cloud['p_slope']:.4f})")
    print(f"  rho: {conv_alb_cloud['rho_min']:.3f} → {conv_alb_cloud['rho_max']:.3f}")

    # ── Test 3: cloud_tau → albedo ──
    ccm_tau_alb = None
    conv_tau_alb = None
    if "cloud_tau_std" in ts.columns and ts["cloud_tau_std"].notna().all():
        print(f"\n--- CCM: Cloud Optical Depth → Albedo ---")
        ccm_tau_alb = ccm(
            ts["cloud_tau_std"].values, ts["albedo_std"].values,
            dim=dim, tau=tau, lib_sizes=lib_sizes, n_reps=n_reps
        )
        conv_tau_alb = test_convergence(ccm_tau_alb)
        print(f"  Converges: {conv_tau_alb['converges']} (slope={conv_tau_alb['slope']:.4f}, p={conv_tau_alb['p_slope']:.4f})")
        print(f"  rho: {conv_tau_alb['rho_min']:.3f} → {conv_tau_alb['rho_max']:.3f}")

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, ccm_result, conv, title in [
        (axes[0], ccm_cloud_to_albedo, conv_cloud_alb, "Cloud Area → Albedo\n(cloud causes albedo?)"),
        (axes[1], ccm_albedo_to_cloud, conv_alb_cloud, "Albedo → Cloud Area\n(albedo causes cloud?)"),
        (axes[2], ccm_tau_alb, conv_tau_alb, "Cloud τ → Albedo\n(optical depth causes albedo?)"),
    ]:
        if ccm_result is None:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        ls = ccm_result["lib_sizes"]
        rm = ccm_result["rho_mean"]
        rs = ccm_result["rho_std"]

        ax.plot(ls, rm, 'o-', color='blue', linewidth=2)
        ax.fill_between(ls, np.array(rm)-np.array(rs), np.array(rm)+np.array(rs),
                        alpha=0.2, color='blue')
        ax.set_xlabel('Library Size')
        ax.set_ylabel('Cross-map skill (ρ)')
        converge_str = f"CONVERGES (p={conv['p_slope']:.3f})" if conv['converges'] else f"No convergence (p={conv['p_slope']:.3f})"
        ax.set_title(f"{title}\n{converge_str}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    fig_path = FIG_DIR / "ccm_cloud_buffering.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    cloud_causes_albedo = conv_cloud_alb["converges"]
    albedo_causes_cloud = conv_alb_cloud["converges"]

    if cloud_causes_albedo and not albedo_causes_cloud:
        interpretation = (
            "UNIDIRECTIONAL CAUSALITY: Cloud area causally drives albedo (CCM converges, p<0.05), "
            "but albedo does NOT causally drive cloud area. This supports H001 (Cloud Buffering) — "
            "clouds are the active agent maintaining albedo stability."
        )
        support = "supports"
    elif cloud_causes_albedo and albedo_causes_cloud:
        interpretation = (
            "BIDIRECTIONAL CAUSALITY: Both cloud→albedo and albedo→cloud CCM converge. "
            "This indicates a coupled feedback system consistent with cloud buffering (H001) "
            "but suggests the relationship is more complex than simple one-way compensation."
        )
        support = "supports"
    elif not cloud_causes_albedo and albedo_causes_cloud:
        interpretation = (
            "REVERSE CAUSALITY: Albedo drives clouds but not vice versa. This CONTRADICTS "
            "the cloud buffering hypothesis — clouds are responding to albedo, not causing stability."
        )
        support = "refutes"
    else:
        interpretation = (
            "NO SIGNIFICANT CAUSAL RELATIONSHIP detected between cloud area and albedo via CCM. "
            "Neither direction shows convergence. The SHAP-detected relationship may be confounded "
            "by a common driver (e.g., large-scale circulation)."
        )
        support = "inconclusive"

    output = {
        "analysis": "A005_convergent_cross_mapping",
        "method": "convergent_cross_mapping",
        "hypothesis": "H001_cloud_buffering",
        "data_source": "CERES_EBAF_Ed4.2.1",
        "n_months": len(ts),
        "parameters": {"dim": dim, "tau": tau, "n_reps": n_reps},
        "results": {
            "cloud_to_albedo": conv_cloud_alb,
            "albedo_to_cloud": conv_alb_cloud,
            "tau_to_albedo": conv_tau_alb,
        },
        "summary": (
            f"CCM analysis: Cloud→Albedo converges={cloud_causes_albedo} "
            f"(rho {conv_cloud_alb['rho_min']:.3f}→{conv_cloud_alb['rho_max']:.3f}), "
            f"Albedo→Cloud converges={albedo_causes_cloud} "
            f"(rho {conv_alb_cloud['rho_min']:.3f}→{conv_alb_cloud['rho_max']:.3f})."
        ),
        "interpretation": interpretation,
        "hypothesis_support": support,
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {OUT_DIR / 'output.json'}")
    print(f"\nInterpretation: {interpretation}")


if __name__ == "__main__":
    main()
