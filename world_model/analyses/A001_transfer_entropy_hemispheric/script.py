#!/usr/bin/env python3
"""
A001: Transfer Entropy Analysis — NH↔SH Albedo Information Flow

Tests H003 (Teleconnections): Is there directional information transfer
between Northern and Southern hemisphere albedo?

Transfer entropy TE(X→Y) measures how much knowing the past of X reduces
uncertainty about Y's future, beyond what Y's own past provides.
If TE(NH→SH) > TE(SH→NH), information flows preferentially from NH to SH.

Method: Kernel-density-based transfer entropy estimation with
permutation-based significance testing.
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── Transfer Entropy Implementation ──────────────────────────────────

def transfer_entropy_knn(source, target, lag=1, k=5):
    """Compute transfer entropy TE(source→target) using k-NN estimator.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)

    Uses the Kraskov-Stögbauer-Grassberger (KSG) estimator via
    nearest-neighbor distances.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(target) - lag

    # Build state vectors
    y_future = target[lag:]       # Y(t)
    y_past = target[:n]           # Y(t-lag)
    x_past = source[:n]           # X(t-lag)

    # Joint space: (y_future, y_past, x_past)
    joint_3d = np.column_stack([y_future, y_past, x_past])
    # Marginal: (y_future, y_past)
    joint_2d = np.column_stack([y_future, y_past])
    # Conditioning: (y_past, x_past)
    cond_2d = np.column_stack([y_past, x_past])
    # y_past alone
    cond_1d = y_past.reshape(-1, 1)

    # KSG estimator: TE = psi(k) - <psi(n_xz+1) - psi(n_z+1) + psi(n_z2+1)>
    # Simplified: use difference of conditional entropies via KNN distances

    # Fit KNN in joint space
    nn_joint = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
    nn_joint.fit(joint_3d)
    distances, _ = nn_joint.kneighbors(joint_3d)
    eps = distances[:, -1]  # distance to k-th neighbor

    # Count neighbors within eps in each marginal space
    from scipy.spatial import KDTree

    tree_yz = KDTree(joint_2d)
    tree_xz = KDTree(cond_2d)
    tree_z = KDTree(cond_1d)

    n_yz = np.array([len(tree_yz.query_ball_point(joint_2d[i], eps[i]+1e-15)) - 1 for i in range(n)])
    n_xz = np.array([len(tree_xz.query_ball_point(cond_2d[i], eps[i]+1e-15)) - 1 for i in range(n)])
    n_z = np.array([len(tree_z.query_ball_point(cond_1d[i], eps[i]+1e-15)) - 1 for i in range(n)])

    # TE = psi(k) - mean(psi(n_yz+1) - psi(n_z+1) + psi(n_xz+1) - psi(n_z+1))
    # Simplified: TE ≈ mean(digamma(n_z+1) - digamma(n_xz+1) - digamma(n_yz+1) + digamma(k))
    from scipy.special import digamma

    te = digamma(k) + np.mean(digamma(n_z + 1) - digamma(n_xz + 1) - digamma(n_yz + 1))

    return te


def transfer_entropy_binned(source, target, lag=1, n_bins=8):
    """Compute transfer entropy using histogram-based estimation.

    Simpler and more robust for short time series (~300 points).
    """
    n = len(target) - lag

    y_future = target[lag:]
    y_past = target[:n]
    x_past = source[:n]

    # Discretize into bins
    y_future_d = np.digitize(y_future, np.linspace(y_future.min(), y_future.max(), n_bins))
    y_past_d = np.digitize(y_past, np.linspace(y_past.min(), y_past.max(), n_bins))
    x_past_d = np.digitize(x_past, np.linspace(x_past.min(), x_past.max(), n_bins))

    # Compute joint and marginal probabilities
    # TE = sum p(y_f, y_p, x_p) * log(p(y_f | y_p, x_p) / p(y_f | y_p))

    te = 0.0
    total = len(y_future_d)

    for yf in range(1, n_bins + 1):
        for yp in range(1, n_bins + 1):
            for xp in range(1, n_bins + 1):
                # Joint count
                mask_joint = (y_future_d == yf) & (y_past_d == yp) & (x_past_d == xp)
                p_joint = mask_joint.sum() / total

                if p_joint == 0:
                    continue

                # p(y_f | y_p, x_p)
                mask_cond = (y_past_d == yp) & (x_past_d == xp)
                p_cond = mask_cond.sum() / total
                p_yf_given_ypxp = p_joint / p_cond if p_cond > 0 else 0

                # p(y_f | y_p)
                mask_yp = (y_past_d == yp)
                p_yp = mask_yp.sum() / total
                mask_yf_yp = (y_future_d == yf) & (y_past_d == yp)
                p_yf_yp = mask_yf_yp.sum() / total
                p_yf_given_yp = p_yf_yp / p_yp if p_yp > 0 else 0

                if p_yf_given_ypxp > 0 and p_yf_given_yp > 0:
                    te += p_joint * np.log2(p_yf_given_ypxp / p_yf_given_yp)

    return te


def permutation_test(source, target, te_func, n_perms=1000, lag=1, **kwargs):
    """Permutation test for transfer entropy significance."""
    te_observed = te_func(source, target, lag=lag, **kwargs)

    te_null = np.zeros(n_perms)
    for i in range(n_perms):
        # Shuffle source to destroy temporal coupling while preserving marginal distribution
        source_shuffled = np.random.permutation(source)
        te_null[i] = te_func(source_shuffled, target, lag=lag, **kwargs)

    p_value = np.mean(te_null >= te_observed)

    return {
        "te_observed": te_observed,
        "te_null_mean": te_null.mean(),
        "te_null_std": te_null.std(),
        "p_value": p_value,
        "z_score": (te_observed - te_null.mean()) / te_null.std() if te_null.std() > 0 else 0,
        "te_null_distribution": te_null
    }


# ── Main Analysis ────────────────────────────────────────────────────

def main():
    print("A001: Transfer Entropy — NH↔SH Information Flow")
    print("=" * 60)

    # Load hemispheric albedo time series
    ts = compute_monthly_hemispheric_timeseries()
    print(f"Time series: {len(ts)} months ({ts.date.min().date()} to {ts.date.max().date()})")

    nh = ts["NH"].values
    sh = ts["SH"].values

    # Remove seasonal cycle (compute anomalies)
    nh_seasonal = ts.groupby("month")["NH"].transform("mean")
    sh_seasonal = ts.groupby("month")["SH"].transform("mean")
    nh_anom = (ts["NH"] - nh_seasonal).values
    sh_anom = (ts["SH"] - sh_seasonal).values

    # Standardize
    nh_anom = (nh_anom - nh_anom.mean()) / nh_anom.std()
    sh_anom = (sh_anom - sh_anom.mean()) / sh_anom.std()

    print(f"NH anomaly std: {nh_anom.std():.4f}, SH anomaly std: {sh_anom.std():.4f}")

    results = {}

    # Test multiple lags
    for lag in [1, 2, 3, 6, 12]:
        print(f"\n--- Lag = {lag} months ---")

        # TE(NH→SH): Does NH past inform SH future?
        print(f"  Computing TE(NH→SH)...")
        nh_to_sh = permutation_test(nh_anom, sh_anom, transfer_entropy_binned,
                                     n_perms=2000, lag=lag, n_bins=6)

        # TE(SH→NH): Does SH past inform NH future?
        print(f"  Computing TE(SH→NH)...")
        sh_to_nh = permutation_test(sh_anom, nh_anom, transfer_entropy_binned,
                                     n_perms=2000, lag=lag, n_bins=6)

        print(f"  TE(NH→SH) = {nh_to_sh['te_observed']:.6f} (p={nh_to_sh['p_value']:.4f}, z={nh_to_sh['z_score']:.2f})")
        print(f"  TE(SH→NH) = {sh_to_nh['te_observed']:.6f} (p={sh_to_nh['p_value']:.4f}, z={sh_to_nh['z_score']:.2f})")

        # Net information flow
        net_flow = nh_to_sh['te_observed'] - sh_to_nh['te_observed']
        print(f"  Net flow (NH→SH - SH→NH) = {net_flow:.6f}")

        results[f"lag_{lag}"] = {
            "TE_NH_to_SH": nh_to_sh['te_observed'],
            "TE_SH_to_NH": sh_to_nh['te_observed'],
            "p_NH_to_SH": nh_to_sh['p_value'],
            "p_SH_to_NH": sh_to_nh['p_value'],
            "z_NH_to_SH": nh_to_sh['z_score'],
            "z_SH_to_NH": sh_to_nh['z_score'],
            "net_flow": net_flow,
        }

    # Also compute on raw (non-deseasonalized) data at lag=1
    print(f"\n--- Raw (with seasonal cycle), Lag = 1 ---")
    nh_raw = (nh - nh.mean()) / nh.std()
    sh_raw = (sh - sh.mean()) / sh.std()

    nh_to_sh_raw = permutation_test(nh_raw, sh_raw, transfer_entropy_binned,
                                     n_perms=2000, lag=1, n_bins=6)
    sh_to_nh_raw = permutation_test(sh_raw, nh_raw, transfer_entropy_binned,
                                     n_perms=2000, lag=1, n_bins=6)
    print(f"  TE(NH→SH) = {nh_to_sh_raw['te_observed']:.6f} (p={nh_to_sh_raw['p_value']:.4f})")
    print(f"  TE(SH→NH) = {sh_to_nh_raw['te_observed']:.6f} (p={sh_to_nh_raw['p_value']:.4f})")

    results["raw_lag_1"] = {
        "TE_NH_to_SH": nh_to_sh_raw['te_observed'],
        "TE_SH_to_NH": sh_to_nh_raw['te_observed'],
        "p_NH_to_SH": nh_to_sh_raw['p_value'],
        "p_SH_to_NH": sh_to_nh_raw['p_value'],
    }

    # ── Plotting ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Hemispheric anomaly time series
    ax = axes[0, 0]
    ax.plot(ts.date, nh_anom, label='NH anomaly', alpha=0.7)
    ax.plot(ts.date, sh_anom, label='SH anomaly', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Albedo anomaly (standardized)')
    ax.set_title('Hemispheric Albedo Anomalies')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: TE vs lag
    ax = axes[0, 1]
    lags = [1, 2, 3, 6, 12]
    te_nh_sh = [results[f"lag_{l}"]["TE_NH_to_SH"] for l in lags]
    te_sh_nh = [results[f"lag_{l}"]["TE_SH_to_NH"] for l in lags]
    ax.plot(lags, te_nh_sh, 'o-', label='TE(NH→SH)', color='blue')
    ax.plot(lags, te_sh_nh, 's-', label='TE(SH→NH)', color='red')
    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('Transfer Entropy (bits)')
    ax.set_title('Transfer Entropy vs Lag')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Significance (p-values)
    ax = axes[1, 0]
    p_nh_sh = [results[f"lag_{l}"]["p_NH_to_SH"] for l in lags]
    p_sh_nh = [results[f"lag_{l}"]["p_SH_to_NH"] for l in lags]
    ax.semilogy(lags, p_nh_sh, 'o-', label='p(NH→SH)', color='blue')
    ax.semilogy(lags, p_sh_nh, 's-', label='p(SH→NH)', color='red')
    ax.axhline(0.05, color='gray', linestyle='--', label='p=0.05')
    ax.axhline(0.01, color='gray', linestyle=':', label='p=0.01')
    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('p-value')
    ax.set_title('Significance of Transfer Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Net information flow
    ax = axes[1, 1]
    net = [results[f"lag_{l}"]["net_flow"] for l in lags]
    colors = ['blue' if n > 0 else 'red' for n in net]
    ax.bar(lags, net, color=colors, alpha=0.7, width=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('Net TE (NH→SH) - (SH→NH)')
    ax.set_title('Net Information Flow Direction')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "transfer_entropy_hemispheric.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Save results ──
    # Find most significant lag
    best_lag = min(results.keys(),
                   key=lambda k: min(results[k].get("p_NH_to_SH", 1), results[k].get("p_SH_to_NH", 1))
                   if k != "raw_lag_1" else 1)
    best = results[best_lag]

    output = {
        "analysis": "A001_transfer_entropy_hemispheric",
        "method": "transfer_entropy",
        "hypothesis": "H003_teleconnections",
        "data_source": "CERES_EBAF_Ed4.2.1",
        "time_range": f"{ts.date.min().date()} to {ts.date.max().date()}",
        "n_months": len(ts),
        "summary": (
            f"Transfer entropy analysis between NH and SH albedo anomalies. "
            f"Best result at {best_lag}: TE(NH→SH)={best['TE_NH_to_SH']:.6f} (p={best['p_NH_to_SH']:.4f}), "
            f"TE(SH→NH)={best['TE_SH_to_NH']:.6f} (p={best['p_SH_to_NH']:.4f}). "
            f"Net flow: {best['net_flow']:.6f}."
        ),
        "results_by_lag": results,
        "interpretation": "",
        "figure_paths": [str(fig_path)]
    }

    # Interpret
    any_significant = any(
        results[k].get("p_NH_to_SH", 1) < 0.05 or results[k].get("p_SH_to_NH", 1) < 0.05
        for k in results if k != "raw_lag_1"
    )

    if any_significant:
        output["interpretation"] = (
            "SIGNIFICANT information transfer detected between hemispheres. "
            "This supports H003 (Teleconnections) — hemispheric albedo anomalies are not independent."
        )
        output["hypothesis_support"] = "supports"
    else:
        output["interpretation"] = (
            "NO significant directional information transfer detected between hemispheric albedo anomalies "
            "at any tested lag (1-12 months). This is consistent with the U-Net null result (no regional control). "
            "Hemispheric albedo anomalies may be driven by independent local processes rather than teleconnections."
        )
        output["hypothesis_support"] = "weakly_refutes"

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {OUT_DIR / 'output.json'}")
    print(f"\nInterpretation: {output['interpretation']}")


if __name__ == "__main__":
    main()
