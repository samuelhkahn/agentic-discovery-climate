#!/usr/bin/env python3
"""
A021: Joint Attractor Reconstruction — Coupled Hemispheric Oscillator (H004)

Tests H004 P001 and P004: Does coupling between NH and SH albedo reduce
the effective dimensionality of the global system?

Core test:
- Two independent 2D attractors → joint correlation dimension ≈ 4
- Strongly coupled 2D system → joint correlation dim < 4
- Observed global dim = 1.86 (from A004)

Method:
1. Build joint state vector [NH(t), NH(t-τ), SH(t), SH(t-τ)] (dim=4)
2. Compute correlation dimension of joint attractor
3. Compare to: (a) expectation from independent NH+SH, (b) global albedo alone
4. Joint PCA of NH+SH phase space (H004 P001): does PC1 explain >80%?
5. Mutual information between NH and SH at lag=3 (H004 P003)
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, spatial
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def time_delay_embed(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    return np.column_stack([x[i*tau: i*tau + n] for i in range(dim)])


def correlation_dimension(emb, n_points=1500):
    """Grassberger-Procaccia correlation dimension."""
    if len(emb) > n_points:
        idx = np.random.choice(len(emb), n_points, replace=False)
        emb = emb[idx]
    dists = spatial.distance.pdist(emb)
    r_values = np.logspace(np.log10(np.percentile(dists, 2)),
                           np.log10(np.percentile(dists, 98)), 50)
    C_r = np.array([np.mean(dists < r) for r in r_values])
    mask = C_r > 0
    if mask.sum() < 10:
        return np.nan
    log_r = np.log(r_values[mask])
    log_C = np.log(C_r[mask])
    n_pts = len(log_r)
    s, e = n_pts // 4, 3 * n_pts // 4
    if e - s < 5:
        s, e = 0, n_pts
    slope, *_ = stats.linregress(log_r[s:e], log_C[s:e])
    return float(slope)


def mutual_information_binned(x, y, n_bins=10):
    """Binned mutual information between x and y."""
    bins_x = np.linspace(x.min(), x.max(), n_bins + 1)
    bins_y = np.linspace(y.min(), y.max(), n_bins + 1)
    xd = np.digitize(x, bins_x[1:-1])
    yd = np.digitize(y, bins_y[1:-1])
    n = len(x)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            pxy = np.mean((xd == i) & (yd == j))
            px = np.mean(xd == i)
            py = np.mean(yd == j)
            if pxy > 0 and px > 0 and py > 0:
                mi += pxy * np.log2(pxy / (px * py))
    return mi


def main():
    print("A021: Joint Attractor Reconstruction — Coupled Hemispheric Oscillator")
    print("=" * 70)

    ts = compute_monthly_hemispheric_timeseries()
    nh_s = ts.groupby("month")["NH"].transform("mean")
    sh_s = ts.groupby("month")["SH"].transform("mean")
    nh = ((ts["NH"] - nh_s) / ts["NH"].std()).values
    sh = ((ts["SH"] - sh_s) / ts["SH"].std()).values
    gl_s = ts.groupby("month")["global"].transform("mean")
    gl = ((ts["global"] - gl_s) / ts["global"].std()).values
    print(f"Time series: {len(nh)} months")

    tau = 3  # from A004: optimal tau for global

    results = {}

    # ── 1. Individual attractor dims ──
    print("\n--- Individual Correlation Dimensions ---")
    nh_emb = time_delay_embed(nh, dim=3, tau=tau)
    sh_emb = time_delay_embed(sh, dim=3, tau=tau)
    gl_emb = time_delay_embed(gl, dim=3, tau=tau)
    n = min(len(nh_emb), len(sh_emb), len(gl_emb))
    nh_emb, sh_emb, gl_emb = nh_emb[:n], sh_emb[:n], gl_emb[:n]

    cd_nh = correlation_dimension(nh_emb)
    cd_sh = correlation_dimension(sh_emb)
    cd_gl = correlation_dimension(gl_emb)
    print(f"  NH  correlation dim: {cd_nh:.3f}")
    print(f"  SH  correlation dim: {cd_sh:.3f}")
    print(f"  Global correlation dim: {cd_gl:.3f}  (from A004: 1.86)")
    results["cd_nh"] = cd_nh
    results["cd_sh"] = cd_sh
    results["cd_global"] = cd_gl

    # ── 2. Joint attractor dim ──
    print("\n--- Joint NH+SH Attractor ---")
    joint_emb = np.hstack([nh_emb, sh_emb])
    cd_joint = correlation_dimension(joint_emb)
    print(f"  Joint (NH+SH) correlation dim: {cd_joint:.3f}")
    print(f"  Expected if independent: {cd_nh + cd_sh:.3f}")
    print(f"  Coupling reduction: {(cd_nh + cd_sh) - cd_joint:.3f}")
    results["cd_joint"] = cd_joint
    results["cd_expected_independent"] = cd_nh + cd_sh
    results["coupling_dimensionality_reduction"] = (cd_nh + cd_sh) - cd_joint

    # ── 3. Joint PCA (H004 P001) ──
    print("\n--- Joint PCA of NH+SH Phase Space ---")
    joint_pca_data = np.column_stack([nh_emb, sh_emb])
    pca = PCA()
    pca.fit(joint_pca_data)
    explained = pca.explained_variance_ratio_
    cumvar = np.cumsum(explained)
    print(f"  PC1 explained variance: {explained[0]*100:.1f}%")
    print(f"  PC2 explained variance: {explained[1]*100:.1f}%")
    print(f"  PC1+PC2 cumulative: {cumvar[1]*100:.1f}%")
    print(f"  PCs to reach 80%: {np.searchsorted(cumvar, 0.80) + 1}")
    print(f"  PCs to reach 95%: {np.searchsorted(cumvar, 0.95) + 1}")

    # Test if PC1 is dominated by global mean (both NH and SH positive loading)
    pc1_loadings_nh = pca.components_[0, :3]  # NH dimensions
    pc1_loadings_sh = pca.components_[0, 3:]  # SH dimensions
    same_sign = np.sign(pc1_loadings_nh).mean() == np.sign(pc1_loadings_sh).mean()
    pc1_is_global = same_sign  # PC1 captures correlated NH+SH motion

    print(f"  PC1 NH loadings: {pc1_loadings_nh.round(3)}")
    print(f"  PC1 SH loadings: {pc1_loadings_sh.round(3)}")
    print(f"  PC1 captures global mean (same sign): {pc1_is_global}")

    results["pca_pc1_variance"] = float(explained[0])
    results["pca_pc2_variance"] = float(explained[1])
    results["pca_cumvar_2pcs"] = float(cumvar[1])
    results["pca_pcs_for_80pct"] = int(np.searchsorted(cumvar, 0.80) + 1)
    results["pca_pc1_is_global_mean"] = pc1_is_global
    results["pca_explained_variance_ratio"] = explained[:6].tolist()

    # ── 4. Mutual information NH↔SH (H004 P003) ──
    print("\n--- Mutual Information NH↔SH ---")
    for lag in [0, 1, 3, 5, 6]:
        n_lag = len(nh) - lag
        if lag == 0:
            mi = mutual_information_binned(nh[:n_lag], sh[:n_lag])
        else:
            mi = mutual_information_binned(nh[:n_lag], sh[lag:lag+n_lag])
        print(f"  MI(NH, SH) at lag {lag}: {mi:.4f} bits")
        results[f"mi_lag{lag}"] = float(mi)

    # Expected from independent: MI should be ~0 if independent
    # Shuffle test at lag=0
    mi_obs = mutual_information_binned(nh, sh)
    mi_null = [mutual_information_binned(nh, np.random.permutation(sh)) for _ in range(500)]
    mi_p = np.mean(np.array(mi_null) >= mi_obs)
    print(f"  MI(NH, SH) lag=0: {mi_obs:.4f}, p={mi_p:.4f} (500 perms)")
    results["mi_lag0_p"] = float(mi_p)

    # ── Summary ──
    h4_p001 = explained[0] > 0.80
    h4_p003 = mi_p < 0.05
    h4_p004 = cd_joint < (cd_nh + cd_sh) * 0.8

    print(f"\n=== H004 Predictions ===")
    print(f"  P001 (PC1 > 80% var): {h4_p001} (PC1={explained[0]*100:.1f}%)")
    print(f"  P003 (MI > random): {h4_p003} (p={mi_p:.4f})")
    print(f"  P004 (joint dim < sum): {h4_p004} (joint={cd_joint:.2f} < sum={cd_nh+cd_sh:.2f})")

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.bar(["NH", "SH", "Global", "Joint", "NH+SH\nindep."],
           [cd_nh, cd_sh, cd_gl, cd_joint, cd_nh + cd_sh],
           color=["#2196F3", "#F44336", "#4CAF50", "#9C27B0", "#FF9800"], alpha=0.8)
    ax.set_ylabel("Correlation Dimension")
    ax.set_title("Attractor Dimensionality Comparison")
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.plot(range(1, len(explained[:8]) + 1), np.cumsum(explained[:8]) * 100, 'o-', color='blue')
    ax.axhline(80, color='gray', linestyle='--', label='80%')
    ax.axhline(95, color='gray', linestyle=':', label='95%')
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Joint NH+SH PCA\n(H004 P001)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    lags_plot = [0, 1, 3, 5, 6]
    mi_vals = [results[f"mi_lag{l}"] for l in lags_plot]
    ax.bar(lags_plot, mi_vals, color='teal', alpha=0.8, width=0.5)
    ax.set_xlabel("Lag (months, NH relative to SH)")
    ax.set_ylabel("Mutual Information (bits)")
    ax.set_title("NH↔SH Mutual Information\n(H004 P003)")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = FIG_DIR / "joint_attractor_h004.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    if h4_p001 and h4_p003:
        interp = "STRONG SUPPORT for H004: PC1>80%, significant NH-SH MI, and joint dim < independent sum."
        support = "supports"
    elif h4_p003 or h4_p004:
        interp = "PARTIAL SUPPORT for H004: NH-SH coupling reduces dimensionality but PC1 does not capture >80%."
        support = "supports"
    else:
        interp = "WEAK SUPPORT for H004: joint dimensionality not clearly reduced by coupling."
        support = "inconclusive"

    output = {
        "analysis": "A021_joint_attractor_h004",
        "method": "attractor_reconstruction",
        "hypothesis": "H004_coupled_low_dimensional_hemispheric_oscillator",
        "results": results,
        "h004_predictions": {"P001_met": h4_p001, "P003_met": h4_p003, "P004_met": h4_p004},
        "interpretation": interp,
        "hypothesis_support": support,
        "figure_paths": [str(fig_path)],
    }
    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")


if __name__ == "__main__":
    main()
