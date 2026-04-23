#!/usr/bin/env python3
"""
A027: Attractor Reconstruction for Cloud Buffering (H001)

Tests H001 via dynamical systems analysis:
1. Does cloud area fraction live on the same low-dimensional manifold as albedo?
2. Is the joint cloud+albedo system lower-dimensional than either alone? (coupling)
3. Does the cloud area attractor show manifold structure that mirrors albedo's?
4. Lyapunov exponent of cloud area: is it on a stable attractor?

If H001 correct: cloud area and albedo share an attractor (not independent),
and their joint system has lower dimensionality than the sum of their individual dims.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, spatial
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def time_delay_embedding(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    return np.column_stack([x[i*tau: i*tau+n] for i in range(dim)])


def optimal_tau_mi(x, max_lag=30, n_bins=10):
    """First minimum of mutual information."""
    mi = []
    for lag in range(1, max_lag + 1):
        x1, x2 = x[:-lag], x[lag:]
        h, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        pxy = h / h.sum()
        px, py = pxy.sum(1), pxy.sum(0)
        mask = pxy > 0
        mi.append(np.sum(pxy[mask] * np.log2(pxy[mask] / (px[:, None] * py[None, :])[mask])))
    mi = np.array(mi)
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i + 1
    return int(np.argmin(mi)) + 1


def false_nearest_neighbors(x, tau, max_dim=8, rtol=15.0):
    fnn_fracs = []
    for dim in range(1, max_dim + 1):
        emb = time_delay_embedding(x, dim, tau)
        if dim < max_dim:
            emb_next = time_delay_embedding(x, dim + 1, tau)
            n = min(len(emb), len(emb_next))
        else:
            n = len(emb)
        tree = spatial.KDTree(emb[:n])
        _, idx = tree.query(emb[:n], k=2)
        n_false, n_tot = 0, 0
        for i in range(n):
            nn = idx[i, 1]
            d_cur = np.linalg.norm(emb[i] - emb[nn])
            if d_cur < 1e-10:
                continue
            if dim < max_dim and i < len(emb_next) and nn < len(emb_next):
                d_next = abs(x[i + dim*tau] - x[nn + dim*tau]) if (i+dim*tau < len(x) and nn+dim*tau < len(x)) else 0
                if d_next / d_cur > rtol:
                    n_false += 1
                n_tot += 1
        fnn_fracs.append(n_false / max(n_tot, 1))
        if fnn_fracs[-1] < 0.01:
            break
    return fnn_fracs


def correlation_dimension(emb, n_pts=1500):
    if len(emb) > n_pts:
        idx = np.random.choice(len(emb), n_pts, replace=False)
        emb = emb[idx]
    dists = spatial.distance.pdist(emb)
    r_vals = np.logspace(np.log10(np.percentile(dists, 2)),
                         np.log10(np.percentile(dists, 98)), 50)
    C_r = np.array([np.mean(dists < r) for r in r_vals])
    mask = C_r > 0
    if mask.sum() < 10:
        return np.nan
    log_r, log_C = np.log(r_vals[mask]), np.log(C_r[mask])
    n = len(log_r)
    s, e = n//4, 3*n//4
    if e-s < 5: s, e = 0, n
    slope, *_ = stats.linregress(log_r[s:e], log_C[s:e])
    return float(slope)


def main():
    print("A027: Attractor Reconstruction — Cloud Buffering Manifold (H001)")
    print("=" * 65)

    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    monthly = df.groupby(["year", "month"]).apply(
        lambda g: pd.Series({
            "albedo": np.average(g["toa_alb_all_mon"].dropna(), weights=g.loc[g["toa_alb_all_mon"].notna(), "weight"]) if g["toa_alb_all_mon"].notna().any() else np.nan,
            "cloud": np.average(g["cldarea_total_mon"].dropna(), weights=g.loc[g["cldarea_total_mon"].notna(), "weight"]) if g["cldarea_total_mon"].notna().any() else np.nan,
            "tau": np.average(g["cldtau_total_mon"].dropna(), weights=g.loc[g["cldtau_total_mon"].notna(), "weight"]) if g["cldtau_total_mon"].notna().any() else np.nan,
        })
    ).reset_index().dropna().sort_values(["year", "month"]).reset_index(drop=True)

    for col in ["albedo", "cloud", "tau"]:
        s = monthly.groupby("month")[col].transform("mean")
        monthly[f"{col}_std"] = (monthly[col] - s) / monthly[col].std()

    print(f"Monthly series: {len(monthly)} months")

    results = {}
    tau = 3  # consistent with A004

    for name, col in [("Albedo", "albedo_std"), ("Cloud", "cloud_std"), ("CloudTau", "tau_std")]:
        x = monthly[col].values
        print(f"\n--- {name} ---")
        fnn = false_nearest_neighbors(x, tau, max_dim=6)
        embed_dim = next((i+1 for i, f in enumerate(fnn) if f < 0.05), len(fnn)+1)
        embed_dim = min(embed_dim, 6)
        emb = time_delay_embedding(x, embed_dim, tau)
        cd = correlation_dimension(emb)
        print(f"  FNN: {[f'{f:.3f}' for f in fnn]}")
        print(f"  Embedding dim: {embed_dim}, Correlation dim: {cd:.3f}")
        results[name] = {"embed_dim": embed_dim, "corr_dim": cd, "fnn": fnn}

    # Joint cloud+albedo attractor
    print("\n--- Joint Cloud+Albedo Attractor ---")
    alb_emb = time_delay_embedding(monthly["albedo_std"].values, 3, tau)
    cld_emb = time_delay_embedding(monthly["cloud_std"].values, 3, tau)
    n_joint = min(len(alb_emb), len(cld_emb))
    joint_emb = np.hstack([alb_emb[:n_joint], cld_emb[:n_joint]])
    cd_joint = correlation_dimension(joint_emb)
    cd_expected = results["Albedo"]["corr_dim"] + results["Cloud"]["corr_dim"]
    print(f"  Joint albedo+cloud dim: {cd_joint:.3f}")
    print(f"  Expected if independent: {cd_expected:.3f}")
    print(f"  Coupling reduction: {cd_expected - cd_joint:.3f} ({(cd_expected-cd_joint)/cd_expected*100:.1f}%)")
    results["joint"] = {"cd_joint": cd_joint, "cd_expected": cd_expected,
                        "coupling_reduction": cd_expected - cd_joint}

    # Manifold alignment test: do albedo and cloud manifolds have similar geometry?
    print("\n--- Manifold Alignment Test ---")
    alb_emb_n = alb_emb[:n_joint]
    cld_emb_n = cld_emb[:n_joint]
    # Compute pairwise distances in each manifold
    n_sub = min(500, n_joint)
    idx_sub = np.random.choice(n_joint, n_sub, replace=False)
    d_alb = spatial.distance.pdist(alb_emb_n[idx_sub])
    d_cld = spatial.distance.pdist(cld_emb_n[idx_sub])
    mantel_r, mantel_p = stats.spearmanr(d_alb, d_cld)
    print(f"  Mantel test (distance correlation): r={mantel_r:.4f}, p={mantel_p:.6g}")
    results["mantel_r"] = float(mantel_r)
    results["mantel_p"] = float(mantel_p)

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for name in ["Albedo", "Cloud", "CloudTau"]:
        fnn = results[name]["fnn"]
        ax.plot(range(1, len(fnn)+1), fnn, 'o-', label=name)
    ax.axhline(0.05, color='gray', linestyle='--')
    ax.set_xlabel('Embedding Dim')
    ax.set_ylabel('FNN Fraction')
    ax.set_title('False Nearest Neighbors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    names = ["Albedo", "Cloud", "CloudTau", "Joint", "Expected"]
    cds = [results[n]["corr_dim"] for n in ["Albedo", "Cloud", "CloudTau"]] + \
          [results["joint"]["cd_joint"], results["joint"]["cd_expected"]]
    colors = ['blue', 'green', 'orange', 'purple', 'gray']
    ax.bar(names, cds, color=colors, alpha=0.8)
    ax.set_ylabel('Correlation Dimension')
    ax.set_title('Attractor Dimensionality\n(H001: joint < expected)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    n_plot = min(300, n_joint)
    ax.scatter(d_alb[:n_plot], d_cld[:n_plot], s=5, alpha=0.3)
    ax.set_xlabel('Pairwise distance (albedo manifold)')
    ax.set_ylabel('Pairwise distance (cloud manifold)')
    ax.set_title(f'Manifold Alignment\n(Mantel r={mantel_r:.3f}, p={mantel_p:.3g})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "attractor_h001.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    coupling = results["joint"]["coupling_reduction"] / results["joint"]["cd_expected"]
    if mantel_r > 0.3 and coupling > 0.1:
        interp = f"Strong support for H001: cloud and albedo share manifold structure (Mantel r={mantel_r:.3f}, p={mantel_p:.3g}), joint dim={cd_joint:.2f} vs expected {cd_expected:.2f} (coupling reduces dim by {coupling*100:.0f}%)."
        support = "supports"
    elif mantel_r > 0.2 or coupling > 0.1:
        interp = f"Partial support for H001: manifold alignment r={mantel_r:.3f}, coupling reduction={coupling*100:.0f}%."
        support = "supports_weakly"
    else:
        interp = "Weak manifold alignment. Inconclusive for H001."
        support = "inconclusive"

    output = {
        "analysis": "A027_attractor_h001",
        "method": "attractor_reconstruction",
        "hypothesis": "H001_cloud_buffering",
        "results": results,
        "interpretation": interp,
        "hypothesis_support": support,
        "statistics": {
            "albedo_corr_dim": results["Albedo"]["corr_dim"],
            "cloud_corr_dim": results["Cloud"]["corr_dim"],
            "joint_corr_dim": cd_joint,
            "expected_independent": cd_expected,
            "coupling_dim_reduction": results["joint"]["coupling_reduction"],
            "coupling_reduction_frac": coupling,
            "mantel_r": mantel_r, "mantel_p": float(mantel_p),
        },
        "figure_paths": [str(fig_path)],
    }
    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")


if __name__ == "__main__":
    main()
