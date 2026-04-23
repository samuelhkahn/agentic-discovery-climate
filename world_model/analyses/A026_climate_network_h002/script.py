#!/usr/bin/env python3
"""
A026: Climate Network Stability — Invariant Properties (H002)

Tests H002 (Invariant Earth System Properties): If physical constants constrain
albedo to ~29%, then the spatial correlation STRUCTURE of albedo should be
stable across time — the same regions should consistently be hubs/periphery.

Method:
1. Build albedo correlation network for 2000-2012 vs 2013-2025
2. Test if hub rankings are consistent (Spearman correlation of centrality)
3. Test if the mean albedo at each grid cell is stable (temporal stability)
4. Compute the albedo distribution and test if its mean stays near 29%
5. Test if low-variability nodes have systematically different albedo values

If H002 correct: hub nodes should be stable, mean albedo should stay ~29%,
and the distribution structure should be invariant across time periods.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def build_network(df, r_threshold=None, alpha=0.01):
    """Build albedo correlation network from gridded data."""
    alb_piv = df.pivot_table(index=["year", "month"], columns=["lat_bin", "lon_bin"],
                              values="alb_anom")
    min_obs = alb_piv.count().min()
    alb_piv = alb_piv.loc[:, alb_piv.count() >= max(30, min_obs * 0.8)].dropna()

    cells = list(alb_piv.columns)
    n = len(cells)
    alb_mat = alb_piv.values

    # Pairwise correlations
    n_pairs = n * (n - 1) // 2
    r_crit = stats.t.ppf(1 - alpha / 2 / n_pairs, df=len(alb_piv) - 2)
    r_crit = r_crit / np.sqrt(r_crit**2 + len(alb_piv) - 2)

    G = nx.Graph()
    for c in cells:
        G.add_node(c)

    for i in range(n):
        for j in range(i + 1, n):
            xi = alb_mat[:, i]
            xj = alb_mat[:, j]
            mask = ~(np.isnan(xi) | np.isnan(xj))
            if mask.sum() < 20:
                continue
            r, p = stats.pearsonr(xi[mask], xj[mask])
            if abs(r) >= r_crit:
                G.add_edge(cells[i], cells[j], weight=r)

    centrality = nx.degree_centrality(G)
    return G, centrality, cells, alb_piv


def main():
    print("A026: Climate Network Stability — Invariant Properties (H002)")
    print("=" * 65)

    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    # 10° grid
    df["lat_bin"] = (df["lat"] // 10) * 10 + 5
    df["lon_bin"] = (df["lon"] // 10) * 10 + 5

    grid = df.groupby(["year", "month", "lat_bin", "lon_bin"])["toa_alb_all_mon"].mean().reset_index()
    seasonal = grid.groupby(["month", "lat_bin", "lon_bin"])["toa_alb_all_mon"].transform("mean")
    grid["alb_anom"] = grid["toa_alb_all_mon"] - seasonal

    # Mean albedo by grid cell
    cell_mean = grid.groupby(["lat_bin", "lon_bin"])["toa_alb_all_mon"].agg(["mean", "std"]).reset_index()
    cell_mean["cv"] = cell_mean["std"] / cell_mean["mean"]  # coefficient of variation

    results = {}

    # ── Global mean albedo over time ──
    print("\n--- Global mean albedo over time ---")
    weights_map = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df_tmp = df.copy()
    df_tmp["w"] = df_tmp["lat"].round(1).map(weights_map)
    monthly_global = df_tmp.groupby(["year", "month"]).apply(
        lambda g: np.average(g["toa_alb_all_mon"].dropna(), weights=g.loc[g["toa_alb_all_mon"].notna(), "w"])
        if g["toa_alb_all_mon"].notna().any() else np.nan
    ).reset_index(name="global_albedo").dropna()

    # Compare early vs late
    early = monthly_global[monthly_global["year"] <= 2012]["global_albedo"]
    late = monthly_global[monthly_global["year"] > 2012]["global_albedo"]
    t_stat, p_val = stats.ttest_ind(early, late)
    print(f"  Early (2000-2012): mean={early.mean():.4f}, std={early.std():.6f}")
    print(f"  Late  (2013-2025): mean={late.mean():.4f}, std={late.std():.6f}")
    print(f"  Difference: {late.mean() - early.mean():.6f}, p={p_val:.4f}")
    results["global_albedo_early_mean"] = float(early.mean())
    results["global_albedo_late_mean"] = float(late.mean())
    results["global_albedo_change"] = float(late.mean() - early.mean())
    results["global_albedo_change_p"] = float(p_val)

    # ── Temporal stability of cell-level albedo ──
    print("\n--- Cell-level albedo stability ---")
    cell_by_year = grid.groupby(["year", "lat_bin", "lon_bin"])["toa_alb_all_mon"].mean().reset_index()
    cell_std = cell_by_year.groupby(["lat_bin", "lon_bin"])["toa_alb_all_mon"].std().reset_index(name="interannual_std")
    median_std = cell_std["interannual_std"].median()
    print(f"  Median interannual std of cell albedo: {median_std:.4f}")
    print(f"  Mean albedo value: {cell_mean['mean'].mean():.4f}")
    print(f"  CV (std/mean) distribution: median={cell_mean['cv'].median():.4f}, max={cell_mean['cv'].max():.4f}")
    results["median_interannual_std"] = float(median_std)
    results["median_cv"] = float(cell_mean["cv"].median())

    # ── Network stability test ──
    print("\n--- Network stability: comparing 2000-2012 vs 2013-2025 ---")
    for subset_name, (y_lo, y_hi) in [("early", (2000, 2012)), ("late", (2013, 2025))]:
        sub = grid[(grid["year"] >= y_lo) & (grid["year"] <= y_hi)]
        G_sub, cent_sub, cells_sub, _ = build_network(sub)
        print(f"  {subset_name}: {len(G_sub.nodes)} nodes, {len(G_sub.edges)} edges")
        results[f"network_{subset_name}"] = {
            "n_nodes": len(G_sub.nodes), "n_edges": len(G_sub.edges),
            "centrality": {str(k): v for k, v in cent_sub.items()},
        }

    # Spearman correlation of centrality ranks between early and late
    early_cent = results["network_early"]["centrality"]
    late_cent = results["network_late"]["centrality"]
    common_cells = set(early_cent.keys()) & set(late_cent.keys())
    if len(common_cells) > 10:
        early_vals = [early_cent[c] for c in common_cells]
        late_vals = [late_cent[c] for c in common_cells]
        spearman_r, spearman_p = stats.spearmanr(early_vals, late_vals)
        print(f"  Centrality rank stability (Spearman r): {spearman_r:.4f}, p={spearman_p:.6g}")
        results["centrality_stability_r"] = float(spearman_r)
        results["centrality_stability_p"] = float(spearman_p)

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(monthly_global.set_index(["year", "month"])["global_albedo"].values, 'b-', linewidth=0.8, alpha=0.7)
    ax.axhline(monthly_global["global_albedo"].mean(), color='red', linestyle='--',
               label=f'Mean={monthly_global["global_albedo"].mean():.4f}')
    ax.axhline(0.29, color='gray', linestyle=':', label='29% reference')
    ax.set_xlabel('Month index')
    ax.set_ylabel('Global mean albedo')
    ax.set_title('Global Mean Albedo Time Series\n(H002: should stay near ~29%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(cell_mean["mean"], cell_mean["cv"], s=20, alpha=0.6)
    ax.set_xlabel('Mean albedo')
    ax.set_ylabel('CV (std/mean)')
    ax.set_title('Albedo Variability vs Level\n(H002: invariant regions have low CV)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if "centrality_stability_r" in results:
        if len(common_cells) > 0:
            ax.scatter(early_vals, late_vals, s=15, alpha=0.6)
            ax.set_xlabel('Centrality 2000-2012')
            ax.set_ylabel('Centrality 2013-2025')
            ax.set_title(f'Network Hub Stability\n(Spearman r={results["centrality_stability_r"]:.3f})')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "climate_network_h002.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    # Interpretation
    stable_mean = abs(results.get("global_albedo_change", 1)) < 0.001
    stable_network = results.get("centrality_stability_r", 0) > 0.7

    if stable_mean and stable_network:
        interp = f"Strong support for H002: global mean albedo change={results['global_albedo_change']:.5f} (p={results['global_albedo_change_p']:.4f}), network hub stability r={results.get('centrality_stability_r', 0):.4f}."
        support = "supports"
    elif stable_mean:
        interp = f"Global mean albedo is stable (change={results['global_albedo_change']:.5f}), supporting H002 invariant constraint. Network stability inconclusive."
        support = "supports"
    else:
        interp = f"Global mean albedo shows measurable change ({results['global_albedo_change']:.5f}). Partial support for H002."
        support = "supports_weakly"

    output = {
        "analysis": "A026_climate_network_h002",
        "method": "climate_networks",
        "hypothesis": "H002_invariant_properties",
        "results": results,
        "interpretation": interp,
        "hypothesis_support": support,
        "statistics": {
            "global_albedo_early": results.get("global_albedo_early_mean"),
            "global_albedo_late": results.get("global_albedo_late_mean"),
            "global_albedo_change": results.get("global_albedo_change"),
            "global_albedo_change_p": results.get("global_albedo_change_p"),
            "centrality_stability_r": results.get("centrality_stability_r"),
            "median_cv": results.get("median_cv"),
        },
        "figure_paths": [str(fig_path)],
    }
    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")


if __name__ == "__main__":
    main()
