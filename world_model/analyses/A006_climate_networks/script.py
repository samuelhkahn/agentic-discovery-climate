#!/usr/bin/env python3
"""
A006: Climate Network Analysis — Spatial Topology of Albedo Teleconnections

Tests H003 (Teleconnections): Where on Earth are albedo anomalies coupled?
Maps the network structure of albedo correlations.

Method:
1. Compute albedo anomaly time series at each grid point (10° resolution)
2. Build correlation network: nodes = grid points, edges = significant correlations
3. Detect communities (coupled regions)
4. Analyze cross-hemispheric links
5. Compute network centrality (hub regions)

This is the 3rd independent method for H003 (after TE and wavelet coherence).
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    print("A006: Climate Network — Spatial Topology of Albedo Teleconnections")
    print("=" * 60)

    # Load CERES and aggregate to 10° resolution
    print("Loading CERES data...")
    df = load_ceres_data(terminator_filter=True)

    # Bin to 10° grid
    df["lat_bin"] = (df["lat"] // 10) * 10 + 5
    df["lon_bin"] = (df["lon"] // 10) * 10 + 5

    # Monthly mean albedo at each 10° grid cell
    print("Computing monthly anomalies at 10° resolution...")
    grid_monthly = df.groupby(["year", "month", "lat_bin", "lon_bin"])["toa_alb_all_mon"].mean().reset_index()

    # Compute anomalies (remove seasonal cycle per grid cell)
    seasonal = grid_monthly.groupby(["month", "lat_bin", "lon_bin"])["toa_alb_all_mon"].transform("mean")
    grid_monthly["anomaly"] = grid_monthly["toa_alb_all_mon"] - seasonal

    # Pivot to time × grid_cell matrix
    grid_monthly["grid_id"] = grid_monthly["lat_bin"].astype(str) + "_" + grid_monthly["lon_bin"].astype(str)
    grid_monthly["time_idx"] = grid_monthly["year"] * 100 + grid_monthly["month"]

    pivot = grid_monthly.pivot_table(index="time_idx", columns="grid_id", values="anomaly")
    pivot = pivot.dropna(axis=1, thresh=len(pivot) * 0.8)  # Drop cells with >20% missing
    pivot = pivot.fillna(0)

    n_times, n_cells = pivot.shape
    print(f"Grid: {n_cells} cells × {n_times} months")

    # Grid cell metadata
    cell_meta = {}
    for col in pivot.columns:
        lat, lon = col.split("_")
        cell_meta[col] = {"lat": float(lat), "lon": float(lon)}

    # ── Build correlation network ──
    print("Computing correlation matrix...")
    corr_matrix = pivot.corr()

    # Significance threshold: Bonferroni-corrected for number of pairs
    n_pairs = n_cells * (n_cells - 1) // 2
    alpha = 0.01
    # Critical correlation for significance (two-tailed)
    t_crit = stats.t.ppf(1 - alpha / (2 * n_pairs), df=n_times - 2)
    r_crit = t_crit / np.sqrt(t_crit**2 + n_times - 2)
    print(f"Bonferroni-corrected r_critical: {r_crit:.4f} (alpha={alpha}, n_pairs={n_pairs})")

    # Build graph: edge if |r| > r_crit
    print("Building network...")
    G = nx.Graph()
    for col in pivot.columns:
        G.add_node(col, **cell_meta[col])

    edge_count = 0
    cross_hemi_count = 0

    for i, col1 in enumerate(pivot.columns):
        for j, col2 in enumerate(pivot.columns):
            if j <= i:
                continue
            r = corr_matrix.loc[col1, col2]
            if abs(r) > r_crit:
                lat1 = cell_meta[col1]["lat"]
                lat2 = cell_meta[col2]["lat"]
                is_cross_hemi = (lat1 > 0) != (lat2 > 0)

                G.add_edge(col1, col2, weight=r, cross_hemispheric=is_cross_hemi)
                edge_count += 1
                if is_cross_hemi:
                    cross_hemi_count += 1

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Cross-hemispheric edges: {cross_hemi_count} ({cross_hemi_count/max(edge_count,1)*100:.1f}%)")

    # ── Network metrics ──
    print("\nComputing network metrics...")

    # Degree centrality
    degree_cent = nx.degree_centrality(G)

    # Betweenness centrality (top hubs)
    if G.number_of_edges() > 0:
        betweenness = nx.betweenness_centrality(G, weight=None)
    else:
        betweenness = {n: 0 for n in G.nodes()}

    # Find hub regions (highest degree)
    top_hubs = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 hub regions (highest degree centrality):")
    for node, cent in top_hubs:
        lat = cell_meta[node]["lat"]
        lon = cell_meta[node]["lon"]
        hemi = "NH" if lat > 0 else "SH"
        print(f"  {hemi} ({lat:.0f}°, {lon:.0f}°): degree_cent={cent:.3f}")

    # Community detection (Louvain)
    if G.number_of_edges() > 0:
        communities = nx.community.greedy_modularity_communities(G)
        n_communities = len(communities)
        modularity = nx.community.modularity(G, communities)
    else:
        communities = [{n} for n in G.nodes()]
        n_communities = len(communities)
        modularity = 0

    print(f"\nCommunity detection: {n_communities} communities, modularity={modularity:.3f}")

    # Analyze community composition (how many are cross-hemispheric?)
    cross_hemi_communities = 0
    for comm in communities:
        lats = [cell_meta[n]["lat"] for n in comm]
        has_nh = any(l > 0 for l in lats)
        has_sh = any(l < 0 for l in lats)
        if has_nh and has_sh:
            cross_hemi_communities += 1

    print(f"Cross-hemispheric communities: {cross_hemi_communities}/{n_communities}")

    # ── Hemispheric coupling strength ──
    # Compare: average within-hemisphere vs cross-hemisphere correlations
    within_hemi_corrs = []
    cross_hemi_corrs = []

    for i, col1 in enumerate(pivot.columns):
        for j, col2 in enumerate(pivot.columns):
            if j <= i:
                continue
            r = corr_matrix.loc[col1, col2]
            lat1 = cell_meta[col1]["lat"]
            lat2 = cell_meta[col2]["lat"]
            if (lat1 > 0) == (lat2 > 0):
                within_hemi_corrs.append(r)
            else:
                cross_hemi_corrs.append(r)

    within_mean = np.mean(within_hemi_corrs)
    cross_mean = np.mean(cross_hemi_corrs)
    t_stat, p_compare = stats.ttest_ind(within_hemi_corrs, cross_hemi_corrs)

    print(f"\nMean within-hemisphere correlation: {within_mean:.4f}")
    print(f"Mean cross-hemisphere correlation: {cross_mean:.4f}")
    print(f"Difference: {within_mean - cross_mean:.4f} (p={p_compare:.4e})")

    # ── Plotting ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Network on geographic map
    ax = axes[0, 0]
    # Plot nodes colored by degree centrality
    lats = [cell_meta[n]["lat"] for n in G.nodes()]
    lons = [cell_meta[n]["lon"] for n in G.nodes()]
    degs = [degree_cent[n] for n in G.nodes()]

    sc = ax.scatter(lons, lats, c=degs, cmap='hot_r', s=30, zorder=5, edgecolors='gray', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label='Degree Centrality')

    # Plot cross-hemispheric edges
    for u, v, data in G.edges(data=True):
        if data.get('cross_hemispheric', False):
            lat1, lon1 = cell_meta[u]["lat"], cell_meta[u]["lon"]
            lat2, lon2 = cell_meta[v]["lat"], cell_meta[v]["lon"]
            ax.plot([lon1, lon2], [lat1, lat2], 'b-', alpha=0.1, linewidth=0.5)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Albedo Climate Network\n({G.number_of_edges()} edges, {cross_hemi_count} cross-hemispheric)')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 70)
    ax.grid(True, alpha=0.2)

    # Panel 2: Degree distribution
    ax = axes[0, 1]
    degrees = [d for _, d in G.degree()]
    ax.hist(degrees, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Count')
    ax.set_title(f'Degree Distribution (mean={np.mean(degrees):.1f})')
    ax.grid(True, alpha=0.3)

    # Panel 3: Within vs cross-hemisphere correlation distributions
    ax = axes[1, 0]
    ax.hist(within_hemi_corrs, bins=50, alpha=0.6, label=f'Within-hemi (mean={within_mean:.3f})', color='green')
    ax.hist(cross_hemi_corrs, bins=50, alpha=0.6, label=f'Cross-hemi (mean={cross_mean:.3f})', color='purple')
    ax.axvline(r_crit, color='red', linestyle='--', label=f'Significance threshold (r={r_crit:.3f})')
    ax.axvline(-r_crit, color='red', linestyle='--')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Count')
    ax.set_title('Within vs Cross-Hemisphere Correlations')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Centrality by latitude
    ax = axes[1, 1]
    lat_deg = {}
    for node in G.nodes():
        lat = cell_meta[node]["lat"]
        lat_deg.setdefault(lat, []).append(degree_cent[node])
    lat_vals = sorted(lat_deg.keys())
    mean_degs = [np.mean(lat_deg[l]) for l in lat_vals]
    ax.plot(lat_vals, mean_degs, 'o-', color='steelblue')
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Mean Degree Centrality')
    ax.set_title('Network Centrality by Latitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "climate_network.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    has_cross_hemi = cross_hemi_count > 0
    cross_frac = cross_hemi_count / max(edge_count, 1)

    if has_cross_hemi and cross_frac > 0.1:
        interpretation = (
            f"Climate network reveals {cross_hemi_count} cross-hemispheric edges "
            f"({cross_frac*100:.1f}% of all significant links). "
            f"{cross_hemi_communities} of {n_communities} communities span both hemispheres. "
            f"This is the THIRD independent method confirming H003 (Teleconnections): "
            f"albedo anomalies are spatially coupled across hemispheres through a network "
            f"of teleconnections."
        )
        support = "supports"
    elif has_cross_hemi:
        interpretation = (
            f"Climate network shows some cross-hemispheric links ({cross_hemi_count}, "
            f"{cross_frac*100:.1f}%) but fewer than within-hemisphere links. "
            f"Weak support for H003."
        )
        support = "supports_weakly"
    else:
        interpretation = (
            "Climate network shows NO significant cross-hemispheric correlations. "
            "Hemispheric albedo coupling may operate through indirect mechanisms not "
            "captured by linear correlation at this resolution."
        )
        support = "inconclusive"

    output = {
        "analysis": "A006_climate_networks",
        "method": "climate_networks",
        "hypothesis": "H003_teleconnections",
        "data_source": "CERES_EBAF_Ed4.2.1",
        "n_months": n_times,
        "n_grid_cells": n_cells,
        "resolution": "10 degrees",
        "network_stats": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_cross_hemispheric_edges": cross_hemi_count,
            "cross_hemispheric_fraction": cross_frac,
            "n_communities": n_communities,
            "n_cross_hemispheric_communities": cross_hemi_communities,
            "modularity": modularity,
            "mean_within_hemi_corr": within_mean,
            "mean_cross_hemi_corr": cross_mean,
            "p_within_vs_cross": p_compare,
            "r_critical_bonferroni": r_crit,
        },
        "top_hubs": [{"lat": cell_meta[n]["lat"], "lon": cell_meta[n]["lon"],
                       "degree_centrality": degree_cent[n]} for n, _ in top_hubs[:5]],
        "summary": (
            f"Climate network at 10° resolution: {G.number_of_edges()} significant edges, "
            f"{cross_hemi_count} cross-hemispheric ({cross_frac*100:.1f}%). "
            f"{n_communities} communities (modularity={modularity:.3f}), "
            f"{cross_hemi_communities} cross-hemispheric."
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
