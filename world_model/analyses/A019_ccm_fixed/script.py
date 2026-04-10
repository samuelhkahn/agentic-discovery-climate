#!/usr/bin/env python3
"""
A019: Fixed Convergent Cross Mapping — Cloud Buffering Causality (H001)

Fixes from A005:
1. Leave-one-out: query point excluded from its own neighborhood
2. Proper simplex projection with temporal separation
3. Library size convergence tested with independent prediction set

CCM logic: If X causally drives Y, then Y's reconstructed attractor
contains information about X. So we map FROM Y's manifold TO predict X.
Convergence of prediction skill with library size = causality.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import spatial, stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"


def time_delay_embedding(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]
    return embedded


def ccm_fixed(source, target, dim=3, tau=1, lib_sizes=None, n_reps=100):
    """Convergent Cross Mapping with proper leave-one-out and temporal exclusion.

    Tests: does source cause target?
    Method: cross-map FROM target's manifold TO predict source.
    If rho increases with library size → source causes target.
    """
    N = len(source)
    max_idx = N - (dim - 1) * tau

    # Embed the TARGET
    target_emb = time_delay_embedding(target, dim, tau)
    # Align source values with embedding indices
    source_aligned = source[(dim - 1) * tau:]
    n_emb = len(target_emb)

    if lib_sizes is None:
        lib_sizes = np.unique(np.linspace(dim + 2, n_emb - 5, 12).astype(int))

    # Temporal exclusion window (avoid autocorrelation contamination)
    excl_window = dim * tau + 1

    results = {"lib_sizes": lib_sizes.tolist(), "rho_mean": [], "rho_std": []}

    for L in lib_sizes:
        rhos = []
        for rep in range(n_reps):
            # Split into library and prediction sets
            all_idx = np.arange(n_emb)
            lib_idx = np.sort(np.random.choice(n_emb, min(L, n_emb), replace=False))
            # Prediction set: everything NOT in library (with temporal buffer)
            pred_mask = np.ones(n_emb, dtype=bool)
            for li in lib_idx:
                pred_mask[max(0, li - excl_window):min(n_emb, li + excl_window + 1)] = False
            pred_idx = np.where(pred_mask)[0]

            if len(pred_idx) < 5:
                # If prediction set too small, use leave-one-out on library
                pred_idx = lib_idx
                loo_mode = True
            else:
                loo_mode = False

            # Build KD-tree from library
            lib_emb = target_emb[lib_idx]
            tree = spatial.KDTree(lib_emb)

            source_pred = []
            source_actual = []

            for pi in pred_idx:
                query = target_emb[pi]

                # Find dim+1 nearest neighbors in library
                k = min(dim + 1, len(lib_idx))
                dists, nn_idx_in_lib = tree.query(query, k=k)

                if isinstance(dists, float):
                    dists = np.array([dists])
                    nn_idx_in_lib = np.array([nn_idx_in_lib])

                # In LOO mode, skip self
                if loo_mode:
                    valid = nn_idx_in_lib != np.searchsorted(lib_idx, pi)
                    if valid.sum() < 2:
                        continue
                    dists = dists[valid][:dim + 1]
                    nn_idx_in_lib = nn_idx_in_lib[valid][:dim + 1]

                # Also enforce temporal exclusion on neighbors
                nn_orig_idx = lib_idx[nn_idx_in_lib]
                temporal_ok = np.abs(nn_orig_idx - pi) > excl_window
                if temporal_ok.sum() < 2:
                    continue  # Not enough temporally separated neighbors
                dists = dists[temporal_ok]
                nn_orig_idx = nn_orig_idx[temporal_ok]

                if len(dists) == 0:
                    continue

                # Exponential weights
                min_dist = max(dists[0], 1e-15)
                weights = np.exp(-dists / min_dist)
                weights /= weights.sum()

                # Predict source value
                pred = np.sum(weights * source_aligned[nn_orig_idx])
                source_pred.append(pred)
                source_actual.append(source_aligned[pi])

            if len(source_pred) > 5:
                sp = np.array(source_pred)
                sa = np.array(source_actual)
                if np.std(sp) > 1e-10 and np.std(sa) > 1e-10:
                    rho = np.corrcoef(sp, sa)[0, 1]
                    if not np.isnan(rho):
                        rhos.append(rho)

        results["rho_mean"].append(np.mean(rhos) if rhos else 0)
        results["rho_std"].append(np.std(rhos) if rhos else 0)

    return results


def test_convergence(results):
    """Test if CCM rho converges with library size."""
    lib_sizes = np.array(results["lib_sizes"])
    rho_mean = np.array(results["rho_mean"])

    if len(rho_mean) < 5 or all(r == 0 for r in rho_mean):
        return {"converges": False, "slope": 0, "p_value": 1.0, "rho_max": 0, "rho_min": 0}

    slope, intercept, r, p, se = stats.linregress(lib_sizes, rho_mean)

    return {
        "converges": slope > 0 and p < 0.05,
        "slope": float(slope),
        "p_slope": float(p),
        "rho_min": float(rho_mean[0]),
        "rho_max": float(rho_mean[-1]),
        "rho_increase": float(rho_mean[-1] - rho_mean[0]),
    }


def main():
    print("A019: Fixed CCM — Cloud Buffering Causality")
    print("=" * 60)

    # Load monthly global means
    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    variables = {"albedo": "toa_alb_all_mon", "cloud_area": "cldarea_total_mon",
                 "cloud_tau": "cldtau_total_mon"}
    variables = {k: v for k, v in variables.items() if v in df.columns}

    monthly = df.groupby(["year", "month"]).apply(
        lambda g: pd.Series({
            name: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "weight"])
            if g[col].notna().any() else np.nan
            for name, col in variables.items()
        })
    ).reset_index().dropna().sort_values(["year", "month"]).reset_index(drop=True)

    # Deseasonalize and standardize
    for col in variables.keys():
        seasonal = monthly.groupby("month")[col].transform("mean")
        monthly[f"{col}_anom"] = monthly[col] - seasonal
        monthly[f"{col}_std"] = (monthly[f"{col}_anom"] - monthly[f"{col}_anom"].mean()) / monthly[f"{col}_anom"].std()

    print(f"Monthly series: {len(monthly)} months")

    # CCM parameters (from attractor reconstruction)
    dim = 3
    tau = 3
    n_reps = 100

    results = {}

    # ── Test 1: Cloud area → albedo ──
    print("\n--- CCM: Cloud Area → Albedo ---")
    print("  (Map FROM albedo manifold TO predict cloud area)")
    ccm1 = ccm_fixed(monthly["cloud_area_std"].values, monthly["albedo_std"].values,
                     dim=dim, tau=tau, n_reps=n_reps)
    conv1 = test_convergence(ccm1)
    print(f"  Converges: {conv1['converges']} (slope={conv1['slope']:.5f}, p={conv1['p_slope']:.4f})")
    print(f"  ρ: {conv1['rho_min']:.3f} → {conv1['rho_max']:.3f}")
    results["cloud_causes_albedo"] = {"ccm": ccm1, "convergence": conv1}

    # ── Test 2: Albedo → cloud area ──
    print("\n--- CCM: Albedo → Cloud Area ---")
    print("  (Map FROM cloud manifold TO predict albedo)")
    ccm2 = ccm_fixed(monthly["albedo_std"].values, monthly["cloud_area_std"].values,
                     dim=dim, tau=tau, n_reps=n_reps)
    conv2 = test_convergence(ccm2)
    print(f"  Converges: {conv2['converges']} (slope={conv2['slope']:.5f}, p={conv2['p_slope']:.4f})")
    print(f"  ρ: {conv2['rho_min']:.3f} → {conv2['rho_max']:.3f}")
    results["albedo_causes_cloud"] = {"ccm": ccm2, "convergence": conv2}

    # ── Test 3: Cloud tau → albedo ──
    if "cloud_tau" in variables:
        print("\n--- CCM: Cloud Tau → Albedo ---")
        ccm3 = ccm_fixed(monthly["cloud_tau_std"].values, monthly["albedo_std"].values,
                         dim=dim, tau=tau, n_reps=n_reps)
        conv3 = test_convergence(ccm3)
        print(f"  Converges: {conv3['converges']} (slope={conv3['slope']:.5f}, p={conv3['p_slope']:.4f})")
        print(f"  ρ: {conv3['rho_min']:.3f} → {conv3['rho_max']:.3f}")
        results["cloud_tau_causes_albedo"] = {"ccm": ccm3, "convergence": conv3}

    # ── NH/SH subset test ──
    for hemi, lat_cond in [("NH", "> 0"), ("SH", "< 0")]:
        hdf = df.query(f"lat {lat_cond}")
        hm = hdf.groupby(["year", "month"]).apply(
            lambda g: pd.Series({
                name: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "weight"])
                if g[col].notna().any() else np.nan
                for name, col in variables.items()
            })
        ).reset_index().dropna().sort_values(["year", "month"]).reset_index(drop=True)

        for col in variables.keys():
            seasonal = hm.groupby("month")[col].transform("mean")
            hm[f"{col}_std"] = ((hm[col] - seasonal) - (hm[col] - seasonal).mean()) / (hm[col] - seasonal).std()

        print(f"\n--- CCM {hemi}: Cloud Area → Albedo ---")
        ccm_h = ccm_fixed(hm["cloud_area_std"].values, hm["albedo_std"].values,
                          dim=dim, tau=tau, n_reps=80)
        conv_h = test_convergence(ccm_h)
        print(f"  Converges: {conv_h['converges']} (ρ: {conv_h['rho_min']:.3f} → {conv_h['rho_max']:.3f})")
        results[f"cloud_causes_albedo_{hemi}"] = {"convergence": conv_h}

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (label, key) in zip(axes, [
        ("Cloud Area → Albedo", "cloud_causes_albedo"),
        ("Albedo → Cloud Area", "albedo_causes_cloud"),
        ("Cloud τ → Albedo", "cloud_tau_causes_albedo"),
    ]):
        if key not in results:
            ax.text(0.5, 0.5, "N/A", ha='center', transform=ax.transAxes)
            ax.set_title(label)
            continue

        ccm_r = results[key]["ccm"]
        conv_r = results[key]["convergence"]
        ls = ccm_r["lib_sizes"]
        rm = ccm_r["rho_mean"]
        rs = ccm_r["rho_std"]

        ax.plot(ls, rm, 'o-', color='blue', linewidth=2)
        ax.fill_between(ls, np.array(rm) - np.array(rs), np.array(rm) + np.array(rs),
                        alpha=0.2, color='blue')
        conv_str = f"CONVERGES (p={conv_r['p_slope']:.3f})" if conv_r["converges"] else f"No convergence (p={conv_r['p_slope']:.3f})"
        ax.set_xlabel('Library Size')
        ax.set_ylabel('Cross-map skill (ρ)')
        ax.set_title(f"{label}\n{conv_str}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "ccm_fixed.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    c_causes_a = results["cloud_causes_albedo"]["convergence"]["converges"]
    a_causes_c = results["albedo_causes_cloud"]["convergence"]["converges"]

    if c_causes_a and not a_causes_c:
        interp = "UNIDIRECTIONAL: Cloud area causally drives albedo but not vice versa. Supports H001."
        support = "supports"
    elif c_causes_a and a_causes_c:
        interp = "BIDIRECTIONAL: Both cloud→albedo and albedo→cloud show causal coupling. Consistent with adversarial finding (feedback system). Supports H001."
        support = "supports"
    elif not c_causes_a and a_causes_c:
        interp = "REVERSE: Albedo drives clouds but not vice versa. Contradicts H001."
        support = "refutes"
    else:
        interp = "NO CAUSAL RELATIONSHIP detected. CCM does not support H001."
        support = "inconclusive"

    output = {
        "analysis": "A019_ccm_fixed", "method": "convergent_cross_mapping",
        "hypothesis": "H001_cloud_buffering",
        "results": {k: v["convergence"] for k, v in results.items()},
        "interpretation": interp, "hypothesis_support": support,
        "statistics": {
            "cloud_to_albedo_converges": c_causes_a,
            "albedo_to_cloud_converges": a_causes_c,
            "cloud_to_albedo_rho_max": results["cloud_causes_albedo"]["convergence"]["rho_max"],
            "p_value": results["cloud_causes_albedo"]["convergence"]["p_slope"],
        },
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{interp}")


if __name__ == "__main__":
    main()
