#!/usr/bin/env python3
"""
A024: CCM NH↔SH — Causal Teleconnection Test (H003, H004)

Applies Convergent Cross Mapping to hemispheric albedo anomalies.
If H003 (teleconnections): NH-SH causal coupling should converge.
If H004 (coupled oscillator): SH→NH should converge with asymmetric strength.

Uses the same fixed CCM implementation as A019 (proper LOO, temporal exclusion).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import spatial, stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def time_delay_embedding(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]
    return embedded


def ccm_fixed(source, target, dim=3, tau=1, lib_sizes=None, n_reps=100):
    """Fixed CCM with LOO and temporal exclusion (from A019)."""
    N = len(source)
    target_emb = time_delay_embedding(target, dim, tau)
    source_aligned = source[(dim - 1) * tau:]
    n_emb = len(target_emb)

    if lib_sizes is None:
        lib_sizes = np.unique(np.linspace(dim + 2, n_emb - 5, 12).astype(int))

    excl_window = dim * tau + 1
    results = {"lib_sizes": lib_sizes.tolist(), "rho_mean": [], "rho_std": []}

    for L in lib_sizes:
        rhos = []
        for rep in range(n_reps):
            all_idx = np.arange(n_emb)
            lib_idx = np.sort(np.random.choice(n_emb, min(L, n_emb), replace=False))
            pred_mask = np.ones(n_emb, dtype=bool)
            for li in lib_idx:
                pred_mask[max(0, li - excl_window):min(n_emb, li + excl_window + 1)] = False
            pred_idx = np.where(pred_mask)[0]

            if len(pred_idx) < 5:
                pred_idx = lib_idx
                loo_mode = True
            else:
                loo_mode = False

            lib_emb = target_emb[lib_idx]
            tree = spatial.KDTree(lib_emb)

            source_pred, source_actual = [], []
            for pi in pred_idx:
                query = target_emb[pi]
                k = min(dim + 1, len(lib_idx))
                dists, nn_idx_in_lib = tree.query(query, k=k)
                if isinstance(dists, float):
                    dists = np.array([dists])
                    nn_idx_in_lib = np.array([nn_idx_in_lib])

                if loo_mode:
                    valid = nn_idx_in_lib != np.searchsorted(lib_idx, pi)
                    if valid.sum() < 2:
                        continue
                    dists = dists[valid][:dim + 1]
                    nn_idx_in_lib = nn_idx_in_lib[valid][:dim + 1]

                nn_orig_idx = lib_idx[nn_idx_in_lib]
                temporal_ok = np.abs(nn_orig_idx - pi) > excl_window
                if temporal_ok.sum() < 2:
                    continue
                dists = dists[temporal_ok]
                nn_orig_idx = nn_orig_idx[temporal_ok]
                if len(dists) == 0:
                    continue

                min_dist = max(dists[0], 1e-15)
                weights = np.exp(-dists / min_dist)
                weights /= weights.sum()

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
    lib_sizes = np.array(results["lib_sizes"])
    rho_mean = np.array(results["rho_mean"])
    if len(rho_mean) < 5 or all(r == 0 for r in rho_mean):
        return {"converges": False, "slope": 0, "p_value": 1.0, "rho_max": 0, "rho_min": 0}
    slope, intercept, r, p, se = stats.linregress(lib_sizes, rho_mean)
    return {
        "converges": slope > 0 and p < 0.05,
        "slope": float(slope), "p_slope": float(p),
        "rho_min": float(rho_mean[0]), "rho_max": float(rho_mean[-1]),
        "rho_increase": float(rho_mean[-1] - rho_mean[0]),
    }


def main():
    print("A024: CCM NH↔SH — Teleconnection Causality (H003, H004)")
    print("=" * 60)

    ts = compute_monthly_hemispheric_timeseries()
    print(f"Time series: {len(ts)} months")

    nh_s = ts.groupby("month")["NH"].transform("mean")
    sh_s = ts.groupby("month")["SH"].transform("mean")
    nh = ((ts["NH"] - nh_s) / ts["NH"].std()).values
    sh = ((ts["SH"] - sh_s) / ts["SH"].std()).values

    dim = 3
    tau = 3  # optimal tau from A004
    n_reps = 100

    results = {}

    print("\n--- CCM: SH → NH (Map FROM NH manifold TO predict SH) ---")
    ccm1 = ccm_fixed(sh, nh, dim=dim, tau=tau, n_reps=n_reps)
    conv1 = test_convergence(ccm1)
    print(f"  Converges: {conv1['converges']} (slope={conv1['slope']:.5f}, p={conv1['p_slope']:.4f})")
    print(f"  ρ: {conv1['rho_min']:.3f} → {conv1['rho_max']:.3f}")
    results["SH_causes_NH"] = {"ccm": ccm1, "convergence": conv1}

    print("\n--- CCM: NH → SH (Map FROM SH manifold TO predict NH) ---")
    ccm2 = ccm_fixed(nh, sh, dim=dim, tau=tau, n_reps=n_reps)
    conv2 = test_convergence(ccm2)
    print(f"  Converges: {conv2['converges']} (slope={conv2['slope']:.5f}, p={conv2['p_slope']:.4f})")
    print(f"  ρ: {conv2['rho_min']:.3f} → {conv2['rho_max']:.3f}")
    results["NH_causes_SH"] = {"ccm": ccm2, "convergence": conv2}

    # Also test with global mean to understand its attractor
    gl_s = ts.groupby("month")["global"].transform("mean")
    gl = ((ts["global"] - gl_s) / ts["global"].std()).values

    print("\n--- CCM: NH → Global (Map FROM Global manifold TO predict NH) ---")
    ccm3 = ccm_fixed(nh, gl, dim=dim, tau=tau, n_reps=80)
    conv3 = test_convergence(ccm3)
    print(f"  Converges: {conv3['converges']} (ρ: {conv3['rho_min']:.3f} → {conv3['rho_max']:.3f})")
    results["NH_causes_global"] = {"ccm": ccm3, "convergence": conv3}

    print("\n--- CCM: SH → Global ---")
    ccm4 = ccm_fixed(sh, gl, dim=dim, tau=tau, n_reps=80)
    conv4 = test_convergence(ccm4)
    print(f"  Converges: {conv4['converges']} (ρ: {conv4['rho_min']:.3f} → {conv4['rho_max']:.3f})")
    results["SH_causes_global"] = {"ccm": ccm4, "convergence": conv4}

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, key) in zip(axes, [
        ("SH → NH (does SH cause NH?)", "SH_causes_NH"),
        ("NH → SH (does NH cause SH?)", "NH_causes_SH"),
    ]):
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
    fig_path = FIG_DIR / "ccm_nh_sh.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    sh_c_nh = results["SH_causes_NH"]["convergence"]["converges"]
    nh_c_sh = results["NH_causes_SH"]["convergence"]["converges"]

    if sh_c_nh and nh_c_sh:
        interp = "BIDIRECTIONAL CCM coupling: both SH→NH and NH→SH converge. Supports H003 (teleconnections) and H004 (coupled oscillator — mutual causality)."
        if results["SH_causes_NH"]["convergence"]["rho_max"] > results["NH_causes_SH"]["convergence"]["rho_max"]:
            interp += " SH→NH coupling is stronger, consistent with SH being the dominant driver (TE finding)."
        support = "supports"
    elif sh_c_nh:
        interp = "UNIDIRECTIONAL: SH causes NH but not reverse. Strongly supports H003 and H004's SH-drives-NH prediction."
        support = "supports"
    elif nh_c_sh:
        interp = "UNIDIRECTIONAL: NH causes SH. Supports H003 but reverse direction from H004."
        support = "supports"
    else:
        interp = "No CCM convergence. H003 not supported by manifold causality."
        support = "inconclusive"

    output = {
        "analysis": "A024_ccm_nh_sh",
        "method": "convergent_cross_mapping",
        "hypotheses": ["H003_teleconnections", "H004_coupled_low_dimensional_hemispheric_oscillator"],
        "results": {k: {"convergence": v["convergence"]} for k, v in results.items()},
        "interpretation": interp,
        "hypothesis_support": support,
        "statistics": {
            "SH_causes_NH_converges": sh_c_nh,
            "NH_causes_SH_converges": nh_c_sh,
            "SH_to_NH_rho_max": results["SH_causes_NH"]["convergence"]["rho_max"],
            "NH_to_SH_rho_max": results["NH_causes_SH"]["convergence"]["rho_max"],
            "p_SH_to_NH": results["SH_causes_NH"]["convergence"]["p_slope"],
            "p_NH_to_SH": results["NH_causes_SH"]["convergence"]["p_slope"],
        },
        "figure_paths": [str(fig_path)],
    }
    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")


if __name__ == "__main__":
    main()
