#!/usr/bin/env python3
"""
A004: Attractor Reconstruction — Is Albedo on a Low-Dimensional Attractor?

Tests H002 (Invariant Properties): If globally-averaged albedo is governed by
a low-dimensional dynamical attractor, this would explain its stability as a
consequence of the system's phase space geometry rather than active compensation.

Methods:
1. Takens time-delay embedding of global albedo time series
2. Estimate embedding dimension via false nearest neighbors (FNN)
3. Compute correlation dimension (Grassberger-Procaccia)
4. Estimate largest Lyapunov exponent (stability measure)
5. Recurrence quantification analysis (RQA) for regime detection

If correlation dimension is low (< 5), albedo is dynamically constrained.
If Lyapunov exponent is negative, the system is stable (perturbations decay).
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, spatial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ── Takens Embedding ─────────────────────────────────────────────────

def time_delay_embedding(x, dim, tau):
    """Create time-delay embedding matrix."""
    n = len(x) - (dim - 1) * tau
    if n <= 0:
        raise ValueError(f"Series too short for dim={dim}, tau={tau}")
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]
    return embedded


def optimal_tau(x, method="mutual_info"):
    """Estimate optimal time delay via first minimum of mutual information
    or first zero crossing of autocorrelation."""
    if method == "autocorrelation":
        # First zero crossing of autocorrelation
        acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        for i in range(1, len(acf)):
            if acf[i] < 0:
                return i
        return len(acf) // 4

    # Mutual information via histogram method
    max_lag = min(len(x) // 4, 50)
    mi = np.zeros(max_lag)
    n_bins = max(10, int(np.sqrt(len(x) / 5)))

    for lag in range(max_lag):
        if lag == 0:
            x1, x2 = x, x
        else:
            x1 = x[:-lag]
            x2 = x[lag:]

        hist2d, _, _ = np.histogram2d(x1, x2, bins=n_bins)
        pxy = hist2d / hist2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        mi_val = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi_val += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        mi[lag] = mi_val

    # First minimum
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i
    return np.argmin(mi[1:]) + 1


def false_nearest_neighbors(x, tau, max_dim=10, rtol=15.0):
    """Estimate embedding dimension using false nearest neighbors."""
    fnn_fractions = []

    for dim in range(1, max_dim + 1):
        emb = time_delay_embedding(x, dim, tau)
        n = len(emb)

        if dim < max_dim:
            emb_next = time_delay_embedding(x, dim + 1, tau)
            n_next = len(emb_next)
            n = min(n, n_next)

        # Find nearest neighbors
        tree = spatial.KDTree(emb[:n])
        _, indices = tree.query(emb[:n], k=2)  # k=2: self + nearest

        n_false = 0
        n_total = 0

        for i in range(n):
            nn_idx = indices[i, 1]  # Nearest neighbor (not self)
            d_current = np.linalg.norm(emb[i] - emb[nn_idx])

            if d_current < 1e-10:
                continue

            if dim < max_dim and i < n_next and nn_idx < n_next:
                # Distance in next dimension
                d_next = np.abs(x[i + dim * tau] - x[nn_idx + dim * tau])
                ratio = d_next / d_current

                if ratio > rtol:
                    n_false += 1
                n_total += 1

        fnn_frac = n_false / max(n_total, 1)
        fnn_fractions.append(fnn_frac)

        if fnn_frac < 0.01:  # Less than 1% FNN — sufficient embedding
            break

    return fnn_fractions


def correlation_dimension(x, tau, dim, n_points=2000):
    """Estimate correlation dimension via Grassberger-Procaccia algorithm."""
    emb = time_delay_embedding(x, dim, tau)

    # Subsample if too many points
    if len(emb) > n_points:
        idx = np.random.choice(len(emb), n_points, replace=False)
        emb = emb[idx]

    n = len(emb)

    # Compute pairwise distances
    dists = spatial.distance.pdist(emb)

    # Correlation integral C(r) for range of r values
    r_values = np.logspace(np.log10(np.percentile(dists, 1)),
                            np.log10(np.percentile(dists, 99)), 50)
    C_r = np.array([np.mean(dists < r) for r in r_values])

    # Remove zeros
    mask = C_r > 0
    if mask.sum() < 10:
        return np.nan, r_values, C_r

    log_r = np.log(r_values[mask])
    log_C = np.log(C_r[mask])

    # Linear fit in the scaling region (middle third)
    n_pts = len(log_r)
    start = n_pts // 4
    end = 3 * n_pts // 4
    if end - start < 5:
        start, end = 0, n_pts

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_r[start:end], log_C[start:end]
    )

    return slope, r_values, C_r


def largest_lyapunov_exponent(x, tau, dim, dt=1.0, max_iter=None):
    """Estimate largest Lyapunov exponent using Rosenstein's method."""
    emb = time_delay_embedding(x, dim, tau)
    n = len(emb)
    if max_iter is None:
        max_iter = n // 4

    # Find nearest neighbors (excluding temporal neighbors)
    tree = spatial.KDTree(emb)
    min_temporal_sep = dim * tau + 1

    divergence = np.zeros(max_iter)
    count = np.zeros(max_iter)

    for i in range(n):
        # Find nearest neighbor with temporal separation
        dists, idxs = tree.query(emb[i], k=20)
        nn_idx = None
        for j in range(1, len(idxs)):
            if abs(idxs[j] - i) >= min_temporal_sep:
                nn_idx = idxs[j]
                break

        if nn_idx is None:
            continue

        # Track divergence
        for k in range(max_iter):
            if i + k >= n or nn_idx + k >= n:
                break
            d = np.linalg.norm(emb[i + k] - emb[nn_idx + k])
            if d > 0:
                divergence[k] += np.log(d)
                count[k] += 1

    # Average divergence
    mask = count > 0
    divergence[mask] /= count[mask]

    # Fit slope to get Lyapunov exponent
    time_axis = np.arange(max_iter) * dt
    valid = mask & (divergence > 0)
    if valid.sum() < 5:
        return np.nan, time_axis, divergence

    # Use first quarter for linear fit
    n_valid = valid.sum()
    fit_end = max(5, n_valid // 4)

    slope, _, r_value, p_value, _ = stats.linregress(
        time_axis[valid][:fit_end], divergence[valid][:fit_end]
    )

    return slope, time_axis, divergence


# ── Recurrence Quantification Analysis ───────────────────────────────

def recurrence_plot(emb, threshold):
    """Compute recurrence matrix."""
    dists = spatial.distance.squareform(spatial.distance.pdist(emb))
    return (dists < threshold).astype(int)


def rqa_metrics(rec_matrix):
    """Compute RQA metrics from recurrence matrix."""
    n = rec_matrix.shape[0]
    rr = rec_matrix.sum() / (n * n)  # Recurrence rate

    # Determinism (fraction of recurrence points forming diagonal lines)
    diag_lengths = []
    for k in range(-n+1, n):
        diag = np.diag(rec_matrix, k)
        in_line = False
        line_len = 0
        for val in diag:
            if val:
                in_line = True
                line_len += 1
            else:
                if in_line and line_len >= 2:
                    diag_lengths.append(line_len)
                in_line = False
                line_len = 0
        if in_line and line_len >= 2:
            diag_lengths.append(line_len)

    det = sum(diag_lengths) / max(rec_matrix.sum(), 1) if diag_lengths else 0

    return {"recurrence_rate": rr, "determinism": det,
            "mean_diagonal_length": np.mean(diag_lengths) if diag_lengths else 0,
            "max_diagonal_length": max(diag_lengths) if diag_lengths else 0}


# ── Main Analysis ────────────────────────────────────────────────────

def main():
    print("A004: Attractor Reconstruction — Dynamical Systems Analysis of Albedo")
    print("=" * 60)

    ts = compute_monthly_hemispheric_timeseries()
    print(f"Time series: {len(ts)} months")

    # Use global mean albedo (deseasonalized)
    global_seasonal = ts.groupby("month")["global"].transform("mean")
    global_anom = (ts["global"] - global_seasonal).values
    global_anom = (global_anom - global_anom.mean()) / global_anom.std()

    # Also analyze NH and SH separately
    nh_seasonal = ts.groupby("month")["NH"].transform("mean")
    sh_seasonal = ts.groupby("month")["SH"].transform("mean")
    nh_anom = ((ts["NH"] - nh_seasonal) / ts["NH"].std()).values
    sh_anom = ((ts["SH"] - sh_seasonal) / ts["SH"].std()).values

    results = {}

    for name, series in [("Global", global_anom), ("NH", nh_anom), ("SH", sh_anom)]:
        print(f"\n{'='*40}")
        print(f"Analyzing: {name} albedo anomaly ({len(series)} points)")
        print(f"{'='*40}")

        # 1. Optimal time delay
        tau = optimal_tau(series, method="mutual_info")
        print(f"  Optimal tau (MI): {tau} months")

        # 2. False nearest neighbors → embedding dimension
        print(f"  Computing FNN...")
        fnn = false_nearest_neighbors(series, tau, max_dim=8)
        # Embedding dimension: where FNN drops below 5%
        embed_dim = 1
        for i, f in enumerate(fnn):
            if f < 0.05:
                embed_dim = i + 1
                break
            embed_dim = i + 2
        embed_dim = min(embed_dim, 8)
        print(f"  Embedding dimension (FNN < 5%): {embed_dim}")
        print(f"  FNN fractions: {[f'{f:.3f}' for f in fnn]}")

        # 3. Correlation dimension
        print(f"  Computing correlation dimension (dim={embed_dim})...")
        corr_dim, r_vals, C_r = correlation_dimension(series, tau, embed_dim)
        print(f"  Correlation dimension: {corr_dim:.2f}")

        # 4. Largest Lyapunov exponent
        print(f"  Computing Lyapunov exponent...")
        lyap, lyap_time, lyap_div = largest_lyapunov_exponent(series, tau, embed_dim)
        print(f"  Largest Lyapunov exponent: {lyap:.4f}")
        if lyap < 0:
            print(f"    → STABLE system (perturbations decay)")
        elif lyap > 0:
            print(f"    → CHAOTIC system (perturbations grow)")
        else:
            print(f"    → MARGINAL stability")

        # 5. RQA
        emb = time_delay_embedding(series, embed_dim, tau)
        threshold = np.percentile(spatial.distance.pdist(emb), 10)
        rec = recurrence_plot(emb, threshold)
        rqa = rqa_metrics(rec)
        print(f"  RQA: recurrence_rate={rqa['recurrence_rate']:.3f}, "
              f"determinism={rqa['determinism']:.3f}")

        results[name] = {
            "tau": int(tau),
            "embedding_dimension": int(embed_dim),
            "fnn_fractions": [float(f) for f in fnn],
            "correlation_dimension": float(corr_dim) if not np.isnan(corr_dim) else None,
            "lyapunov_exponent": float(lyap) if not np.isnan(lyap) else None,
            "lyapunov_stable": bool(lyap < 0) if not np.isnan(lyap) else None,
            "rqa": {k: float(v) for k, v in rqa.items()},
        }

    # ── Plotting ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Panel 1: FNN for all three series
    ax = axes[0, 0]
    for name in ["Global", "NH", "SH"]:
        fnn = results[name]["fnn_fractions"]
        ax.plot(range(1, len(fnn)+1), fnn, 'o-', label=name)
    ax.axhline(0.05, color='gray', linestyle='--', label='5% threshold')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('FNN Fraction')
    ax.set_title('False Nearest Neighbors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Correlation dimension scaling
    ax = axes[0, 1]
    for name in ["Global", "NH", "SH"]:
        cd = results[name]["correlation_dimension"]
        ed = results[name]["embedding_dimension"]
        ax.bar(name, cd if cd else 0, alpha=0.7)
        ax.text(name, (cd if cd else 0) + 0.1, f'd={cd:.1f}' if cd else 'N/A',
                ha='center', fontsize=10)
    ax.set_ylabel('Correlation Dimension')
    ax.set_title('Correlation Dimension')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Lyapunov exponents
    ax = axes[0, 2]
    names = ["Global", "NH", "SH"]
    lyaps = [results[n]["lyapunov_exponent"] or 0 for n in names]
    colors = ['green' if l < 0 else 'red' for l in lyaps]
    ax.bar(names, lyaps, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Largest Lyapunov Exponent')
    ax.set_title('Dynamical Stability\n(green = stable, red = chaotic)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: 2D phase portrait (global)
    ax = axes[1, 0]
    tau_g = results["Global"]["tau"]
    n_plot = len(global_anom) - tau_g
    ax.plot(global_anom[:n_plot], global_anom[tau_g:tau_g+n_plot],
            'b-', linewidth=0.5, alpha=0.7)
    ax.scatter(global_anom[:n_plot], global_anom[tau_g:tau_g+n_plot],
               c=np.arange(n_plot), cmap='viridis', s=5, zorder=5)
    ax.set_xlabel(f'Albedo(t)')
    ax.set_ylabel(f'Albedo(t+{tau_g})')
    ax.set_title(f'Phase Portrait (Global, τ={tau_g}mo)')
    ax.grid(True, alpha=0.3)

    # Panel 5: Recurrence plot (global)
    ax = axes[1, 1]
    dim_g = results["Global"]["embedding_dimension"]
    emb_g = time_delay_embedding(global_anom, dim_g, tau_g)
    threshold_g = np.percentile(spatial.distance.pdist(emb_g), 10)
    rec_g = recurrence_plot(emb_g, threshold_g)
    ax.imshow(rec_g, cmap='binary', origin='lower', aspect='auto')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Time (months)')
    ax.set_title(f'Recurrence Plot (Global, dim={dim_g})')

    # Panel 6: RQA comparison
    ax = axes[1, 2]
    det_vals = [results[n]["rqa"]["determinism"] for n in names]
    rr_vals = [results[n]["rqa"]["recurrence_rate"] for n in names]
    x_pos = np.arange(len(names))
    ax.bar(x_pos - 0.15, det_vals, 0.3, label='Determinism', alpha=0.7)
    ax.bar(x_pos + 0.15, rr_vals, 0.3, label='Recurrence Rate', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_ylabel('Fraction')
    ax.set_title('Recurrence Quantification')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = FIG_DIR / "attractor_reconstruction.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    global_cd = results["Global"]["correlation_dimension"]
    global_lyap = results["Global"]["lyapunov_exponent"]
    global_det = results["Global"]["rqa"]["determinism"]

    interpretation_parts = []

    if global_cd is not None and global_cd < 5:
        interpretation_parts.append(
            f"Low correlation dimension ({global_cd:.1f}) suggests albedo is governed by "
            f"a low-dimensional dynamical system — consistent with H002 (invariant constraints)."
        )
    elif global_cd is not None:
        interpretation_parts.append(
            f"Correlation dimension ({global_cd:.1f}) is moderate-to-high, suggesting complex "
            f"dynamics not easily reducible to a simple attractor."
        )

    if global_lyap is not None and global_lyap < 0:
        interpretation_parts.append(
            f"Negative Lyapunov exponent ({global_lyap:.4f}) indicates the system is "
            f"DYNAMICALLY STABLE — perturbations decay over time. This directly supports "
            f"H002: stability is an intrinsic property of the dynamical system."
        )
    elif global_lyap is not None and global_lyap > 0:
        interpretation_parts.append(
            f"Positive Lyapunov exponent ({global_lyap:.4f}) indicates chaotic dynamics. "
            f"Stability must arise from external constraints, not intrinsic dynamics."
        )

    if global_det > 0.5:
        interpretation_parts.append(
            f"High determinism in RQA ({global_det:.2f}) indicates albedo evolution "
            f"follows deterministic trajectories, not random fluctuations."
        )

    interpretation = " ".join(interpretation_parts)
    hypothesis_support = "supports" if (global_lyap and global_lyap < 0) else "inconclusive"

    cd_str = f"{global_cd:.1f}" if global_cd else "N/A"
    lyap_str = f"{global_lyap:.4f}" if global_lyap and not np.isnan(global_lyap) else "N/A"

    output = {
        "analysis": "A004_attractor_reconstruction",
        "method": "attractor_reconstruction",
        "hypothesis": "H002_invariant_properties",
        "data_source": "CERES_EBAF_Ed4.2.1",
        "time_range": f"{ts.date.min().date()} to {ts.date.max().date()}",
        "n_months": len(ts),
        "results": results,
        "summary": (
            f"Dynamical systems analysis of global albedo: "
            f"correlation dimension={cd_str}, "
            f"Lyapunov exponent={lyap_str}, "
            f"RQA determinism={global_det:.2f}."
        ),
        "interpretation": interpretation,
        "hypothesis_support": hypothesis_support,
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved: {OUT_DIR / 'output.json'}")
    print(f"\nInterpretation: {interpretation}")


if __name__ == "__main__":
    main()
