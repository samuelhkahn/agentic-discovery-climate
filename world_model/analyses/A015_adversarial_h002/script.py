#!/usr/bin/env python3
"""
A015: ADVERSARIAL — Challenge H002 (Invariant Properties)

Attacks:
1. HMM "2 states" triviality: state duration = entire series = meaningless
2. Bayesian poly2 spatial confound: does poly(lat,lon) fit ANY field, not just albedo?
3. Attractor dimension sensitivity: is 1.86 robust to series length?
4. Is "nonlinear invariant" just "spatial structure"?
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, spatial
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"


def time_delay_embedding(x, dim, tau):
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau: i * tau + n]
    return embedded


def correlation_dimension(x, tau, dim, n_points=1500):
    emb = time_delay_embedding(x, dim, tau)
    if len(emb) > n_points:
        idx = np.random.choice(len(emb), n_points, replace=False)
        emb = emb[idx]
    dists = spatial.distance.pdist(emb)
    r_values = np.logspace(np.log10(np.percentile(dists, 1)),
                            np.log10(np.percentile(dists, 99)), 50)
    C_r = np.array([np.mean(dists < r) for r in r_values])
    mask = C_r > 0
    if mask.sum() < 10:
        return np.nan
    log_r = np.log(r_values[mask])
    log_C = np.log(C_r[mask])
    n_pts = len(log_r)
    start, end = n_pts // 4, 3 * n_pts // 4
    slope, _, _, _, _ = stats.linregress(log_r[start:end], log_C[start:end])
    return slope


def main():
    print("A015: ADVERSARIAL — Challenging H002")
    print("=" * 60)

    attacks = {}

    # ════════════════════════════════════════════════════════════
    # ATTACK 1: HMM Triviality
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 1: HMM State Duration Triviality ---")
    print("The HMM found 2 states with mean duration = 301 months (entire series).")
    print("This means the HMM found essentially ONE state — no regime switching.")
    print("VERDICT: HMM result is TRIVIAL and should NOT count as supporting evidence.")
    print("The 2-state BIC win is because Gaussian mixture > single Gaussian for any")
    print("non-perfectly-normal distribution, not because of regime structure.")

    attacks["hmm_triviality"] = {
        "target": "A014 HMM",
        "attack": "State duration equals entire series — no actual regime switching detected",
        "verdict": "INVALIDATED",
        "impact": "H002 loses 1 confirming method (5→4)",
        "severity": "HIGH"
    }

    # ════════════════════════════════════════════════════════════
    # ATTACK 2: Bayesian Poly2 Spatial Confound
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 2: Does poly(lat,lon) fit ANY spatial field? ---")

    df = load_ceres_data(terminator_filter=True)
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["time"]).dt.month

    # Subsample
    sample = df.sample(min(200000, len(df)), random_state=42)

    # Test: fit poly2(lat,lon,month) to RANDOM spatial noise
    np.random.seed(42)
    n_random_tests = 20
    random_r2s = []

    X_inv = sample[["lat", "lon", "month"]].values
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_inv)

    for i in range(n_random_tests):
        # Generate spatially-structured random field (not purely random)
        y_random = np.sin(sample["lat"].values * np.pi / 180) * np.random.randn() + \
                   np.cos(sample["lon"].values * np.pi / 180) * np.random.randn() + \
                   np.random.randn(len(sample)) * 0.5
        lr = LinearRegression().fit(X_poly, y_random)
        r2_random = r2_score(y_random, lr.predict(X_poly))
        random_r2s.append(r2_random)

    # Also test on pure white noise
    pure_noise_r2s = []
    for i in range(n_random_tests):
        y_noise = np.random.randn(len(sample))
        lr = LinearRegression().fit(X_poly, y_noise)
        r2_noise = r2_score(y_noise, lr.predict(X_poly))
        pure_noise_r2s.append(r2_noise)

    # Compare: albedo R² vs random spatial field R²
    y_albedo = sample["toa_alb_all_mon"].values
    mask = ~np.isnan(y_albedo)
    lr_alb = LinearRegression().fit(X_poly[mask], y_albedo[mask])
    r2_albedo = r2_score(y_albedo[mask], lr_alb.predict(X_poly[mask]))

    print(f"  Poly2(lat,lon,month) R² on ALBEDO:         {r2_albedo:.4f}")
    print(f"  Poly2(lat,lon,month) R² on RANDOM spatial: {np.mean(random_r2s):.4f} ± {np.std(random_r2s):.4f}")
    print(f"  Poly2(lat,lon,month) R² on PURE NOISE:     {np.mean(pure_noise_r2s):.6f} ± {np.std(pure_noise_r2s):.6f}")

    # Is albedo's R² significantly above random spatial fields?
    t_stat, p_val = stats.ttest_1samp(random_r2s, r2_albedo)
    albedo_above_random = r2_albedo > np.mean(random_r2s) + 2 * np.std(random_r2s)

    if albedo_above_random:
        print(f"  ALBEDO R² is significantly ABOVE random spatial fields → poly2 captures real physics")
        spatial_verdict = "SURVIVES — albedo's spatial structure is non-trivial"
    else:
        print(f"  ALBEDO R² is NOT above random spatial fields → poly2 just fits any spatial field")
        spatial_verdict = "INVALIDATED — poly2 fits any spatial structure, not physics"

    attacks["poly2_spatial_confound"] = {
        "target": "A013 Bayesian model comparison",
        "attack": "Does poly2(lat,lon,month) fit ANY spatial field?",
        "r2_albedo": float(r2_albedo),
        "r2_random_spatial_mean": float(np.mean(random_r2s)),
        "r2_random_spatial_std": float(np.std(random_r2s)),
        "r2_pure_noise_mean": float(np.mean(pure_noise_r2s)),
        "albedo_above_random": bool(albedo_above_random),
        "verdict": spatial_verdict,
        "severity": "HIGH" if not albedo_above_random else "LOW"
    }

    # ════════════════════════════════════════════════════════════
    # ATTACK 3: Attractor Dimension Sensitivity to Series Length
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 3: Correlation Dimension vs Series Length ---")

    ts = compute_monthly_hemispheric_timeseries()
    global_seasonal = ts.groupby("month")["global"].transform("mean")
    global_anom = ((ts["global"] - global_seasonal) / ts["global"].std()).values

    tau = 3
    dim = 3

    # Compute correlation dimension on progressively shorter series
    lengths = [301, 250, 200, 150, 100]
    cd_vs_length = []
    for L in lengths:
        if L > len(global_anom):
            continue
        series = global_anom[:L]
        cd = correlation_dimension(series, tau, dim)
        cd_vs_length.append({"length": L, "corr_dim": float(cd)})
        print(f"  Length={L}: correlation dimension = {cd:.2f}")

    # Also compute on surrogates (shuffled data destroys deterministic structure)
    n_surrogates = 20
    surrogate_cds = []
    for i in range(n_surrogates):
        surr = np.random.permutation(global_anom)
        cd_surr = correlation_dimension(surr, tau, dim)
        if not np.isnan(cd_surr):
            surrogate_cds.append(cd_surr)

    print(f"  Surrogates (shuffled): mean CD = {np.mean(surrogate_cds):.2f} ± {np.std(surrogate_cds):.2f}")
    print(f"  Original CD = {cd_vs_length[0]['corr_dim']:.2f}")

    cd_below_surrogates = cd_vs_length[0]["corr_dim"] < np.mean(surrogate_cds) - np.std(surrogate_cds)

    if cd_below_surrogates:
        attractor_verdict = "SURVIVES — CD is below shuffled surrogates, indicating real low-D structure"
    else:
        attractor_verdict = "WEAKENED — CD is not clearly below surrogates"

    attacks["attractor_sensitivity"] = {
        "target": "A004 Attractor reconstruction",
        "attack": "Is correlation dimension sensitive to series length? Does it differ from surrogates?",
        "cd_vs_length": cd_vs_length,
        "surrogate_cd_mean": float(np.mean(surrogate_cds)),
        "surrogate_cd_std": float(np.std(surrogate_cds)),
        "cd_below_surrogates": bool(cd_below_surrogates),
        "verdict": attractor_verdict,
        "severity": "MEDIUM" if not cd_below_surrogates else "LOW"
    }

    # ════════════════════════════════════════════════════════════
    # ATTACK 4: Is "nonlinear invariant" = "spatial structure"?
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 4: 'Invariant properties' vs 'spatial structure' ---")
    print("  The poly2 model fits lat²/lon² — this captures the SHAPE of Earth's albedo")
    print("  field (bright poles, dark tropics), not physics per se.")
    print("  Counter: the SHAPE is determined by physics (insolation geometry, ice caps).")
    print("  Resolution: H002 is about whether the system is CONSTRAINED by geometry.")
    print("  The poly2 result shows geometry explains 62% — this IS the invariant constraint.")
    print("  The fact that it's 'spatial structure' doesn't invalidate it — Earth's geometry IS invariant.")

    attacks["invariant_vs_spatial"] = {
        "target": "H002 conceptual framing",
        "attack": "Is 'invariant properties' just 'spatial structure'?",
        "verdict": "CONCEPTUAL — spatial structure IS the invariant constraint. Not invalidated but needs careful framing.",
        "severity": "LOW"
    }

    # ── Plotting ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Random vs albedo R²
    ax = axes[0, 0]
    ax.hist(random_r2s, bins=10, alpha=0.6, color='gray', label='Random spatial fields')
    ax.hist(pure_noise_r2s, bins=10, alpha=0.6, color='lightblue', label='Pure noise')
    ax.axvline(r2_albedo, color='red', linewidth=2, label=f'Albedo R²={r2_albedo:.3f}')
    ax.set_xlabel('R² of poly2(lat,lon,month)')
    ax.set_ylabel('Count')
    ax.set_title('Attack 2: Poly2 Spatial Confound')
    ax.legend(fontsize=8)

    # Panel 2: CD vs length
    ax = axes[0, 1]
    if cd_vs_length:
        ls = [c["length"] for c in cd_vs_length]
        cds = [c["corr_dim"] for c in cd_vs_length]
        ax.plot(ls, cds, 'bo-', label='Original data')
        ax.axhline(np.mean(surrogate_cds), color='red', linestyle='--',
                   label=f'Surrogates (mean={np.mean(surrogate_cds):.2f})')
        ax.fill_between(ls, np.mean(surrogate_cds)-np.std(surrogate_cds),
                        np.mean(surrogate_cds)+np.std(surrogate_cds), alpha=0.2, color='red')
    ax.set_xlabel('Series Length (months)')
    ax.set_ylabel('Correlation Dimension')
    ax.set_title('Attack 3: CD Sensitivity to Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Adversarial scorecard
    ax = axes[1, 0]
    attack_names = list(attacks.keys())
    severities = [attacks[a]["severity"] for a in attack_names]
    colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    y_pos = range(len(attack_names))
    ax.barh(y_pos, [{"HIGH": 3, "MEDIUM": 2, "LOW": 1}[s] for s in severities],
            color=[colors[s] for s in severities], alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([a[:25] for a in attack_names], fontsize=8)
    ax.set_xlabel('Severity')
    ax.set_title('Adversarial Attack Severity')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['LOW', 'MEDIUM', 'HIGH'])

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "ADVERSARIAL RESULTS — H002\n\n"
    for name, result in attacks.items():
        summary_text += f"• {name[:30]}:\n  {result['verdict'][:60]}\n  Severity: {result['severity']}\n\n"
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    fig_path = FIG_DIR / "adversarial_h002.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Impact Assessment ──
    high_severity = sum(1 for a in attacks.values() if a["severity"] == "HIGH")
    methods_invalidated = sum(1 for a in attacks.values() if "INVALIDATED" in a.get("verdict", ""))

    if methods_invalidated >= 2:
        impact = f"H002 DOWNGRADED: {methods_invalidated} methods invalidated. Methods confirming reduced from 5 to {5-methods_invalidated}."
        h002_still_converged = (5 - methods_invalidated) >= 5
    else:
        impact = f"H002 survives adversarial testing with {methods_invalidated} method(s) invalidated."
        h002_still_converged = (5 - methods_invalidated) >= 5

    output = {
        "analysis": "A015_adversarial_h002", "method": "adversarial_testing",
        "hypothesis": "H002_invariant_properties",
        "attacks": attacks,
        "methods_invalidated": methods_invalidated,
        "high_severity_attacks": high_severity,
        "h002_still_converged": h002_still_converged,
        "revised_methods_confirming": 5 - methods_invalidated,
        "impact": impact,
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"IMPACT: {impact}")
    print(f"H002 still converged? {h002_still_converged}")

if __name__ == "__main__":
    main()
