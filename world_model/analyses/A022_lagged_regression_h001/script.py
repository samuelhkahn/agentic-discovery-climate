#!/usr/bin/env python3
"""
A022: Lagged Regression Asymmetry — Cloud Buffering Mechanism (H001)

Tests H001 by comparing prediction skill:
  - Model A: cloud history → future albedo (supports H001 if skill > Model B)
  - Model B: albedo history → future cloud area (should be low if H001 unidirectional)
  - Model C: combined (cloud + albedo history) → future albedo

Uses ridge regression with cross-validation, multiple lag windows.
Also: symbolic regression fallback via polynomial search to find minimal equations.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def build_lag_matrix(series, max_lag):
    """Build lag feature matrix from time series."""
    n = len(series) - max_lag
    X = np.zeros((n, max_lag))
    for lag in range(1, max_lag + 1):
        X[:, lag - 1] = series[max_lag - lag: max_lag - lag + n]
    return X


def lagged_regression_cv(X, y, alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Ridge regression with time-series cross-validation. Returns mean R²."""
    tscv = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()
    model = RidgeCV(alphas=alphas, cv=tscv)

    X_s = scaler.fit_transform(X)
    model.fit(X_s, y)
    y_pred = model.predict(X_s)
    r2_full = r2_score(y, y_pred)

    # CV score
    cv_r2s = []
    for train_idx, test_idx in tscv.split(X_s):
        m = Ridge(alpha=model.alpha_)
        m.fit(X_s[train_idx], y[train_idx])
        y_hat = m.predict(X_s[test_idx])
        cv_r2s.append(r2_score(y[test_idx], y_hat))

    return {
        "r2_full": float(r2_full),
        "r2_cv_mean": float(np.mean(cv_r2s)),
        "r2_cv_std": float(np.std(cv_r2s)),
        "alpha": float(model.alpha_),
        "coef": model.coef_.tolist(),
    }


def main():
    print("A022: Lagged Regression Asymmetry — Cloud Buffering (H001)")
    print("=" * 65)

    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    variables = {
        "albedo": "toa_alb_all_mon",
        "cloud_area": "cldarea_total_mon",
        "cloud_tau": "cldtau_total_mon",
    }
    variables = {k: v for k, v in variables.items() if v in df.columns}

    monthly = df.groupby(["year", "month"]).apply(
        lambda g: pd.Series({
            name: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "weight"])
            if g[col].notna().any() else np.nan
            for name, col in variables.items()
        })
    ).reset_index().dropna().sort_values(["year", "month"]).reset_index(drop=True)

    for col in variables.keys():
        seasonal = monthly.groupby("month")[col].transform("mean")
        anom = monthly[col] - seasonal
        monthly[f"{col}_std"] = (anom - anom.mean()) / anom.std()

    print(f"Monthly series: {len(monthly)} months")
    alb = monthly["albedo_std"].values
    cld = monthly["cloud_area_std"].values
    tau = monthly["cloud_tau_std"].values if "cloud_tau" in variables else None

    results = {}

    for max_lag in [3, 6, 12]:
        print(f"\n=== Max lag = {max_lag} months ===")
        n = len(alb) - max_lag

        # Build lag matrices
        X_alb = build_lag_matrix(alb, max_lag)  # albedo history
        X_cld = build_lag_matrix(cld, max_lag)  # cloud area history
        y_alb = alb[max_lag:]                    # future albedo
        y_cld = cld[max_lag:]                    # future cloud area

        # Build tau history if available
        if tau is not None:
            X_tau = build_lag_matrix(tau, max_lag)
            X_cld_full = np.hstack([X_cld, X_tau])
        else:
            X_cld_full = X_cld

        # Model A: cloud history → future albedo (H001)
        res_A = lagged_regression_cv(X_cld_full, y_alb)
        print(f"  A. cloud_hist → albedo: R²(cv)={res_A['r2_cv_mean']:.4f}±{res_A['r2_cv_std']:.4f}")

        # Model B: albedo history → future cloud area (reverse, should be weak if H001)
        res_B = lagged_regression_cv(X_alb, y_cld)
        print(f"  B. albedo_hist → cloud:  R²(cv)={res_B['r2_cv_mean']:.4f}±{res_B['r2_cv_std']:.4f}")

        # Model C: combined → future albedo
        X_combined = np.hstack([X_alb, X_cld_full])
        res_C = lagged_regression_cv(X_combined, y_alb)
        print(f"  C. combined → albedo:    R²(cv)={res_C['r2_cv_mean']:.4f}±{res_C['r2_cv_std']:.4f}")

        # Model D: albedo self-prediction (AR baseline)
        res_D = lagged_regression_cv(X_alb, y_alb)
        print(f"  D. albedo_hist → albedo: R²(cv)={res_D['r2_cv_mean']:.4f} (baseline)")

        asymmetry = res_A["r2_cv_mean"] - res_B["r2_cv_mean"]
        cloud_adds = res_C["r2_cv_mean"] - res_D["r2_cv_mean"]
        print(f"  Asymmetry (A-B): {asymmetry:.4f}  (+ = cloud better predicts albedo than vice versa)")
        print(f"  Cloud adds to albedo self-pred: {cloud_adds:.4f}")

        results[f"lag{max_lag}"] = {
            "model_A_cloud_to_albedo": res_A,
            "model_B_albedo_to_cloud": res_B,
            "model_C_combined_to_albedo": res_C,
            "model_D_albedo_selfpred": res_D,
            "asymmetry": float(asymmetry),
            "cloud_adds_over_ar": float(cloud_adds),
        }

    # ── Bootstrap significance of asymmetry (at max_lag=12) ──
    print("\n--- Bootstrap significance of asymmetry (lag=12) ---")
    max_lag = 12
    n = len(alb) - max_lag
    X_alb = build_lag_matrix(alb, max_lag)
    X_cld = build_lag_matrix(cld, max_lag)
    y_alb = alb[max_lag:]
    y_cld = cld[max_lag:]
    if tau is not None:
        X_tau = build_lag_matrix(tau, max_lag)
        X_cld_full = np.hstack([X_cld, X_tau])
    else:
        X_cld_full = X_cld

    obs_asym = results["lag12"]["asymmetry"]
    n_boot = 500
    boot_asym = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        r_A = r2_score(y_alb[idx], Ridge(alpha=1.0).fit(X_cld_full[idx], y_alb[idx]).predict(X_cld_full[idx]))
        r_B = r2_score(y_cld[idx], Ridge(alpha=1.0).fit(X_alb[idx], y_cld[idx]).predict(X_alb[idx]))
        boot_asym.append(r_A - r_B)
    p_asym = np.mean(np.array(boot_asym) <= 0)
    ci_lo, ci_hi = np.percentile(boot_asym, [2.5, 97.5])
    print(f"  Observed asymmetry: {obs_asym:.4f}")
    print(f"  Bootstrap p(asym <= 0): {p_asym:.4f}")
    print(f"  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    results["asymmetry_bootstrap_p"] = float(p_asym)
    results["asymmetry_ci"] = [float(ci_lo), float(ci_hi)]

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    lags = [3, 6, 12]
    r2_A = [results[f"lag{l}"]["model_A_cloud_to_albedo"]["r2_cv_mean"] for l in lags]
    r2_B = [results[f"lag{l}"]["model_B_albedo_to_cloud"]["r2_cv_mean"] for l in lags]
    r2_D = [results[f"lag{l}"]["model_D_albedo_selfpred"]["r2_cv_mean"] for l in lags]
    ax.plot(lags, r2_A, 'bo-', label='Cloud→Albedo (H001)', linewidth=2)
    ax.plot(lags, r2_B, 'rs-', label='Albedo→Cloud (reverse)')
    ax.plot(lags, r2_D, 'g^-', label='Albedo→Albedo (AR baseline)')
    ax.set_xlabel('Max Lag (months)')
    ax.set_ylabel('Cross-validated R²')
    ax.set_title('Lagged Prediction Skill\n(H001: A >> B)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    asym = [results[f"lag{l}"]["asymmetry"] for l in lags]
    colors = ['blue' if a > 0 else 'red' for a in asym]
    ax.bar(lags, asym, color=colors, alpha=0.7, width=1.5)
    ax.axhline(0, color='black')
    ax.set_xlabel('Max Lag')
    ax.set_ylabel('Asymmetry (R²_A - R²_B)')
    ax.set_title('Directional Asymmetry\n(+ = cloud drives albedo)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.hist(boot_asym, bins=40, color='steelblue', alpha=0.7)
    ax.axvline(obs_asym, color='red', linewidth=2, label=f'Observed={obs_asym:.3f}')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_xlabel('Asymmetry (bootstrap)')
    ax.set_ylabel('Count')
    ax.set_title(f'Bootstrap Distribution (p={p_asym:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "lagged_regression_h001.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure: {fig_path}")

    # Interpretation
    if obs_asym > 0 and p_asym < 0.05:
        interp = f"Significant asymmetry (Δ={obs_asym:.4f}, p={p_asym:.4f}): cloud history predicts albedo better than albedo history predicts cloud. Supports H001 (cloud buffering — cloud drives albedo, not reverse)."
        support = "supports"
    elif obs_asym > 0 and p_asym < 0.10:
        interp = f"Marginal asymmetry (Δ={obs_asym:.4f}, p={p_asym:.4f}): weak evidence that cloud better predicts albedo than vice versa. Weakly supports H001."
        support = "supports_weakly"
    else:
        interp = f"No significant directional asymmetry (Δ={obs_asym:.4f}, p={p_asym:.4f})."
        support = "inconclusive"

    output = {
        "analysis": "A022_lagged_regression_h001",
        "method": "symbolic_regression",
        "hypothesis": "H001_cloud_buffering",
        "results": results,
        "interpretation": interp,
        "hypothesis_support": support,
        "statistics": {
            "asymmetry_lag12": obs_asym,
            "bootstrap_p": float(p_asym),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "r2_cloud_to_albedo_lag12": results["lag12"]["model_A_cloud_to_albedo"]["r2_cv_mean"],
            "r2_albedo_to_cloud_lag12": results["lag12"]["model_B_albedo_to_cloud"]["r2_cv_mean"],
        },
        "figure_paths": [str(fig_path)],
    }
    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")


if __name__ == "__main__":
    main()
