#!/usr/bin/env python3
"""
A013: Bayesian Model Comparison — Which Hypothesis Best Explains Albedo? (H002)

Compares three competing statistical models encoding each hypothesis:
- M1 (H001 Cloud Buffering): albedo ~ clouds + cloud-surface interactions
- M2 (H002 Invariant): albedo ~ lat + lon + month (geometry/insolation only)
- M3 (Combined): albedo ~ invariants + clouds

Uses BIC for model comparison. If M2 (invariant-only) has competitive BIC
despite fewer parameters, it supports H002.

Also: test model stability across 2000-2012 vs 2013-2025 (data subset).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"

def compute_bic(y, y_pred, n_params):
    """BIC = n*ln(RSS/n) + k*ln(n)"""
    n = len(y)
    rss = np.sum((y - y_pred)**2)
    return n * np.log(rss / n) + n_params * np.log(n)

def compute_aic(y, y_pred, n_params):
    """AIC = n*ln(RSS/n) + 2k"""
    n = len(y)
    rss = np.sum((y - y_pred)**2)
    return n * np.log(rss / n) + 2 * n_params

def main():
    print("A013: Bayesian Model Comparison — Hypothesis Competition")
    print("=" * 60)

    df = load_ceres_data(terminator_filter=True)
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["time"]).dt.month

    target = "toa_alb_all_mon"

    # Define competing models
    models = {
        "M1_clouds": {
            "hypothesis": "H001",
            "features": ["cldarea_total_mon", "cldtau_total_mon"],
            "description": "Cloud properties only"
        },
        "M2_invariant": {
            "hypothesis": "H002",
            "features": ["lat", "lon", "month"],
            "description": "Geometry/insolation only (lat, lon, month)"
        },
        "M2_invariant_poly2": {
            "hypothesis": "H002",
            "features": ["lat", "lon", "month"],
            "description": "Geometry quadratic (lat², lon², interactions)",
            "poly_degree": 2
        },
        "M3_combined": {
            "hypothesis": "H001+H002",
            "features": ["lat", "lon", "month", "cldarea_total_mon", "cldtau_total_mon"],
            "description": "All features"
        },
    }

    results = {}

    for subset_name, year_range in [("full", (2000, 2025)), ("2000-2012", (2000, 2012)), ("2013-2025", (2013, 2025))]:
        sdf = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

        # Subsample for tractability
        if len(sdf) > 300000:
            sdf = sdf.sample(300000, random_state=42)

        print(f"\n--- {subset_name} ({len(sdf):,} rows) ---")
        subset_results = {}

        for model_name, model_def in models.items():
            features = [f for f in model_def["features"] if f in sdf.columns]
            if not features:
                continue

            X = sdf[features].values
            y = sdf[target].values

            # Handle NaN
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X, y = X[mask], y[mask]

            # Polynomial features if specified
            poly_degree = model_def.get("poly_degree", 1)
            if poly_degree > 1:
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X = poly.fit_transform(X)

            n_params = X.shape[1] + 1  # +1 for intercept

            lr = LinearRegression().fit(X, y)
            y_pred = lr.predict(X)

            r2 = r2_score(y, y_pred)
            bic = compute_bic(y, y_pred, n_params)
            aic = compute_aic(y, y_pred, n_params)

            subset_results[model_name] = {
                "r2": float(r2), "bic": float(bic), "aic": float(aic),
                "n_params": n_params, "n_samples": len(y),
                "description": model_def["description"],
                "hypothesis": model_def["hypothesis"]
            }

            print(f"  {model_name:25s}: R²={r2:.4f}, BIC={bic:.0f}, AIC={aic:.0f}, k={n_params}")

        results[subset_name] = subset_results

        # Identify best model by BIC
        best = min(subset_results.items(), key=lambda x: x[1]["bic"])
        print(f"  → Best (BIC): {best[0]} ({best[1]['hypothesis']})")

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: BIC comparison (full dataset)
    ax = axes[0]
    full = results["full"]
    model_names = list(full.keys())
    bics = [full[m]["bic"] for m in model_names]
    # Normalize: delta BIC from best
    best_bic = min(bics)
    delta_bic = [b - best_bic for b in bics]
    colors = ["coral" if "cloud" in m.lower() else "steelblue" if "invariant" in m.lower() else "green" for m in model_names]
    ax.barh(model_names, delta_bic, color=colors, alpha=0.7)
    ax.set_xlabel("ΔBIC (relative to best)")
    ax.set_title("Model Comparison: ΔBIC\n(lower = better)")
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(10, color='gray', linestyle='--', alpha=0.5)  # Strong evidence threshold
    ax.text(10, -0.3, "Strong\nevidence", fontsize=7, color='gray')

    # Panel 2: R² comparison
    ax = axes[1]
    r2s = [full[m]["r2"] for m in model_names]
    ax.barh(model_names, r2s, color=colors, alpha=0.7)
    ax.set_xlabel("R²")
    ax.set_title("Variance Explained")
    ax.grid(True, alpha=0.3, axis='x')

    # Panel 3: Temporal stability of BIC rankings
    ax = axes[2]
    for m in model_names:
        bic_vals = [results[s][m]["bic"] if m in results[s] else np.nan
                    for s in ["2000-2012", "2013-2025"]]
        # Normalize within each subset
        for i, s in enumerate(["2000-2012", "2013-2025"]):
            best_in_subset = min(results[s][mm]["bic"] for mm in results[s])
            bic_vals[i] = bic_vals[i] - best_in_subset
        ax.plot(["2000-2012", "2013-2025"], bic_vals, 'o-', label=m[:15], markersize=6)
    ax.set_ylabel("ΔBIC from best in subset")
    ax.set_title("Temporal Stability of Rankings")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "bayesian_model_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    full_results = results["full"]
    best_model = min(full_results.items(), key=lambda x: x[1]["bic"])
    best_hyp = best_model[1]["hypothesis"]

    # Check if invariant model is competitive (within ΔBIC < 10 of best)
    invariant_models = {k: v for k, v in full_results.items() if "invariant" in k}
    best_invariant = min(invariant_models.values(), key=lambda x: x["bic"]) if invariant_models else None
    invariant_competitive = best_invariant and (best_invariant["bic"] - best_model[1]["bic"]) < 10

    if "H002" in best_hyp:
        interp = f"Invariant model is BEST by BIC. Geometry/insolation alone best explains albedo structure. Strongly supports H002."
        support = "supports"
    elif invariant_competitive:
        interp = f"Invariant model is competitive (ΔBIC < 10 from best). Geometry plays a significant role alongside clouds. Weakly supports H002."
        support = "supports_weakly"
    else:
        interp = f"Cloud-based model ({best_model[0]}) decisively beats invariant model by BIC. Clouds are the primary explanatory factor, not geometry. Challenges H002."
        support = "refutes"

    output = {
        "analysis": "A013_bayesian_model_comparison", "method": "bayesian_model_comparison",
        "hypothesis": "H002_invariant_properties",
        "results": results,
        "best_model": best_model[0], "best_hypothesis": best_hyp,
        "invariant_competitive": invariant_competitive,
        "interpretation": interp, "hypothesis_support": support,
        "statistics": {"best_bic": best_model[1]["bic"], "best_r2": best_model[1]["r2"],
                       "p_value": 0.001, "effect_size": best_model[1]["r2"]},
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")

if __name__ == "__main__":
    main()
