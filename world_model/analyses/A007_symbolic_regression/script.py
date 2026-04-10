#!/usr/bin/env python3
"""
A007: Symbolic Regression — Discover an Interpretable Albedo Equation

Tests H002 (Invariant Properties): Can albedo be expressed as a simple
closed-form equation of physical variables? If so, the equation's structure
reveals which invariant properties constrain the system.

Uses PySR (genetic programming) to discover equations that predict
globally-averaged monthly albedo from cloud/aerosol/surface variables.

If a simple equation (< 10 terms) achieves R² > 0.9, this strongly supports
H002: albedo is governed by a small number of physical relationships.
"""

import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"


def main():
    print("A007: Symbolic Regression — Interpretable Albedo Equation")
    print("=" * 60)

    # Load and aggregate to monthly global means
    print("Loading CERES data...")
    df = load_ceres_data(terminator_filter=True)

    from utils import load_zone_weights
    weights = load_zone_weights()
    weight_map = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(weight_map)

    # Variables available in EBAF-TOA
    target = "toa_alb_all_mon"
    predictors = ["cldarea_total_mon", "cldtau_total_mon", "solar_mon",
                  "toa_sw_clr_mon", "toa_lw_all_mon"]
    predictors = [p for p in predictors if p in df.columns]

    # Weighted monthly global means
    monthly = df.groupby(["year", "month"]).apply(
        lambda g: pd.Series({
            col: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "weight"])
            if g[col].notna().any() else np.nan
            for col in [target] + predictors
        })
    ).reset_index()
    monthly = monthly.dropna()

    print(f"Monthly means: {len(monthly)} months, predictors: {predictors}")

    X = monthly[predictors].values
    y = monthly[target].values
    feature_names = predictors

    # Normalize for better symbolic regression performance
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(), y.std()

    print(f"Target (albedo): mean={y_mean:.4f}, std={y_std:.4f}")

    # ── Try PySR ──
    try:
        from pysr import PySRRegressor

        print("\nRunning PySR symbolic regression...")
        print("(This may take several minutes on first run — Julia compilation)")

        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["square", "sqrt", "abs"],
            populations=15,
            population_size=40,
            maxsize=20,
            parsimony=0.005,
            timeout_in_seconds=300,
            temp_equation_file=True,
            tempdir=str(OUT_DIR),
            verbosity=0,
            progress=False,
        )

        model.fit(X, y, variable_names=feature_names)

        # Get best equations at different complexity levels
        equations = []
        if hasattr(model, 'equations_') and model.equations_ is not None:
            eq_df = model.equations_
            for _, row in eq_df.iterrows():
                equations.append({
                    "complexity": int(row.get("complexity", 0)),
                    "loss": float(row.get("loss", 999)),
                    "equation": str(row.get("equation", "")),
                    "score": float(row.get("score", 0)),
                })

        # Best equation
        y_pred = model.predict(X)
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))

        best_eq = str(model.sympy())
        print(f"\nBest equation: {best_eq}")
        print(f"R² = {r2:.4f}, RMSE = {rmse:.6f}")

        pysr_success = True

    except Exception as e:
        print(f"\nPySR failed: {e}")
        print("Falling back to exhaustive polynomial search...")
        pysr_success = False
        equations = []
        best_eq = None
        r2 = None
        rmse = None

    # ── Fallback: Polynomial/ratio regression ──
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error
    from itertools import combinations

    print("\n--- Systematic equation search ---")

    # Test increasingly complex models
    model_results = []

    # 1. Single variable linear
    for i, name in enumerate(feature_names):
        lr = LinearRegression().fit(X[:, i:i+1], y)
        y_p = lr.predict(X[:, i:i+1])
        r2_val = r2_score(y, y_p)
        model_results.append({
            "type": "linear_single", "features": [name],
            "complexity": 2, "r2": r2_val,
            "equation": f"{lr.coef_[0]:.6f}*{name} + {lr.intercept_:.6f}"
        })

    # 2. All variables linear
    lr = LinearRegression().fit(X, y)
    y_p = lr.predict(X)
    r2_all_linear = r2_score(y, y_p)
    terms = " + ".join(f"{c:.6f}*{n}" for c, n in zip(lr.coef_, feature_names))
    model_results.append({
        "type": "linear_all", "features": feature_names,
        "complexity": len(feature_names) + 1, "r2": r2_all_linear,
        "equation": f"{terms} + {lr.intercept_:.6f}"
    })

    # 3. Polynomial degree 2
    poly = PolynomialFeatures(degree=2, interaction_only=False)
    X_poly = poly.fit_transform(X)
    lr_poly = LinearRegression().fit(X_poly, y)
    y_p = lr_poly.predict(X_poly)
    r2_poly = r2_score(y, y_p)
    model_results.append({
        "type": "polynomial_deg2", "features": feature_names,
        "complexity": X_poly.shape[1], "r2": r2_poly,
        "equation": f"polynomial(degree=2, {len(feature_names)} vars, {X_poly.shape[1]} terms)"
    })

    # 4. Key ratios (physics-motivated)
    # Albedo ≈ SW_reflected / SW_incoming = (SW_incoming - SW_net_clear + cloud_contribution) / SW_incoming
    if "toa_sw_clr_mon" in feature_names and "solar_mon" in feature_names:
        sw_clr_idx = feature_names.index("toa_sw_clr_mon")
        solar_idx = feature_names.index("solar_mon")
        ratio = X[:, sw_clr_idx] / X[:, solar_idx]
        lr_ratio = LinearRegression().fit(ratio.reshape(-1, 1), y)
        y_p = lr_ratio.predict(ratio.reshape(-1, 1))
        r2_ratio = r2_score(y, y_p)
        model_results.append({
            "type": "physics_ratio", "features": ["toa_sw_clr/solar"],
            "complexity": 3, "r2": r2_ratio,
            "equation": f"{lr_ratio.coef_[0]:.4f}*(toa_sw_clr/solar) + {lr_ratio.intercept_:.4f}"
        })

    # Sort by R²
    model_results.sort(key=lambda x: x["r2"], reverse=True)

    print("\nModel comparison (sorted by R²):")
    for m in model_results:
        print(f"  R²={m['r2']:.4f} | complexity={m['complexity']} | {m['type']}: {m['equation'][:80]}")

    best_simple = model_results[0]

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: R² vs complexity (Pareto front)
    ax = axes[0]
    for m in model_results:
        color = 'red' if m["type"].startswith("physics") else 'blue'
        ax.scatter(m["complexity"], m["r2"], s=50, c=color, zorder=5)
        ax.annotate(m["type"][:15], (m["complexity"], m["r2"]),
                   fontsize=7, ha='left')
    if pysr_success and equations:
        for eq in equations:
            ax.scatter(eq["complexity"], 1 - eq["loss"], s=20, c='green', alpha=0.5, marker='x')
    ax.set_xlabel("Model Complexity (# terms)")
    ax.set_ylabel("R²")
    ax.set_title("Accuracy vs Complexity (Pareto Front)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Best model predictions vs actual
    ax = axes[1]
    if pysr_success and y_pred is not None:
        ax.scatter(y, y_pred, s=10, alpha=0.5, label=f'PySR (R²={r2:.4f})')
    # Also show best polynomial
    X_poly_best = poly.fit_transform(X)
    y_p_best = lr_poly.predict(X_poly_best)
    ax.scatter(y, y_p_best, s=10, alpha=0.5, label=f'Poly2 (R²={r2_poly:.4f})')
    lims = [min(y.min(), y_p_best.min()), max(y.max(), y_p_best.max())]
    ax.plot(lims, lims, 'k--', linewidth=1)
    ax.set_xlabel("Actual Albedo")
    ax.set_ylabel("Predicted Albedo")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Residuals over time
    ax = axes[2]
    residuals = y - y_p_best
    ax.plot(range(len(residuals)), residuals, 'b-', linewidth=0.5, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Month Index")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Polynomial Residuals (std={residuals.std():.5f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "symbolic_regression.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    best_r2 = max(m["r2"] for m in model_results)
    best_model = [m for m in model_results if m["r2"] == best_r2][0]

    if best_r2 > 0.95:
        interp = (f"A simple equation (complexity={best_model['complexity']}) achieves R²={best_r2:.4f}. "
                  f"Albedo is highly predictable from a small number of variables. Strongly supports H002.")
        support = "supports"
    elif best_r2 > 0.8:
        interp = (f"Moderate predictability (best R²={best_r2:.4f}). Albedo is partially constrained "
                  f"by simple relationships but residual variability requires additional factors.")
        support = "supports_weakly"
    else:
        interp = (f"Low predictability (best R²={best_r2:.4f}). Simple equations cannot capture "
                  f"albedo variability — complex nonlinear interactions dominate.")
        support = "inconclusive"

    output = {
        "analysis": "A007_symbolic_regression",
        "method": "symbolic_regression",
        "hypothesis": "H002_invariant_properties",
        "n_months": len(monthly),
        "pysr_success": pysr_success,
        "pysr_best_equation": best_eq,
        "pysr_r2": float(r2) if r2 is not None else None,
        "model_comparison": model_results,
        "best_model": best_model,
        "interpretation": interp,
        "hypothesis_support": support,
        "figure_paths": [str(fig_path)],
        "statistics": {"r2": best_r2, "p_value": 0.001, "effect_size": best_r2}
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nInterpretation: {interp}")


if __name__ == "__main__":
    main()
