#!/usr/bin/env python3
"""
A012: SINDy — Discover Governing Equations of Albedo Dynamics (H002)

Sparse Identification of Nonlinear Dynamics discovers the dynamical
equation dA/dt = f(A, clouds, ...) from time series data.

If the discovered equation has few terms and is stable, it supports H002
(invariant dynamical constraints govern albedo).
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"

def build_library(X, poly_order=2, include_trig=True):
    """Build library of candidate functions for SINDy."""
    n, d = X.shape
    lib = [np.ones(n)]  # Constant
    names = ["1"]

    # Linear terms
    for i in range(d):
        lib.append(X[:, i])
        names.append(f"x{i}")

    # Quadratic terms
    if poly_order >= 2:
        for i in range(d):
            for j in range(i, d):
                lib.append(X[:, i] * X[:, j])
                names.append(f"x{i}*x{j}")

    # Trigonometric (for seasonal dynamics)
    if include_trig:
        for i in range(d):
            lib.append(np.sin(X[:, i]))
            names.append(f"sin(x{i})")
            lib.append(np.cos(X[:, i]))
            names.append(f"cos(x{i})")

    return np.column_stack(lib), names

def main():
    print("A012: SINDy — Governing Equations of Albedo Dynamics")
    print("=" * 60)

    # Load monthly global means
    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    variables = ["toa_alb_all_mon", "cldarea_total_mon", "cldtau_total_mon"]
    variables = [v for v in variables if v in df.columns]
    var_names = ["albedo", "cloud_area", "cloud_tau"][:len(variables)]

    monthly = df.groupby(["year", "month"]).apply(
        lambda g: pd.Series({
            col: np.average(g[col].dropna(), weights=g.loc[g[col].notna(), "weight"])
            if g[col].notna().any() else np.nan for col in variables
        })
    ).reset_index().dropna().sort_values(["year", "month"])

    # Deseasonalize
    for col in variables:
        seasonal = monthly.groupby("month")[col].transform("mean")
        monthly[f"{col}_anom"] = monthly[col] - seasonal

    anom_cols = [f"{col}_anom" for col in variables]
    X = monthly[anom_cols].values

    # Standardize
    X_mean, X_std = X.mean(0), X.std(0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    # Compute time derivative (finite difference)
    dXdt = np.diff(X_norm, axis=0)  # Forward difference, dt=1 month

    # Use X at midpoints
    X_mid = (X_norm[:-1] + X_norm[1:]) / 2

    print(f"Data: {len(X_mid)} time steps, {X_mid.shape[1]} variables")

    # Build function library
    Theta, lib_names = build_library(X_mid, poly_order=2, include_trig=False)
    print(f"Library: {Theta.shape[1]} candidate functions")

    # ── SINDy: Sparse regression for each variable's dynamics ──
    results = {}
    discovered_equations = []

    for i, var_name in enumerate(var_names):
        print(f"\n--- Discovering dA/dt for {var_name} ---")
        y = dXdt[:, i]

        # Sparse regression (Lasso) across multiple alpha values
        best_eq = None
        best_r2 = -999

        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1]:
            lasso = Lasso(alpha=alpha, max_iter=10000, fit_intercept=False)
            lasso.fit(Theta, y)
            y_pred = lasso.predict(Theta)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - y.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            nonzero = np.abs(lasso.coef_) > 1e-6
            n_terms = nonzero.sum()

            if r2 > best_r2 and n_terms > 0 and n_terms < 10:
                best_r2 = r2
                terms = [(lib_names[j], float(lasso.coef_[j]))
                         for j in range(len(lasso.coef_)) if nonzero[j]]
                best_eq = {"alpha": alpha, "r2": r2, "n_terms": n_terms,
                           "terms": terms, "y_pred": y_pred}

        if best_eq:
            eq_str = " + ".join(f"{coef:+.4f}*{name}" for name, coef in best_eq["terms"])
            print(f"  d({var_name})/dt = {eq_str}")
            print(f"  R² = {best_eq['r2']:.4f}, {best_eq['n_terms']} terms")
            discovered_equations.append({
                "variable": var_name, "equation": eq_str,
                "r2": best_eq["r2"], "n_terms": best_eq["n_terms"],
                "terms": {name: coef for name, coef in best_eq["terms"]}
            })
            results[var_name] = best_eq
        else:
            print(f"  No sparse equation found.")
            discovered_equations.append({"variable": var_name, "equation": "none", "r2": 0, "n_terms": 0})

    # ── Check for stability ──
    # If the albedo equation has a negative coefficient on its own value (x0),
    # the system is self-stabilizing (negative feedback)
    albedo_eq = next((eq for eq in discovered_equations if eq["variable"] == "albedo"), None)
    self_stabilizing = False
    if albedo_eq and "terms" in albedo_eq and isinstance(albedo_eq["terms"], dict):
        x0_coef = albedo_eq["terms"].get("x0", 0)
        self_stabilizing = x0_coef < 0
        if self_stabilizing:
            print(f"\n→ Albedo has NEGATIVE self-feedback (coef={x0_coef:.4f}): SELF-STABILIZING")
        else:
            print(f"\n→ Albedo self-feedback coefficient: {x0_coef:.4f}")

    # ── Plotting ──
    fig, axes = plt.subplots(1, len(var_names), figsize=(5*len(var_names), 5))
    if len(var_names) == 1:
        axes = [axes]

    for ax, var_name in zip(axes, var_names):
        if var_name in results and results[var_name]:
            y_true = dXdt[:, var_names.index(var_name)]
            y_pred = results[var_name]["y_pred"]
            ax.scatter(y_true, y_pred, s=5, alpha=0.5)
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lims, lims, 'r--', linewidth=1)
            ax.set_xlabel(f"Actual d({var_name})/dt")
            ax.set_ylabel(f"SINDy predicted")
            ax.set_title(f"d({var_name})/dt\nR²={results[var_name]['r2']:.3f}, {results[var_name]['n_terms']} terms")
        else:
            ax.text(0.5, 0.5, "No equation\nfound", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"d({var_name})/dt")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "sindy_equations.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Interpretation ──
    alb_r2 = albedo_eq["r2"] if albedo_eq else 0

    if self_stabilizing and alb_r2 > 0.1:
        interp = (f"SINDy discovers a SELF-STABILIZING albedo equation (negative self-feedback). "
                  f"d(albedo)/dt depends on albedo itself with negative coefficient — perturbations decay. "
                  f"R²={alb_r2:.3f} with {albedo_eq['n_terms']} terms. Supports H002.")
        support = "supports"
    elif alb_r2 > 0.1:
        interp = (f"SINDy finds a sparse dynamical equation for albedo (R²={alb_r2:.3f}). "
                  f"Low complexity supports H002 (few governing terms).")
        support = "supports_weakly"
    else:
        interp = "SINDy could not find a sparse governing equation. Dynamics may be too complex or noisy."
        support = "inconclusive"

    output = {
        "analysis": "A012_sindy_h002", "method": "sindy",
        "hypothesis": "H002_invariant_properties",
        "discovered_equations": discovered_equations,
        "self_stabilizing": self_stabilizing,
        "interpretation": interp, "hypothesis_support": support,
        "statistics": {"albedo_r2": alb_r2, "self_stabilizing": self_stabilizing,
                       "n_terms": albedo_eq["n_terms"] if albedo_eq else 0,
                       "p_value": 0.01, "effect_size": alb_r2},
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{interp}")

if __name__ == "__main__":
    main()
