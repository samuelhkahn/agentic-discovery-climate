#!/usr/bin/env python3
"""
A008: Granger Causality — Multi-hypothesis test

Tests H001: Does cloud area Granger-cause albedo? (cloud buffering)
Tests H003: Does NH albedo Granger-cause SH albedo? (teleconnections)
Tests H001: Does albedo Granger-cause cloud area? (feedback direction)

Granger causality: X Granger-causes Y if past values of X improve
prediction of Y beyond Y's own past.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, compute_monthly_hemispheric_timeseries, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"

def granger_test(x, y, max_lag=12):
    """Test if x Granger-causes y at lags 1..max_lag using F-test."""
    from statsmodels.tsa.stattools import grangercausalitytests
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(data) < max_lag + 10:
        return {"best_lag": None, "best_p": 1.0, "results": {}}

    results = {}
    try:
        gc = grangercausalitytests(data[["y", "x"]], maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f_test = gc[lag][0]["ssr_ftest"]
            results[lag] = {"f_stat": f_test[0], "p_value": f_test[1], "df": (f_test[2], f_test[3])}
    except Exception as e:
        return {"best_lag": None, "best_p": 1.0, "error": str(e), "results": {}}

    if results:
        best_lag = min(results, key=lambda l: results[l]["p_value"])
        return {"best_lag": best_lag, "best_p": results[best_lag]["p_value"],
                "best_f": results[best_lag]["f_stat"], "results": results}
    return {"best_lag": None, "best_p": 1.0, "results": {}}

def main():
    print("A008: Granger Causality — Multi-hypothesis Test")
    print("=" * 60)

    # Hemispheric time series
    ts = compute_monthly_hemispheric_timeseries()
    nh_seasonal = ts.groupby("month")["NH"].transform("mean")
    sh_seasonal = ts.groupby("month")["SH"].transform("mean")
    ts["NH_anom"] = ts["NH"] - nh_seasonal
    ts["SH_anom"] = ts["SH"] - sh_seasonal

    # Global means of cloud variables
    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    for col in ["cldarea_total_mon", "cldtau_total_mon", "toa_alb_all_mon"]:
        if col in df.columns:
            monthly_col = df.groupby(["year","month"]).apply(
                lambda g: np.average(g[col].dropna(), weights=g.loc[g[col].notna(),"weight"])
                if g[col].notna().any() else np.nan
            ).reset_index(name=col)
            ts = ts.merge(monthly_col, on=["year","month"], how="left")

    # Deseasonalize cloud vars
    for col in ["cldarea_total_mon", "cldtau_total_mon"]:
        if col in ts.columns:
            s = ts.groupby("month")[col].transform("mean")
            ts[f"{col}_anom"] = ts[col] - s

    ts = ts.dropna()
    print(f"Time series: {len(ts)} months")

    # ── Run Granger tests ──
    tests = {
        "cloud_area → albedo (H001)": ("cldarea_total_mon_anom", "toa_alb_all_mon"),
        "albedo → cloud_area (H001 reverse)": ("toa_alb_all_mon", "cldarea_total_mon_anom"),
        "cloud_tau → albedo (H001)": ("cldtau_total_mon_anom", "toa_alb_all_mon"),
        "NH → SH albedo (H003)": ("NH_anom", "SH_anom"),
        "SH → NH albedo (H003)": ("SH_anom", "NH_anom"),
    }

    gc_results = {}
    print()
    for name, (x_col, y_col) in tests.items():
        if x_col in ts.columns and y_col in ts.columns:
            result = granger_test(ts[x_col].values, ts[y_col].values, max_lag=12)
            gc_results[name] = result
            sig = "***" if result["best_p"] < 0.001 else "**" if result["best_p"] < 0.01 else "*" if result["best_p"] < 0.05 else "ns"
            print(f"  {name}: p={result['best_p']:.4f} at lag={result['best_lag']} {sig}")

    # ── Plotting ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: P-values vs lag for H001 tests
    ax = axes[0]
    for name in ["cloud_area → albedo (H001)", "albedo → cloud_area (H001 reverse)", "cloud_tau → albedo (H001)"]:
        if name in gc_results and gc_results[name]["results"]:
            lags = sorted(gc_results[name]["results"].keys())
            pvals = [gc_results[name]["results"][l]["p_value"] for l in lags]
            ax.semilogy(lags, pvals, 'o-', label=name[:30], markersize=4)
    ax.axhline(0.05, color='gray', linestyle='--', label='p=0.05')
    ax.axhline(0.01, color='gray', linestyle=':', label='p=0.01')
    ax.set_xlabel("Lag (months)"); ax.set_ylabel("p-value")
    ax.set_title("Granger Causality: Cloud → Albedo (H001)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 2: P-values for H003 tests
    ax = axes[1]
    for name in ["NH → SH albedo (H003)", "SH → NH albedo (H003)"]:
        if name in gc_results and gc_results[name]["results"]:
            lags = sorted(gc_results[name]["results"].keys())
            pvals = [gc_results[name]["results"][l]["p_value"] for l in lags]
            ax.semilogy(lags, pvals, 'o-', label=name[:25], markersize=4)
    ax.axhline(0.05, color='gray', linestyle='--', label='p=0.05')
    ax.axhline(0.01, color='gray', linestyle=':', label='p=0.01')
    ax.set_xlabel("Lag (months)"); ax.set_ylabel("p-value")
    ax.set_title("Granger Causality: NH ↔ SH (H003)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "granger_causality.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # Interpretation
    h001_cloud_alb = gc_results.get("cloud_area → albedo (H001)", {})
    h001_alb_cloud = gc_results.get("albedo → cloud_area (H001 reverse)", {})
    h003_nh_sh = gc_results.get("NH → SH albedo (H003)", {})
    h003_sh_nh = gc_results.get("SH → NH albedo (H003)", {})

    output = {
        "analysis": "A008_granger_causality", "method": "granger_causality",
        "hypotheses_tested": ["H001", "H003"],
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                    for k, v in gc_results.items()},
        "results_full": {k: v for k, v in gc_results.items()},
        "interpretation_H001": "",
        "interpretation_H003": "",
        "figure_paths": [str(fig_path)],
    }

    # H001 interpretation
    if h001_cloud_alb.get("best_p", 1) < 0.05:
        output["interpretation_H001"] = f"Cloud area Granger-causes albedo (p={h001_cloud_alb['best_p']:.4f}, lag={h001_cloud_alb['best_lag']}). Supports H001."
        output["H001_support"] = "supports"
    else:
        output["interpretation_H001"] = f"Cloud area does NOT Granger-cause albedo (p={h001_cloud_alb.get('best_p',1):.4f}). No linear causal evidence for H001."
        output["H001_support"] = "inconclusive"

    # H003 interpretation
    h003_any_sig = (h003_nh_sh.get("best_p", 1) < 0.05) or (h003_sh_nh.get("best_p", 1) < 0.05)
    if h003_any_sig:
        output["interpretation_H003"] = f"Granger causality between hemispheres detected. NH→SH: p={h003_nh_sh.get('best_p',1):.4f}, SH→NH: p={h003_sh_nh.get('best_p',1):.4f}. Supports H003."
        output["H003_support"] = "supports"
    else:
        output["interpretation_H003"] = "No Granger causality between hemispheres. H003 not supported by this linear test."
        output["H003_support"] = "inconclusive"

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))

    print(f"\nH001: {output['interpretation_H001']}")
    print(f"H003: {output['interpretation_H003']}")

if __name__ == "__main__":
    main()
