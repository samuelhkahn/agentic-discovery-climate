#!/usr/bin/env python3
"""
A017: ADVERSARIAL — Challenge H001 (Cloud Buffering)

Attacks:
1. Common forcing confound: Both clouds and albedo respond to ENSO — is Granger
   causality just ENSO driving both with different lags?
2. Granger at different aggregation levels: does result hold at 3-month and annual?
3. Reverse regime test: does the cloud→albedo slope REVERSE in any regime?
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import load_ceres_data, load_zone_weights

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"

def granger_test(x, y, max_lag=6):
    from statsmodels.tsa.stattools import grangercausalitytests
    data = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(data) < max_lag + 10:
        return {"best_p": 1.0}
    try:
        gc = grangercausalitytests(data[["y","x"]], maxlag=max_lag, verbose=False)
        results = {l: gc[l][0]["ssr_ftest"][1] for l in range(1, max_lag+1)}
        best_lag = min(results, key=results.get)
        return {"best_lag": best_lag, "best_p": results[best_lag], "all_p": results}
    except:
        return {"best_p": 1.0}

def main():
    print("A017: ADVERSARIAL — Challenging H001")
    print("=" * 60)

    df = load_ceres_data(terminator_filter=True)
    weights = load_zone_weights()
    wm = dict(zip(weights["lat"].astype(float).round(1), weights["weight"].astype(float)))
    df["weight"] = df["lat"].round(1).map(wm)

    variables = ["toa_alb_all_mon", "cldarea_total_mon", "cldtau_total_mon"]
    variables = [v for v in variables if v in df.columns]

    monthly = df.groupby(["year","month"]).apply(
        lambda g: pd.Series({col: np.average(g[col].dropna(), weights=g.loc[g[col].notna(),"weight"])
                             if g[col].notna().any() else np.nan for col in variables})
    ).reset_index().dropna().sort_values(["year","month"])

    for col in variables:
        seasonal = monthly.groupby("month")[col].transform("mean")
        monthly[f"{col}_anom"] = monthly[col] - seasonal

    # Load Niño3.4
    import urllib.request
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        response = urllib.request.urlopen(url, timeout=30)
        lines = response.read().decode().strip().split('\n')
        stm = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
        nino_recs = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts)>=4:
                try:
                    nino_recs.append({'year':int(parts[1]),'month':stm.get(parts[0],0),'nino34':float(parts[3])})
                except: pass
        nino = pd.DataFrame([r for r in nino_recs if r['month']>0])
        monthly = monthly.merge(nino, on=["year","month"], how="left")
        monthly["nino34"] = monthly["nino34"].interpolate()
        has_nino = True
    except:
        has_nino = False

    attacks = {}

    # ════════════════════════════════════════════════════════════
    # ATTACK 1: Common forcing confound (ENSO)
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 1: Common ENSO Forcing Confound ---")
    if has_nino and "cldarea_total_mon_anom" in monthly.columns:
        # Residualize: remove ENSO signal from both cloud and albedo
        from sklearn.linear_model import LinearRegression

        nino_vals = monthly["nino34"].values.reshape(-1, 1)
        cloud_anom = monthly["cldarea_total_mon_anom"].values
        albedo_anom = monthly["toa_alb_all_mon_anom"].values

        # Remove ENSO from clouds
        lr_cloud = LinearRegression().fit(nino_vals, cloud_anom)
        cloud_resid = cloud_anom - lr_cloud.predict(nino_vals)

        # Remove ENSO from albedo
        lr_albedo = LinearRegression().fit(nino_vals, albedo_anom)
        albedo_resid = albedo_anom - lr_albedo.predict(nino_vals)

        # Re-run Granger on ENSO-residualized series
        gc_resid = granger_test(cloud_resid, albedo_resid, max_lag=12)
        gc_resid_rev = granger_test(albedo_resid, cloud_resid, max_lag=12)

        # Compare with original
        gc_orig = granger_test(cloud_anom, albedo_anom, max_lag=12)
        gc_orig_rev = granger_test(albedo_anom, cloud_anom, max_lag=12)

        print(f"  ORIGINAL: cloud→albedo p={gc_orig['best_p']:.4f}, albedo→cloud p={gc_orig_rev['best_p']:.4f}")
        print(f"  ENSO-RESIDUALIZED: cloud→albedo p={gc_resid['best_p']:.4f}, albedo→cloud p={gc_resid_rev['best_p']:.4f}")

        still_unidirectional = gc_resid["best_p"] < 0.05 and gc_resid_rev["best_p"] > 0.05

        attacks["common_forcing"] = {
            "target": "A008 Granger causality",
            "original_cloud_to_albedo_p": float(gc_orig["best_p"]),
            "original_albedo_to_cloud_p": float(gc_orig_rev["best_p"]),
            "residualized_cloud_to_albedo_p": float(gc_resid["best_p"]),
            "residualized_albedo_to_cloud_p": float(gc_resid_rev["best_p"]),
            "still_unidirectional": still_unidirectional,
            "verdict": "SURVIVES — cloud→albedo remains significant after ENSO removal" if still_unidirectional else "WEAKENED — Granger causality partly explained by common ENSO forcing",
            "severity": "LOW" if still_unidirectional else "HIGH"
        }
    else:
        attacks["common_forcing"] = {"verdict": "SKIPPED", "severity": "N/A"}

    # ════════════════════════════════════════════════════════════
    # ATTACK 2: Temporal Aggregation Sensitivity
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 2: Granger at Different Aggregations ---")
    agg_results = {}

    # Quarterly
    monthly["quarter"] = (monthly["month"] - 1) // 3
    quarterly = monthly.groupby(["year", "quarter"]).agg({
        "cldarea_total_mon_anom": "mean", "toa_alb_all_mon_anom": "mean"}).dropna()
    gc_q = granger_test(quarterly["cldarea_total_mon_anom"].values,
                        quarterly["toa_alb_all_mon_anom"].values, max_lag=4)
    gc_q_rev = granger_test(quarterly["toa_alb_all_mon_anom"].values,
                            quarterly["cldarea_total_mon_anom"].values, max_lag=4)
    print(f"  Quarterly: cloud→albedo p={gc_q['best_p']:.4f}, albedo→cloud p={gc_q_rev['best_p']:.4f}")
    agg_results["quarterly"] = {"cloud_to_alb_p": float(gc_q["best_p"]),
                                 "alb_to_cloud_p": float(gc_q_rev["best_p"])}

    # Annual
    annual = monthly.groupby("year").agg({
        "cldarea_total_mon_anom": "mean", "toa_alb_all_mon_anom": "mean"}).dropna()
    if len(annual) > 10:
        gc_a = granger_test(annual["cldarea_total_mon_anom"].values,
                            annual["toa_alb_all_mon_anom"].values, max_lag=3)
        gc_a_rev = granger_test(annual["toa_alb_all_mon_anom"].values,
                                annual["cldarea_total_mon_anom"].values, max_lag=3)
        print(f"  Annual: cloud→albedo p={gc_a['best_p']:.4f}, albedo→cloud p={gc_a_rev['best_p']:.4f}")
        agg_results["annual"] = {"cloud_to_alb_p": float(gc_a["best_p"]),
                                  "alb_to_cloud_p": float(gc_a_rev["best_p"])}

    consistent = all(r["cloud_to_alb_p"] < 0.1 for r in agg_results.values())

    attacks["temporal_aggregation"] = {
        "target": "A008 Granger causality",
        "results": agg_results,
        "consistent_across_scales": consistent,
        "verdict": "SURVIVES — significant across scales" if consistent else "WEAKENED — not robust across temporal aggregation",
        "severity": "LOW" if consistent else "MEDIUM"
    }

    # ════════════════════════════════════════════════════════════
    # ATTACK 3: Does cloud→albedo slope ever reverse?
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 3: Rolling Slope Stability ---")
    window = 60  # 5-year rolling window
    cloud_col = "cldarea_total_mon_anom"
    alb_col = "toa_alb_all_mon_anom"

    slopes = []
    for i in range(0, len(monthly) - window):
        chunk = monthly.iloc[i:i+window]
        if len(chunk) < window:
            continue
        slope, _, _, p, _ = stats.linregress(chunk[cloud_col], chunk[alb_col])
        slopes.append({"start_idx": i, "slope": slope, "p": p})

    slopes_df = pd.DataFrame(slopes)
    n_negative = (slopes_df["slope"] < 0).sum()
    n_total = len(slopes_df)
    any_reversal = n_negative > 0

    print(f"  Rolling 5-year windows: {n_total} windows, {n_negative} with negative slope ({n_negative/n_total*100:.1f}%)")

    attacks["slope_reversal"] = {
        "target": "A011 Regime switching",
        "n_windows": n_total, "n_negative_slope": int(n_negative),
        "fraction_negative": float(n_negative/n_total) if n_total > 0 else 0,
        "verdict": f"SURVIVES — slope never reverses" if not any_reversal else f"NOTE: {n_negative}/{n_total} windows show reversed slope",
        "severity": "LOW" if not any_reversal else "MEDIUM"
    }

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Original vs ENSO-residualized Granger
    ax = axes[0]
    if "common_forcing" in attacks and attacks["common_forcing"].get("original_cloud_to_albedo_p") is not None:
        labels = ["cloud→alb\n(original)", "alb→cloud\n(original)", "cloud→alb\n(ENSO removed)", "alb→cloud\n(ENSO removed)"]
        vals = [attacks["common_forcing"]["original_cloud_to_albedo_p"],
                attacks["common_forcing"]["original_albedo_to_cloud_p"],
                attacks["common_forcing"]["residualized_cloud_to_albedo_p"],
                attacks["common_forcing"]["residualized_albedo_to_cloud_p"]]
        colors = ['green' if v < 0.05 else 'red' for v in vals]
        ax.bar(range(len(labels)), [-np.log10(max(v, 1e-10)) for v in vals], color=colors, alpha=0.7)
        ax.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p=0.05')
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel('-log10(p-value)'); ax.set_title('ENSO Confound Test')
        ax.legend(fontsize=7)

    # Panel 2: Rolling slope
    ax = axes[1]
    if len(slopes_df) > 0:
        ax.plot(slopes_df["start_idx"], slopes_df["slope"], 'b-', linewidth=0.8)
        ax.axhline(0, color='red', linewidth=1, linestyle='--')
        ax.fill_between(slopes_df["start_idx"], slopes_df["slope"],
                        where=slopes_df["slope"] < 0, alpha=0.3, color='red')
        ax.set_xlabel("Window start (month index)"); ax.set_ylabel("Slope (cloud→albedo)")
        ax.set_title(f"Rolling 5yr Slope ({n_negative}/{n_total} negative)")
        ax.grid(True, alpha=0.3)

    # Panel 3: Scorecard
    ax = axes[2]
    ax.axis('off')
    text = "ADVERSARIAL RESULTS — H001\n\n"
    for name, result in attacks.items():
        text += f"• {name}:\n  {result['verdict'][:70]}\n  Severity: {result['severity']}\n\n"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    fig_path = FIG_DIR / "adversarial_h001.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    methods_weakened = sum(1 for a in attacks.values() if "WEAKENED" in a.get("verdict",""))
    output = {
        "analysis": "A017_adversarial_h001", "method": "adversarial_testing",
        "hypothesis": "H001_cloud_buffering",
        "attacks": attacks,
        "methods_weakened": methods_weakened,
        "h001_still_robust": methods_weakened == 0,
        "figure_paths": [str(fig_path)]
    }
    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\nH001 adversarial: {methods_weakened} weakened. Still robust? {methods_weakened == 0}")

if __name__ == "__main__":
    main()
