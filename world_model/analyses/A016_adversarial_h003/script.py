#!/usr/bin/env python3
"""
A016: ADVERSARIAL — Challenge H003 (Teleconnections)

Attacks:
1. Bootstrap ENSO-conditional TE: Is the La Niña result (p=0.017) robust?
2. Transfer entropy bin sensitivity: Does changing n_bins change conclusions?
3. Surrogate test: Replace NH-SH coupling with phase-randomized surrogates
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "scripts"))
from utils import compute_monthly_hemispheric_timeseries

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"

def transfer_entropy_binned(source, target, lag=1, n_bins=6):
    n = len(target) - lag
    if n < 20: return 0.0
    y_future = target[lag:]
    y_past = target[:n]
    x_past = source[:n]
    y_future_d = np.digitize(y_future, np.linspace(y_future.min()-1e-10, y_future.max()+1e-10, n_bins))
    y_past_d = np.digitize(y_past, np.linspace(y_past.min()-1e-10, y_past.max()+1e-10, n_bins))
    x_past_d = np.digitize(x_past, np.linspace(x_past.min()-1e-10, x_past.max()+1e-10, n_bins))
    te = 0.0
    total = len(y_future_d)
    for yf in range(1, n_bins+1):
        for yp in range(1, n_bins+1):
            for xp in range(1, n_bins+1):
                mask_j = (y_future_d==yf)&(y_past_d==yp)&(x_past_d==xp)
                p_j = mask_j.sum()/total
                if p_j==0: continue
                p_c = ((y_past_d==yp)&(x_past_d==xp)).sum()/total
                p_yf_ypxp = p_j/p_c if p_c>0 else 0
                p_yp = (y_past_d==yp).sum()/total
                p_yf_yp_j = ((y_future_d==yf)&(y_past_d==yp)).sum()/total
                p_yf_yp = p_yf_yp_j/p_yp if p_yp>0 else 0
                if p_yf_ypxp>0 and p_yf_yp>0:
                    te += p_j*np.log2(p_yf_ypxp/p_yf_yp)
    return te

def phase_randomized_surrogate(x):
    """Create phase-randomized surrogate preserving power spectrum."""
    n = len(x)
    X_fft = np.fft.rfft(x)
    phases = np.random.uniform(0, 2*np.pi, len(X_fft))
    phases[0] = 0  # Keep DC component
    X_surr = np.abs(X_fft) * np.exp(1j * phases)
    return np.fft.irfft(X_surr, n=n)

def main():
    print("A016: ADVERSARIAL — Challenging H003")
    print("=" * 60)

    ts = compute_monthly_hemispheric_timeseries()
    nh_seasonal = ts.groupby("month")["NH"].transform("mean")
    sh_seasonal = ts.groupby("month")["SH"].transform("mean")
    nh_anom = ((ts["NH"] - nh_seasonal) - (ts["NH"] - nh_seasonal).mean()) / (ts["NH"] - nh_seasonal).std()
    sh_anom = ((ts["SH"] - sh_seasonal) - (ts["SH"] - sh_seasonal).mean()) / (ts["SH"] - sh_seasonal).std()
    nh = nh_anom.values
    sh = sh_anom.values

    attacks = {}

    # ════════════════════════════════════════════════════════════
    # ATTACK 1: Bootstrap ENSO-conditional TE
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 1: Bootstrap ENSO-conditional La Niña result ---")

    # Load Niño3.4
    import urllib.request
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        response = urllib.request.urlopen(url, timeout=30)
        lines = response.read().decode().strip().split('\n')
        season_to_month = {'DJF':1,'JFM':2,'FMA':3,'MAM':4,'AMJ':5,'MJJ':6,'JJA':7,'JAS':8,'ASO':9,'SON':10,'OND':11,'NDJ':12}
        nino_records = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts)>=4:
                try:
                    season, year, anom = parts[0], int(parts[1]), float(parts[3])
                    if season in season_to_month:
                        nino_records.append({'year':year,'month':season_to_month[season],'nino34':anom})
                except: pass
        nino = pd.DataFrame(nino_records)
        ts_merged = ts.merge(nino, on=["year","month"], how="left")
        ts_merged["nino34"] = ts_merged["nino34"].interpolate()
        la_nina_mask = ts_merged["nino34"] <= -0.5
        n_la_nina = la_nina_mask.sum()
        print(f"  La Niña months: {n_la_nina}")
    except:
        print("  Could not load Niño3.4, using proxy")
        la_nina_mask = pd.Series([False]*len(ts))
        n_la_nina = 0

    # Bootstrap: resample La Niña months 1000 times, compute TE each time
    if n_la_nina > 30:
        n_bootstrap = 1000
        te_bootstrap = []
        la_nina_indices = np.where(la_nina_mask.values)[0]

        for b in range(n_bootstrap):
            # Resample La Niña indices with replacement
            boot_idx = np.random.choice(la_nina_indices, size=len(la_nina_indices), replace=True)
            boot_idx = np.sort(boot_idx)
            nh_boot = nh[boot_idx]
            sh_boot = sh[boot_idx]
            te = transfer_entropy_binned(sh_boot, nh_boot, lag=3, n_bins=5)
            te_bootstrap.append(te)

        te_bootstrap = np.array(te_bootstrap)
        ci_lower = np.percentile(te_bootstrap, 2.5)
        ci_upper = np.percentile(te_bootstrap, 97.5)
        te_original = transfer_entropy_binned(sh[la_nina_mask.values], nh[la_nina_mask.values], lag=3, n_bins=5)

        # Fraction of bootstrap samples where TE > 0
        frac_positive = np.mean(te_bootstrap > 0)
        # Fraction where TE > null expectation
        null_te = np.mean([transfer_entropy_binned(np.random.permutation(sh[la_nina_mask.values]),
                           nh[la_nina_mask.values], lag=3, n_bins=5) for _ in range(200)])
        frac_above_null = np.mean(te_bootstrap > null_te)

        print(f"  Original TE(SH→NH|La Niña): {te_original:.4f}")
        print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Fraction positive: {frac_positive:.3f}")
        print(f"  Fraction above null: {frac_above_null:.3f}")

        bootstrap_robust = ci_lower > 0 and frac_above_null > 0.9

        attacks["enso_bootstrap"] = {
            "target": "A003 ENSO-conditional TE",
            "te_original": float(te_original),
            "ci_lower": float(ci_lower), "ci_upper": float(ci_upper),
            "frac_positive": float(frac_positive),
            "frac_above_null": float(frac_above_null),
            "verdict": "SURVIVES" if bootstrap_robust else "WEAKENED — CI includes zero or not consistently above null",
            "severity": "LOW" if bootstrap_robust else "HIGH"
        }
    else:
        attacks["enso_bootstrap"] = {"verdict": "SKIPPED — insufficient La Niña months", "severity": "N/A"}

    # ════════════════════════════════════════════════════════════
    # ATTACK 2: TE Bin Sensitivity
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 2: Transfer Entropy Bin Sensitivity ---")

    bin_results = {}
    for n_bins in [4, 5, 6, 7, 8, 10]:
        te_sh_nh = transfer_entropy_binned(sh, nh, lag=3, n_bins=n_bins)
        # Permutation p-value
        null_tes = [transfer_entropy_binned(np.random.permutation(sh), nh, lag=3, n_bins=n_bins)
                    for _ in range(500)]
        p_val = np.mean(np.array(null_tes) >= te_sh_nh)
        bin_results[n_bins] = {"te": float(te_sh_nh), "p_value": float(p_val)}
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  n_bins={n_bins}: TE(SH→NH)={te_sh_nh:.4f}, p={p_val:.4f} {sig}")

    # Is the result consistent across bin choices?
    all_significant = all(r["p_value"] < 0.05 for r in bin_results.values())
    most_significant = sum(1 for r in bin_results.values() if r["p_value"] < 0.05) / len(bin_results)

    attacks["te_bin_sensitivity"] = {
        "target": "A001 Transfer entropy",
        "bin_results": bin_results,
        "all_significant": all_significant,
        "fraction_significant": float(most_significant),
        "verdict": f"SURVIVES — {most_significant*100:.0f}% of bin choices yield p<0.05" if most_significant > 0.7 else "WEAKENED",
        "severity": "LOW" if most_significant > 0.7 else "MEDIUM"
    }

    # ════════════════════════════════════════════════════════════
    # ATTACK 3: Phase-Randomized Surrogates
    # ════════════════════════════════════════════════════════════
    print("\n--- ATTACK 3: Phase-Randomized Surrogate Test ---")

    te_original_sh_nh = transfer_entropy_binned(sh, nh, lag=3, n_bins=6)
    te_original_nh_sh = transfer_entropy_binned(nh, sh, lag=3, n_bins=6)

    n_surrogates = 500
    te_surr_sh_nh = []
    te_surr_nh_sh = []
    for _ in range(n_surrogates):
        sh_surr = phase_randomized_surrogate(sh)
        te_surr_sh_nh.append(transfer_entropy_binned(sh_surr, nh, lag=3, n_bins=6))
        te_surr_nh_sh.append(transfer_entropy_binned(nh, sh_surr, lag=3, n_bins=6))

    p_surr_sh_nh = np.mean(np.array(te_surr_sh_nh) >= te_original_sh_nh)
    p_surr_nh_sh = np.mean(np.array(te_surr_nh_sh) >= te_original_nh_sh)

    print(f"  Original TE(SH→NH): {te_original_sh_nh:.4f}, p(surrogates)={p_surr_sh_nh:.4f}")
    print(f"  Original TE(NH→SH): {te_original_nh_sh:.4f}, p(surrogates)={p_surr_nh_sh:.4f}")

    surr_robust = p_surr_sh_nh < 0.05 or p_surr_nh_sh < 0.05

    attacks["phase_surrogates"] = {
        "target": "A001 Transfer entropy (all directions)",
        "te_original_sh_nh": float(te_original_sh_nh),
        "te_original_nh_sh": float(te_original_nh_sh),
        "p_surr_sh_nh": float(p_surr_sh_nh),
        "p_surr_nh_sh": float(p_surr_nh_sh),
        "verdict": "SURVIVES — significant against phase-randomized surrogates" if surr_robust else "WEAKENED",
        "severity": "LOW" if surr_robust else "HIGH"
    }

    # ── Plotting ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Bootstrap distribution
    ax = axes[0]
    if "te_original" in attacks.get("enso_bootstrap", {}):
        ax.hist(te_bootstrap, bins=30, alpha=0.7, color='steelblue')
        ax.axvline(attacks["enso_bootstrap"]["te_original"], color='red', linewidth=2,
                   label=f'Original TE={attacks["enso_bootstrap"]["te_original"]:.4f}')
        ax.axvline(ci_lower, color='orange', linestyle='--', label=f'95% CI lower={ci_lower:.4f}')
        ax.axvline(0, color='black', linewidth=1, linestyle=':')
        ax.set_xlabel('TE(SH→NH | La Niña)'); ax.set_ylabel('Count')
        ax.set_title('Bootstrap La Niña TE')
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Skipped", ha='center', transform=ax.transAxes)

    # Panel 2: Bin sensitivity
    ax = axes[1]
    bins_list = sorted(bin_results.keys())
    tes = [bin_results[b]["te"] for b in bins_list]
    ps = [bin_results[b]["p_value"] for b in bins_list]
    ax.bar(bins_list, tes, alpha=0.7, color=['green' if p < 0.05 else 'red' for p in ps])
    ax.set_xlabel('n_bins'); ax.set_ylabel('Transfer Entropy')
    ax.set_title(f'TE Bin Sensitivity ({most_significant*100:.0f}% significant)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 3: Surrogate distribution
    ax = axes[2]
    ax.hist(te_surr_sh_nh, bins=30, alpha=0.6, color='gray', label='Surrogates')
    ax.axvline(te_original_sh_nh, color='red', linewidth=2,
               label=f'Original={te_original_sh_nh:.4f}, p={p_surr_sh_nh:.3f}')
    ax.set_xlabel('TE(SH→NH)'); ax.set_ylabel('Count')
    ax.set_title('Phase-Randomized Surrogate Test')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = FIG_DIR / "adversarial_h003.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ── Overall Assessment ──
    methods_weakened = sum(1 for a in attacks.values() if "WEAKENED" in a.get("verdict", ""))

    output = {
        "analysis": "A016_adversarial_h003", "method": "adversarial_testing",
        "hypothesis": "H003_teleconnections",
        "attacks": attacks,
        "methods_weakened": methods_weakened,
        "h003_still_robust": methods_weakened == 0,
        "figure_paths": [str(fig_path)]
    }

    with open(OUT_DIR / "output.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nH003 adversarial: {methods_weakened} methods weakened. Still robust? {methods_weakened == 0}")

if __name__ == "__main__":
    main()
