# Adversarial Testing Plan

## Vulnerabilities Identified

### H002 (Invariant Properties) — MOST VULNERABLE
1. **HMM "2 states" is trivial**: Mean state duration = 301 months = the ENTIRE series. This means the HMM found essentially ONE state. This is not evidence of discrete regime structure — it's evidence of NO regime structure.
2. **Bayesian poly2 model has 10 parameters vs cloud model's 3**: BIC penalizes complexity but poly2 fitting lat²/lon²/month² on gridded data is partially fitting the SPATIAL STRUCTURE of Earth, not invariant physics. A random spatial field would also be fit well by lat/lon polynomials.
3. **Attractor dimension ~1.9 on 301 points**: Short time series may underestimate true dimensionality. Need to test sensitivity to series length.
4. **Reconciliation between linear refutation and nonlinear support**: If invariant constraints only appear nonlinearly, are they "invariant properties" or just "spatial structure"?

### H003 (Teleconnections) — MODERATE VULNERABILITY
5. **ENSO-conditional TE on 88 La Niña months**: Small sample → low power → possible false positive from subsampling noise. Need bootstrap validation.
6. **Transfer entropy with 6 bins on 301 points**: Binning choice may inflate significance. Test sensitivity to n_bins.

### H001 (Cloud Buffering) — SOME VULNERABILITY
7. **Granger causality at monthly resolution**: Monthly aggregation destroys sub-monthly causal structure. Cloud changes and albedo changes may be simultaneous at finer resolution — Granger at monthly scale may find spurious one-directionality.
8. **Common forcing confound**: Both clouds and albedo may respond to ENSO/large-scale circulation. The "unidirectional" Granger result could be an artifact of clouds responding FASTER to a common driver.

### Cross-Cutting
9. **Multiple comparison problem**: 14 analyses × 3 hypotheses = many tests. Some "significant" results may be false positives. Need Bonferroni/FDR correction across the full battery.

## Adversarial Cycles

- **Cycle 15**: Attack H002 (HMM triviality, Bayesian spatial confound, attractor sensitivity)
- **Cycle 16**: Attack H003 (bootstrap ENSO-conditional, TE bin sensitivity)
- **Cycle 17**: Attack H001 (Granger resolution, common forcing confound)
- **Cycle 18**: Multiple comparison correction + overall robustness assessment
