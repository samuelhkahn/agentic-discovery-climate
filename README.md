# Agentic Discovery Climate

An autonomous AI scientist system — inspired by [Kosmos](https://arxiv.org/abs/2511.02824) — that explores hypotheses about Earth's top-of-atmosphere albedo stability and hemispheric symmetry using novel methodologies.

This system was applied to the CERES EBAF satellite dataset to test three hypotheses from [Feldman et al. (2026) *Nature*](https://github.com/samuelhkahn/ai-albedo-research), discovering results across 12 methodologies that were not explored in the original paper.

## Key Discoveries

| Hypothesis | Status | Key Method | Key Finding |
|---|---|---|---|
| **Cloud Buffering** | Supported (conf=0.85) | Convergent Cross Mapping | Clouds *causally* drive albedo (p=0.001) but not vice versa |
| **Invariant Properties** | Inconclusive (conf=0.50) | Attractor Reconstruction | Albedo on ~2D attractor but hard to attribute to invariants |
| **Teleconnections** | Supported (conf=0.90) | Transfer Entropy + 5 others | ENSO mediates NH-SH coupling via tropical Pacific hubs |

**Unified mechanism**: Earth's geometry sets a ~2D attractor envelope → clouds causally maintain albedo within it → ENSO coordinates clouds across hemispheres to maintain symmetry.

## Reports

- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) — Complete report of all 17 analyses with methodology explanations, results, and flowcharts
- [`FINDINGS_SYNTHESIS.md`](FINDINGS_SYNTHESIS.md) — Synthesis of discoveries and unified mechanism
- [`kosmos_albedo_design.md`](kosmos_albedo_design.md) — System design document

## Project Structure

```
agentic-discovery-climate/
├── README.md
├── EXPERIMENT_REPORT.md         # Full experiment report
├── FINDINGS_SYNTHESIS.md        # Discovery synthesis
├── kosmos_albedo_design.md      # System design
├── scripts/
│   ├── utils.py                 # Data loading, world model I/O
│   ├── run_cycle.py             # Orchestration engine
│   └── download_all_data.py     # Data download helpers
└── world_model/
    ├── meta.json                # System state
    ├── hypotheses.json          # Hypothesis tracker with full history
    ├── exploration_map.json     # Method × hypothesis coverage matrix
    ├── literature.json          # Literature knowledge base
    ├── prior_work.yaml          # What the Nature paper already tried
    ├── analyses/                # 17 analysis scripts
    │   ├── A001_transfer_entropy_hemispheric/
    │   ├── A002_wavelet_coherence_hemispheric/
    │   ├── A003_enso_conditional_transfer_entropy/
    │   ├── A004_attractor_reconstruction/
    │   ├── A006_climate_networks/
    │   ├── A007_symbolic_regression/
    │   ├── A008_granger_causality/
    │   ├── A009_variance_decomposition/
    │   ├── A010_ica_decomposition/
    │   ├── A011_regime_switching_h001/
    │   ├── A012_sindy_h002/
    │   ├── A013_bayesian_model_comparison/
    │   ├── A014_hidden_markov_h002/
    │   ├── A015_adversarial_h002/
    │   ├── A016_adversarial_h003/
    │   ├── A017_adversarial_h001/
    │   └── A019_ccm_fixed/
    ├── findings/                # 21 structured findings (JSON)
    └── cycle_summaries/         # 19 cycle logs + decision documents
```

## Data Setup

Data files are not included in the repo (too large). To reproduce:

### 1. CERES EBAF-TOA Ed4.2.1

Download from [NASA Earthdata](https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF-TOA_Edition4.2.1):

```bash
# Requires NASA Earthdata account (~/.netrc credentials)
python scripts/download_all_data.py ceres
```

Or manually download from the [CERES ordering tool](https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA421Selection.jsp).

Place parquet files in `../data/` relative to the repo root.

### 2. Zone Weights

Copy `zone_weights_lou.txt` from the [original research repo](https://github.com/samuelhkahn/ai-albedo-research).

### 3. MERRA-2 (optional, for Pinatubo analysis)

```bash
python scripts/download_all_data.py merra2
```

## Requirements

```
numpy pandas scipy scikit-learn matplotlib seaborn
statsmodels pywavelets networkx torch xarray netCDF4
pyarrow hmmlearn nolds
```

## Running the System

```bash
# Check current state and get next cycle prompt
python scripts/run_cycle.py

# Run a specific analysis
python world_model/analyses/A001_transfer_entropy_hemispheric/script.py
```

## Methodology Overview

12 novel methodologies applied (none used in the original Nature paper):

| Method | Family | What it tests |
|---|---|---|
| Transfer Entropy | Information Theory | Directional information flow NH↔SH |
| Wavelet Coherence | Spectral | Time-frequency structure of coupling |
| Convergent Cross Mapping | Causal Inference | Nonlinear causality (cloud→albedo) |
| Climate Networks | Graph Theory | Spatial topology of teleconnections |
| Attractor Reconstruction | Dynamical Systems | Dimensionality of albedo dynamics |
| Symbolic Regression | Equation Discovery | Interpretable albedo equation |
| Granger Causality | Causal Inference | Linear temporal causality |
| ICA | Signal Processing | Independent sources of variability |
| Regime Switching | Statistics | Stability across regimes |
| SINDy | Equation Discovery | Governing dynamical equations |
| Bayesian Model Comparison | Bayesian | Hypothesis competition via BIC |
| Hidden Markov Model | Probabilistic | Discrete regime structure |

## Citation

If using this system or its findings:

```
Kahn, S. (2026). Agentic Discovery Climate: An AI Scientist for Earth's Albedo
Stability and Hemispheric Symmetry. https://github.com/samuelhkahn/agentic-discovery-climate

Built on: Feldman, D.R., Gristey, J.J., Hakuba, M.Z., Hellinger, D., & Kahn, S.
(2026). Towards a theory for Earth's albedo stability and hemispheric symmetry
in the 21st Century. Nature.
```

## License

MIT
