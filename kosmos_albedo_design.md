# Design Document: Kosmos-Inspired AI Scientist for Earth's Albedo Research

## Context

The Nature paper and codebase establish what has **already been tried**:
- **Models**: FFNN (5-layer), U-Net, Prophet BSTS, ExtraTrees, Linear Regression, Explainable Boosting
- **Methods**: SHAP feature attribution, perturbation analysis, hemispheric decomposition, additive impact analysis, kernel time-varying regression
- **Hypotheses explored**: Cloud Buffering (CAF vs surface albedo), Invariant Properties (implicit in SHAP/latitude importance), Teleconnections (U-Net ablation)

The goal is an **exploration-first AI scientist** that treats the above as a **known baseline** and autonomously discovers **novel methodologies, models, and hypothesis tests** across the full (model × hypothesis × solution approach) space — approaches the research team hasn't considered.

---

## 1. Core Design Principle: Explore the Unknown

The agent receives:
1. **The data** (CERES EBAF, MERRA-2, CMIP6)
2. **The hypotheses** (Cloud Buffering, Invariant Properties, Teleconnections, + emergent)
3. **What has already been tried** (the existing notebook analyses, summarized as a "prior work" document)

Its mandate: **Find approaches we didn't think of.** The agent should:
- Propose novel statistical/ML methods not in the existing codebase
- Test hypotheses using frameworks from other scientific domains
- Discover unexpected connections in the data
- Challenge the existing results with alternative methodologies
- Explore the full combinatorial space of (model × hypothesis × analysis approach)

---

## 2. Prior Work Summary (Input to Agent)

The agent is seeded with a structured summary of what's been done, so it can focus on what **hasn't**:

```yaml
prior_work:
  models_used:
    - FFNN (5-layer, ReLU, BatchNorm) → val_loss 0.001
    - U-Net (encoder-decoder, 64→1024 channels)
    - Prophet BSTS with MCMC
    - ExtraTrees Regressor
    - Linear Regression
    - Explainable Boosting Regressor

  methods_used:
    - SHAP (DeepSHAP + TreeSHAP) for feature importance
    - SHAP interaction matrices
    - Hemisphere-specific perturbation experiments via FFNN
    - Prophet additive decomposition (per-feature regression coefficients)
    - Kernel time-varying regression (Orbit-ML)
    - U-Net ablation study (stochastic noise injection)
    - Multi-resolution latitude binning (5°→55°)
    - MinMaxScaler normalization
    - Cosine-latitude area weighting
    - Terminator filtering (|lat| < 66°)

  hypotheses_tested:
    - Cloud Buffering: CAF responds inversely to surface albedo (FFNN perturbation) → SUPPORTED
    - Feature importance: Cloud properties dominate, latitude is key spatial feature → CONFIRMED
    - Model complexity: FFNN outperforms MLR (AIC comparison) → CONFIRMED
    - Teleconnections: No single region controls global albedo (U-Net single-gridbox test) → NULL RESULT
    - FFNN temporal degradation: Model skill degrades ~0.002 bias over 4 years withheld → OBSERVED

  gaps_identified:
    - No causal inference methods used (only correlational/attributional)
    - No information-theoretic analysis
    - No spectral/frequency-domain analysis
    - No graph-based or network approaches
    - No Bayesian model comparison or model averaging
    - No explicit dynamical systems analysis
    - No coupling with ocean/atmospheric circulation indices
    - No regime-switching or hidden Markov models
    - No optimal transport or Wasserstein distance approaches
    - No symbolic regression or equation discovery
    - CMIP6 comparison not yet implemented in code
    - No cross-validated temporal predictions (only spatial cross-validation)
```

---

## 3. Methodology Exploration Space

The agent should systematically explore methodologies **not yet tried**. Each method is rated on:
- **Priority** (1-5): How likely is this to produce novel, publishable insights for albedo research? Based on: match to the physics, feasibility with available data, potential to discriminate between hypotheses.
- **Differentiation** (1-5): How different is this from what's already been tried? 5 = entirely novel paradigm; 1 = incremental variant of existing approach.

### Causal and Counterfactual Methods
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Convergent cross mapping | Causality in nonlinear coupled systems (cloud↔albedo) | **5** | **5** | Only method that detects causality in deterministic nonlinear systems; directly tests whether cloud buffering is *causal* vs. correlational. No analogue in existing work. |
| Instrumental variables (volcanic eruptions) | Use Pinatubo as natural experiment for causal albedo response | **5** | **4** | Volcanic events are near-ideal instruments; paper discusses this conceptually but never implements formal IV regression. Bridges theory and data. |
| DoWhy / causal DAG | Full causal graph: cloud→albedo, aerosol→cloud→albedo, surface→cloud→albedo | **4** | **5** | Entirely different paradigm from SHAP (attributional, not causal). Could reveal confounded pathways missed by correlational methods. |
| Granger causality | Do cloud changes temporally precede albedo changes? | **3** | **3** | Well-established but linear and somewhat redundant with Prophet decomposition. Useful as baseline causal test. |
| Synthetic control | Counterfactual "no-cloud-buffering" albedo trajectory | **4** | **4** | Novel framing; constructs what albedo would look like without cloud compensation. Directly tests H001. |

### Information Theory
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Transfer entropy (NH↔SH) | Directional information flow between hemispheres | **5** | **5** | Directly measures information transfer across hemispheres — the core teleconnection question (H003). Nonlinear, model-free. Nothing like this in existing work. |
| Predictive information decomposition (PID) | Unique vs. redundant vs. synergistic info from features about albedo | **5** | **5** | Goes far beyond SHAP: identifies which features carry *unique* information vs. *synergistic* information (only informative jointly). Could reveal hidden feature couplings. |
| Mutual information (features→albedo) | Nonlinear dependence without parametric assumptions | **3** | **3** | Useful but somewhat overlaps with SHAP for feature importance. Main advantage: captures dependence SHAP misses. |
| Complexity measures (permutation entropy) | Is albedo time series complexity changing? | **3** | **4** | Novel angle — tracks whether the albedo system is becoming more/less predictable over time. Could signal approaching tipping point. |

### Dynamical Systems
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Attractor reconstruction (Takens embedding) | Is albedo governed by a low-dimensional attractor? | **5** | **5** | Fundamentally different question from all existing work. If albedo lives on a low-D attractor, it explains stability without requiring cloud buffering. Tests H002 from dynamical systems perspective. |
| Lyapunov exponents | Quantitative stability of the albedo system | **4** | **5** | Provides a *number* for stability — how fast perturbations decay. Directly quantifies the "stability" in "albedo stability." Not attempted in any albedo literature. |
| Recurrence quantification analysis | Detect regime transitions and recurrence patterns | **3** | **4** | Could reveal hidden state transitions not visible in mean albedo time series. |
| Bifurcation analysis | Are there tipping points in parameter space? | **3** | **5** | High differentiation but feasibility uncertain — requires model of albedo dynamics, not just data. Could use FFNN as surrogate model. |

### Spectral and Frequency Domain
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Wavelet coherence (NH-SH) | Time-varying frequency coupling between hemispheres | **5** | **4** | Reveals *when* and *at what timescale* hemispheres are coupled. Feldman et al. (2021) did frequency analysis of reflected SW but not wavelet coherence between hemispheres. |
| Singular spectrum analysis | Extract oscillatory modes from albedo | **4** | **4** | Separates albedo into trend + oscillatory modes + noise. Complements Prophet decomposition with a nonparametric approach. |
| Cross-spectral analysis | Phase relationships between albedo components | **3** | **3** | Standard but informative; reveals whether cloud and surface components are in-phase or anti-phase (cloud buffering signature). |
| Spectral Granger causality | Frequency-specific causal links | **2** | **3** | Specialized variant; lower priority unless initial spectral analysis reveals frequency-specific phenomena. |

### Graph and Network Methods
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Climate networks (correlation graphs) | Teleconnection topology of albedo field | **5** | **5** | Entirely new paradigm for albedo. Tsonis et al. and Donges et al. used climate networks for temperature; never applied to albedo. Could reveal the teleconnection structure the U-Net didn't find. |
| Community detection | Identify coupled albedo regions | **4** | **5** | Clusters the globe into albedo "communities" — regions that co-vary. Could answer: is symmetry maintained by a few key cross-hemispheric links? |
| Network centrality analysis | Which regions are "hubs" for albedo? | **4** | **4** | Complements U-Net single-gridbox analysis but from a graph theory perspective. |
| Graph neural networks | Spatially-aware albedo prediction | **2** | **2** | Incremental over U-Net; higher complexity without guaranteed insight. Lower priority. |

### Bayesian and Probabilistic
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Bayesian model comparison (WAIC/LOO-CV) | Which hypothesis family best explains the data? | **5** | **4** | Directly compares H001 vs H002 vs H003 as competing models. Existing work only compared MLR vs FFNN via AIC. This puts *hypotheses* into competition, not just models. |
| Bayesian hierarchical models | Pool information across latitudes/hemispheres | **4** | **4** | Allows partial pooling — learn global albedo structure while allowing local variation. Natural framework for symmetry analysis. |
| Gaussian processes | Probabilistic albedo with uncertainty | **3** | **3** | Provides calibrated uncertainty estimates the FFNN doesn't. Useful but computationally expensive on 17M rows. |
| Bayesian change-point detection | Structural breaks in albedo time series | **3** | **3** | Overlaps somewhat with Prophet trend detection. Main advantage: principled uncertainty on break locations. |
| Probabilistic graphical models | Joint distribution of all albedo drivers | **3** | **4** | Could reveal conditional independence structure (which drivers are independent given others?). Novel angle. |

### Equation Discovery and Symbolic
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Symbolic regression (PySR) | Discover interpretable albedo equations | **5** | **5** | Could discover a *closed-form expression* for albedo as function of physical variables. If successful, would be a major contribution — an interpretable "albedo equation" rather than a black-box FFNN. Directly supports theory development. |
| SINDy | Discover governing equations of albedo dynamics | **4** | **5** | Discovers dynamical equations (dA/dt = f(A, clouds, aerosols, ...)). Could reveal the feedback structure that maintains stability. Entirely different from existing static regression. |
| Neural ODE | Learn continuous-time albedo dynamics | **3** | **4** | Learns differential equations from data. More flexible than SINDy but less interpretable. |

### Regime and Mixture Methods
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Hidden Markov models | Latent albedo regimes | **4** | **4** | Tests whether albedo switches between discrete states (e.g., ENSO-modulated regimes). Could explain apparent stability as regime persistence. |
| Regime-switching regression | Do driver-albedo relationships change by state? | **4** | **4** | Tests whether SHAP-derived importances are state-dependent. If cloud buffering only operates in certain regimes, this would find it. |
| Gaussian mixture models on FFNN residuals | Identify distinct error modes | **2** | **2** | Incremental diagnostic; lower priority unless residuals show clear multimodality. |

### Optimal Transport and Distribution Methods
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Wasserstein distance (NH vs SH distributions) | Quantify hemispheric symmetry precisely | **4** | **5** | Goes beyond comparing means: tests whether the *entire distributions* of NH and SH albedo match. Provides a richer symmetry metric than mean difference. |
| Earth mover's distance over time | Track distributional shifts | **3** | **4** | Monitors whether the symmetry is evolving at the distributional level, not just in means. |
| Distributional regression | Predict full albedo distribution | **2** | **3** | Interesting but likely overkill for the symmetry question. Lower priority. |

### Dimensionality Reduction Beyond PCA
| Method | Tests | Priority | Differentiation | Rationale |
|---|---|---|---|---|
| Independent Component Analysis | Separate independent sources of albedo variability | **4** | **4** | Unlike PCA (orthogonal components), ICA finds *statistically independent* sources. Could isolate cloud buffering, surface albedo, and aerosol signals as independent components. |
| Diffusion maps | Intrinsic dimensionality of albedo manifold | **3** | **5** | Answers: how many degrees of freedom does the albedo system really have? Low dimensionality supports H002 (invariant properties constrain the system). |
| UMAP/t-SNE on spatial patterns | Albedo regime clusters | **2** | **2** | Exploratory visualization; useful early in the process but unlikely to produce a publishable finding on its own. |

### Priority Summary

**Tier 1 (Priority 5, explore first):**
- Convergent cross mapping (causal test of cloud buffering)
- Transfer entropy NH↔SH (teleconnection information flow)
- Predictive information decomposition (beyond SHAP)
- Attractor reconstruction (dynamical stability)
- Wavelet coherence NH-SH (time-frequency teleconnections)
- Climate networks (teleconnection topology)
- Bayesian model comparison (hypothesis competition)
- Symbolic regression (interpretable albedo equation)
- Instrumental variables / volcanic natural experiment

**Tier 2 (Priority 4, explore after Tier 1 yields leads):**
- Lyapunov exponents, SINDy, Bayesian hierarchical models, ICA, Hidden Markov models, Regime-switching regression, Singular spectrum analysis, Community detection, Network centrality, Wasserstein distance, Synthetic control, DoWhy causal DAGs

**Tier 3 (Priority 2-3, explore opportunistically):**
- Granger causality, mutual information, complexity measures, cross-spectral analysis, GPs, change-point detection, neural ODE, GMM on residuals, distributional regression, UMAP/t-SNE, GNNs

The agent is **not limited to this list** and should discover and propose additional methods from literature.

---

## 4. Exploration Strategy: (Model × Hypothesis × Method) Space

The agent explores a **combinatorial space**:

```
For each HYPOTHESIS (Cloud Buffering, Invariant Props, Teleconnections, Emergent):
  For each METHOD FAMILY (causal, info-theoretic, dynamical, spectral, ...):
    For each SPECIFIC METHOD in family:
      1. Is this method applicable to this hypothesis? (quick feasibility check)
      2. Has it been tried? (check prior work summary)
      3. Execute the analysis
      4. Validate the finding (statistical rigor, ScholarEval)
      5. Record evidence for/against hypothesis
      6. If surprising → spawn new hypothesis or follow-up
```

The agent should **not** exhaustively enumerate this space. Instead, it should use **adaptive exploration**:
- Start broad (sample methods across families)
- Go deep on promising leads
- Periodically return to unexplored families
- Use literature search to find domain-specific precedents for each method
- Cross-pollinate: if a method works well for one hypothesis, try it on others

---

## 5. World Model Schema

```
world_model/
  meta.json                       # Cycle count, exploration state
  prior_work.yaml                 # What's been tried (seeded from notebooks)
  hypotheses.json                 # Hypothesis status + convergence
  methodology_tracker.json        # Which methods tried for which hypotheses
  findings/
    cycle_*/task_*/findings.json   # Individual findings with full statistics
  analyses/
    A001_*/
      script.py                   # The exact code run
      output.json                 # Structured results
      figures/
  literature/
    papers.json
  convergence.json                # Per-hypothesis convergence metrics
  exploration_map.json            # The (method × hypothesis) space with coverage
  cycle_summaries/
    cycle_*_summary.json
```

### Exploration Map (`exploration_map.json`)
Tracks which cells in the (method × hypothesis) matrix have been explored:

```json
{
  "methods_explored": {
    "granger_causality": {
      "H001_cloud_buffering": {"status": "completed", "finding_id": "E042", "result": "supported"},
      "H002_invariant": {"status": "not_started"},
      "H003_teleconnections": {"status": "in_progress"}
    },
    "transfer_entropy": {
      "H001_cloud_buffering": {"status": "not_started"},
      ...
    }
  },
  "coverage": 0.23,  // fraction of space explored
  "frontier": ["wavelet_coherence:H003", "symbolic_regression:H002"]  // highest priority unexplored
}
```

---

## 6. Convergence (unchanged from prior revision)

A hypothesis is resolved when:

**Positive**: p < 0.01, meaningful effect size, robust across ≥2 independent methods, ≥2 data subsets, no unresolved contradictions, adversarial testing attempted.

**Null**: >80% power, CI includes zero and excludes minimum meaningful effect, TOST confirms, consistent across methods and subsets.

**The system runs until all hypotheses resolve or it determines it cannot resolve them with available data.**

---

## 7. Implementation Plan

### Step 1: Create world model scaffold + prior work summary
- `world_model/` directory with all schema files
- `prior_work.yaml` summarizing existing notebook analyses
- `exploration_map.json` initialized with method × hypothesis matrix (mostly "not_started")
- Seed H001-H003 with initial predictions

### Step 2: Create shared utilities
- Data loading (CERES, MERRA-2, zone weights, FFNN model)
- World model read/write helpers
- Finding/Hypothesis serialization
- Statistical test wrappers

### Step 3: Create orchestration engine (`run_cycle.py`)
Each cycle:
1. Read world model state + exploration map
2. Select highest-priority unexplored (method × hypothesis) cell
3. Feasibility check: is this method applicable?
4. If yes: write analysis script, execute, validate, record finding
5. If no: mark as "not_applicable", move to next cell
6. Perform targeted literature search for the method being tested
7. Update exploration map + convergence state
8. Compress context for next cycle
9. Loop

### Step 4: Seed with existing results
Load findings from existing notebooks so they're in the world model as baseline evidence.

### Step 5: Run continuously
The agent explores the (method × hypothesis × model) space freely, following leads, proposing new methods from literature, generating emergent hypotheses, running until convergence.

---

## 8. Literature Knowledge Base (Input to Agent)

The agent is seeded with a structured knowledge base of the albedo stability/symmetry literature. This is **not** a static list — the agent actively searches for new papers, extracts findings, and updates this base during each cycle.

### Core Literature Corpus

The following papers and their key findings form the foundational knowledge:

**Albedo Stability and Phenomenology:**
- Stephens et al. (2015) "The Albedo of Earth" — Global/annual albedo ≈29%; comprehensive review of observational record
- Loeb et al. (2018) CERES EBAF TOA Ed4.0 — Definitive satellite albedo product; uncertainty characterization
- Datseris & Stevens (2021) "Earth's albedo and its symmetry" — Analysis of symmetry phenomenon and its persistence
- Wu et al. (2023) — Long-term albedo trends from lunar observatory
- Hanson & Kharecha (2025) — Large cloud feedback confirms high climate sensitivity

**Hemispheric Symmetry:**
- Voigt et al. (2013) — Observed hemispheric symmetry in reflected SW despite different surface compositions
- Rugenstein & Hakuba (2021) — Time evolution of hemispheric albedo asymmetry
- Jönsson & Bender (2023) — Persistence and variability of interhemispheric albedo symmetry
- Bender et al. (2006, 2011) — GCM representation of planetary albedo; 22 models compared

**Cloud-Radiation Interactions:**
- Ramanathan et al. (1989) — Cloud-radiative forcing; ERBE results
- Stephens (2005) — Cloud feedbacks in the climate system critical review
- Baker (1997) — Cloud microphysics and climate
- Klein & Hall (2015) — Emergent constraints for cloud feedbacks
- Klein et al. (2018) — Low-cloud feedbacks from cloud-controlling factors
- Kemsley et al. (2024) — Systematic evaluation of high-cloud controlling factors
- Gristey & Feingold (2025) — SAI would change cloud brightness

**Volcanic Perturbations (Natural Experiments):**
- Minnis et al. (1993) — Pinatubo radiative climate forcing
- Stenchikov et al. (1998) — Pinatubo radiative forcing quantification
- Ramachandran et al. (2000) — Pinatubo lower stratospheric response
- Khaykin et al. — Stratospheric aerosol evolution from lidar/satellite

**Aerosol-Albedo Coupling:**
- Hodnebrog et al. (2024) — Recent aerosol emission reductions increased Earth's energy imbalance
- MATCH aerosol optical depth products in CERES EBAF

**Earth Energy Balance:**
- Hansen et al. (2005) — Earth's energy imbalance confirmation
- Hakuba et al. (2024) — Trends in energy imbalance and ocean heat uptake
- Allan et al. (2014) — Changes in global net radiative imbalance 1985-2012
- Trenberth et al. (2016) — Earth's energy imbalance from multiple sources

**Model Evaluation:**
- Jian et al. (2020) — CMIP6 planetary albedo evaluation against satellite obs
- Eyring et al. (2016) — CMIP6 experimental design
- Golaz et al. (2013) — Cloud tuning impact on 20th century warming
- Hourdin et al. (2017) — Art and science of climate model tuning
- Schmidt et al. (2017) — Practice of climate model tuning across US centers
- Stephens et al. (2023) — Changing nature of Earth system constraints on albedo

**Spatial and Temporal Scales:**
- Feldman et al. (2021) — Sub-diurnal to interannual frequency analysis of reflected SW
- Hakuba et al. (2016) — Zonal near-constancy of fractional solar absorption

**AI/ML in Climate:**
- Jones (2017) — ML for climate forecast improvement
- van Straaten et al. (2022) — Explainable ML for subseasonal temperature drivers

### Literature-Derived Hypotheses and Open Questions

The agent should be aware of these **open questions from the literature** that remain unresolved:

1. **Why ≈29%?** — The theory does not need to explain the absolute value, but this remains a fundamental open question
2. **Will symmetry persist under forcing?** — CMIP6 models project a wide range; observations needed
3. **Cloud feedback sign and magnitude** — Still the largest uncertainty in climate sensitivity
4. **Aerosol reductions breaking symmetry?** — Hodnebrog et al. suggest recent aerosol declines affect energy balance
5. **Role of marine stratocumulus** — Low cloud decks may be disproportionately important for albedo stability
6. **Volcanic eruption as falsification test** — Next major eruption will test both hypotheses in real-time
7. **Tipping points** — Are there albedo thresholds that, once crossed, cannot recover?
8. **Libera continuity** — Will next-generation observations reveal unprecedented changes?

### How the Agent Uses Literature

During each cycle, the agent should:
1. **Search for relevant papers** when designing a new experiment (find methodological precedents)
2. **Compare its findings against published values** (does my SHAP result agree with known cloud forcing estimates?)
3. **Identify contradictions** between its results and literature (potential novel findings)
4. **Extract quantitative benchmarks** from papers (effect sizes, uncertainties, trends to validate against)
5. **Discover new hypotheses** from literature that haven't been tested with this dataset
6. **Track evolving consensus** on key questions (cloud feedback, aerosol forcing, symmetry persistence)

The literature base is **append-only** — new papers found during the run are added, never removed. Each literature entry includes: citation, key quantitative findings, relevance to specific hypotheses, and consistency with the agent's own analysis results.

---

## 9. Key Files (Codebase)

| File | Role |
|---|---|
| `Albedo Machine Learning with no TS model.ipynb` | FFNN architecture, existing SHAP/perturbation (prior work) |
| `FFNN Albedo Perturbations.ipynb` | Existing virtual lab framework (prior work) |
| `Additive Impacts & Kernel Time Varying Regression.ipynb` | Existing Prophet/temporal analysis (prior work) |
| `Cleaned Up EDA.ipynb` | Existing MERRA-2 cross-validation (prior work) |
| `albedo_ffnn_5layer_relu_v0_66_*.pth` | Trained FFNN checkpoint (reusable) |
| `zone_weights_lou.txt` | Latitude area weights |
| `../CERES_2022-11-09_25176/ceres.parquet.gzip` | Primary CERES data |

---

## 9. Quality Control

- **ScholarEval scoring**: rigor (0.25), impact (0.20), novelty (0.15), reproducibility (0.15), clarity (0.10), coherence (0.10), limitations (0.03), ethics (0.02). Threshold: 0.75.
- **Multiple method corroboration**: No hypothesis resolved by single method
- **Adversarial testing**: Must attempt to falsify before confirming
- **Null results valued**: Documented with power analysis
- **Reproducibility**: Every finding links to exact script
- **Human-in-the-loop**: Asynchronous review via cycle summaries; can inject guidance anytime
