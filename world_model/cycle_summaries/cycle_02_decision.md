# Cycle 2 Decision: Agent Rollout Path Selection

## Decision Framework

Each candidate rollout is scored on three criteria (from design doc):
1. **Information gain** (0-5): How much uncertainty does this reduce across all hypotheses?
2. **Proximity to convergence** (0-5): Does this move a hypothesis closer to resolution? (≥3 methods + ≥2 data subsets needed)
3. **Novelty of recent leads** (0-5): Does this follow up on a surprising finding from Cycle 1?

**Score = 0.4 × Information Gain + 0.3 × Proximity + 0.3 × Novelty**

## Current State After Cycle 1

| Hypothesis | Confirming | Refuting | Confidence | Status |
|---|---|---|---|---|
| H001 Cloud Buffering | 1 (FFNN perturbation, prior) | 0 | 0.6 | Needs causal evidence |
| H002 Invariant Properties | 0 | 0 | 0.4 | UNTOUCHED — biggest gap |
| H003 Teleconnections | 1 (transfer entropy) | 1 (U-Net null) | 0.5 | Promising new lead |

**Cycle 1 surprise**: SH→NH information dominance at 3-6 month lags was unexpected. Why does the Southern Hemisphere lead?

---

## Rollout Path 1: Wavelet Coherence × H003

**What**: Time-frequency decomposition of NH-SH coupling. Identifies WHICH timescales carry the information flow discovered in Cycle 1.

- Information gain: **4** — Reveals frequency structure of teleconnections (annual? interannual? ENSO-scale?)
- Proximity to convergence: **4** — Would give H003 a second confirming method (2/3 needed)
- Novelty follow-up: **5** — Directly follows the SH→NH surprise; could reveal if it's ENSO-driven

**Score: 0.4(4) + 0.3(4) + 0.3(5) = 4.3**

## Rollout Path 2: Attractor Reconstruction × H002

**What**: Takens embedding of global albedo time series. Tests whether albedo lives on a low-dimensional attractor (would explain stability from dynamical systems perspective).

- Information gain: **5** — H002 has ZERO evidence; any result is maximally informative
- Proximity to convergence: **2** — Would be first of 3 needed methods; far from resolution
- Novelty follow-up: **1** — Unrelated to Cycle 1 finding

**Score: 0.4(5) + 0.3(2) + 0.3(1) = 2.9**

## Rollout Path 3: Convergent Cross Mapping × H001

**What**: Tests CAUSAL coupling between cloud area fraction and albedo. Goes beyond SHAP correlation to determine if clouds *cause* albedo stability.

- Information gain: **4** — Distinguishes causation from correlation for the leading hypothesis
- Proximity to convergence: **3** — Would give H001 a second method (2/3), and a fundamentally different one (causal vs correlational)
- Novelty follow-up: **2** — Somewhat related (if clouds causally buffer, that could explain hemispheric coupling)

**Score: 0.4(4) + 0.3(3) + 0.3(2) = 3.1**

## Rollout Path 4: Deep Dive on SH→NH Dominance

**What**: Investigate WHY Southern Hemisphere leads at 3-month lag. Decompose by cloud/surface/aerosol components. Check if it's ENSO-driven (tropical Pacific → midlatitude teleconnection). Cross-reference with MERRA-2 if available.

- Information gain: **3** — Deepens one finding rather than broadening
- Proximity to convergence: **3** — Strengthens H003 evidence but same method family
- Novelty follow-up: **5** — Directly exploits the most surprising result

**Score: 0.4(3) + 0.3(3) + 0.3(5) = 3.6**

---

## Decision

| Path | Score | Rationale |
|---|---|---|
| **1. Wavelet Coherence × H003** | **4.3** | Best overall: high information gain, moves H003 toward convergence, directly follows surprise |
| 4. SH→NH Deep Dive | 3.6 | Strong novelty follow-up but doesn't add a new method to the convergence count |
| 3. CCM × H001 | 3.1 | Important but no connection to Cycle 1 findings |
| 2. Attractor × H002 | 2.9 | Maximum information gain but isolated from current momentum |

**Selected: Path 1 — Wavelet Coherence × H003 (Teleconnections)**

This maximizes overall score because it:
- Corroborates Cycle 1 with an independent method (moves H003 to 2/3 confirming)
- Reveals the FREQUENCY structure of the SH→NH information flow
- Could identify if the 3-month lag corresponds to a specific climate mode (ENSO period ~3-7 years)
