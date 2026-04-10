# Cycle 3 Decision: Agent Rollout Path Selection

## Current State After Cycle 2

| Hypothesis | Confirming | Refuting | Confidence | Status |
|---|---|---|---|---|
| H001 Cloud Buffering | 1 (FFNN perturbation, prior) | 0 | 0.6 | Needs causal evidence |
| H002 Invariant Properties | 0 | 0 | 0.4 | UNTOUCHED — biggest gap |
| H003 Teleconnections | 2 (TE + wavelet) | 1 (U-Net null) | 0.7 | One method + one data subset away |

**Cycle 2 insight**: ENSO-band periods dominate NH-SH coupling. SH leads NH at interannual timescales.

## Candidate Rollout Paths

### Path 1: Climate Networks × H003 (convergence push)
Third independent method → moves H003 to 3/3 confirming. Maps spatial topology.
- Information gain: **3** — Confirms rather than discovers
- Proximity to convergence: **5** — Would satisfy method requirement
- Novelty follow-up: **3** — Related but different angle from cycles 1-2
- **Score: 0.4(3) + 0.3(5) + 0.3(3) = 3.6**

### Path 2: Attractor Reconstruction × H002 (fill the gap)
H002 has ZERO evidence. Takens embedding tests if albedo is dynamically constrained.
- Information gain: **5** — Maximum; completely untested hypothesis
- Proximity to convergence: **2** — First of 3 needed
- Novelty follow-up: **1** — Unrelated to ENSO finding
- **Score: 0.4(5) + 0.3(2) + 0.3(1) = 2.9**

### Path 3: CCM × H001 (causal test)
Convergent cross mapping tests if cloud→albedo is causal, not just correlational.
- Information gain: **4** — Causal vs correlational is a qualitative leap
- Proximity to convergence: **3** — Second method for H001
- Novelty follow-up: **2** — Could connect to ENSO finding (ENSO→clouds→albedo)
- **Score: 0.4(4) + 0.3(3) + 0.3(2) = 3.1**

### Path 4: ENSO-conditional transfer entropy (deepen Cycle 1-2)
Split data by ENSO phase, re-run TE. Tests if teleconnection is ENSO-mediated.
- Information gain: **3** — Deepens existing finding
- Proximity to convergence: **3** — Could count as second data subset for H003
- Novelty follow-up: **5** — Directly follows the ENSO-band discovery
- **Score: 0.4(3) + 0.3(3) + 0.3(5) = 3.6**

## Decision

| Path | Score |
|---|---|
| **1. Climate Networks × H003** | **3.6** |
| **4. ENSO-conditional TE** | **3.6** |
| 3. CCM × H001 | 3.1 |
| 2. Attractor × H002 | 2.9 |

**Tie between Paths 1 and 4.** Tiebreaker: Path 4 (ENSO-conditional TE) serves double duty — it follows the novel ENSO lead AND counts as a second data subset for H003 convergence (testing on ENSO-stratified subsets vs. full data). This is more efficient.

**Selected: Path 4 — ENSO-conditional Transfer Entropy**

If the SH→NH coupling disappears during La Niña but persists during El Niño (or vice versa), we've identified the mechanism mediating hemispheric albedo teleconnections.
