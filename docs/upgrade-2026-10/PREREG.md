# Pre-registered evaluation plan: S&P 500 Regime Monitor, model v2

**Frozen on 2026-10-12, before any 2019+ result exists for candidates C1-C5.**

Commit this file as `wf1/prereg/PREREG.md`. `decision.json` must record the SHA-256 of this file, of `recipes.json` and of every evaluation script, and it must be written before the holdout loader can run.

---

## 0. Ground rules

1. **Holdout.** The holdout is 2019-01-02 onward. Only `run_holdout.py` may load it, and only after `decision.json` names the single selected candidate c*.
   - It runs once.
   - If any recipe, metric or threshold code changes after holdout metrics exist, the holdout is void and the live model stays as it is.
   - The only fixes allowed after the run starts are fixes unrelated to outcomes (paths, dtypes, crashes). Log each one with its diff in `holdout_log.txt`.
2. **Development data.** Development uses data up to 2018-12-31 only. The loader hard-truncates at that date, as `evalkit.load()` does.
3. **The six regimes are fixed.** Their names, order, colours and anchor definitions do not change: Bull/Sideways/Bear by the sign of the 200-day trend and the 1-month return; Quiet/Volatile by VIX against the training median.
4. **Known contamination.** This is declared here, not fixable:
   - (a) Every proposer saw the baseline's 2019-2026 diagnostics (choppiness, over-confidence, Bear Quiet). Those diagnostics motivated tempering and the label discussion.
   - (b) The data audit's linear screen used a 2013-2026 test window, so credit (C5) was partly suggested by holdout-era data. C5 therefore needs a stricter margin (section 8).
   - No hyperparameter (τ, clip, bridge, feature set) was chosen with 2019+ data. The z_1d idea came from a proposal that loaded only data up to 2018.
5. **Proposal numbers are indicative only.** The proposals' own figures do not count. Selection uses one unified re-run of B0 and C1-C5 under sections 4-8, with one code base.

---

## 1. Judge's assessment of the three proposals

Scores are out of 5. For implementation risk, 5 means the lowest risk.

| Proposal | Accuracy gain | Robustness | Honest evaluation | Label preservation | Live feasibility | Implementation risk | Total |
|---|---|---|---|---|---|---|---|
| Data-first | 3 | 2 | 3 | 4 | 4 | 4 | 20 |
| Structure-first | 5 | 4 | 4 | 4 | 5 | 4 | 26 |
| Evaluation-first | 2 | 3 | 5 | 5 | 5 | 5 | 25 |

### Key rulings

**VIX-scaled return (z_1d, structure-first).** This is the largest and most consistent gain:
- next-day log score 3.2827 vs 3.2102 over the 2001-18 walk-forward, ahead in both halves (2001-10 +0.098, 2011-18 +0.041);
- beats walk-forward GARCH(1,1)-t (3.2748);
- vol Spearman 0.766 vs 0.677;
- 8.3-8.5 switches a year vs 12.3.

I read `feat.py` and `evalwf.py`, and the claims hold:
- the scale uses the *previous* day's VIX, so the feature is causal;
- the predictive density is N(σ_t·m_k, σ_t²·v_k), evaluated at the raw r_{t+1}, so the Jacobian is included;
- GARCH-t is refit on the same expanding windows.

Weak spots:
- QLIKE in 2011-18 is 4.6% worse (0.364 vs 0.348);
- one more Bear-Volatile miss (2 vs 1).

**Tempering at τ = 0.5.** Two proposals picked it independently on development data. It roughly halves flip-flops and cuts the share of days at "99%+ sure" from 74% to about 58%, at little cost to the scores. It is adopted in every candidate.

**1986 VXO-bridged history.**
- Positive in data-first (+26 mnats/day, CI -7 to +74) and in evaluation-first (A vs B).
- Neutral on the z model (+0.0008).
- Standardisation and the VIX median must be computed on 1990+ rows. Without that, honesty against hindsight was significantly worse (evaluation-first, A5).
- It is kept as a tested option (C2, C4).

**dd_252 swap (data audit).** Significantly worse in two independent HMM tests. Rejected.

**dd_63 swap (data-first).** Its gain is almost entirely the 1990-start fix, and it makes 2011-18 vol ranking worse. Rejected; tempering handles the overlap problem instead.

**Credit (BAA10Y change).** The evidence conflicts:
- data-first: better identification AUCs but no density gain;
- structure-first: +0.003 to +0.004 nats/day but slower Bear calls (15-16 vs 10 days);
- evaluation-first: neutral to worse.

It is kept as one gated candidate (C5), because Max asked for more data and it is the only external series that survived the audit with full history and a clean lag.

**Student-t emissions.** No gain after VIX scaling, significantly worse on the current features, and 10x the fit time. Rejected.

**Primary metric choice.** The primary is the model's own one-step predictive log score, which is proper and needs no extra fitted pieces. The regime-head scores (evaluation-first) become guards. They make sure a candidate cannot win on the VIX-scaled density alone while its regime probabilities get less informative.

---

## 2. Data

All data comes from the cached files in `scratchpad/data`; nothing is downloaded during evaluation.

### Series

| Series | Source | Coverage | Lag |
|---|---|---|---|
| ^GSPC close (C) | `yahoo/_GSPC.csv` | 1985-01-02 → | 0: same-day close, or the live partial bar |
| ^VIX close (V) | `yahoo/_VIX.csv` | 1990-01-02 → | 0 |
| Bridged VIX | `wf1/data-auditor/out/vix_spliced_1986.csv`, column `vix_spliced` | 1986-01-02 to 1989-12-29 (1986-05-15 missing) | frozen file |
| BAA10Y (C5 only) | `fred/BAA10Y.csv` | 1986-01-02 → | `availability.align(series, spx_index, RULES['BAA10Y'])` = `fred_next_bday`: usable from the first S&P trading day after its next-Fed-business-day release, about d+2 |

Further rules:
- The VIX close is joined onto S&P dates and never forward-filled.
- The bridged VIX is exp(0.0610 + 0.9784·ln VXOCLS). It is never refitted.

### Calendar

- The S&P 500 trading calendar is the master.
- All S&P features are computed on the full S&P calendar from 1985-01-02 *before* VIX is joined.
- Rows with any missing input are dropped. The filter treats the next row as consecutive, as today.

### Feature definitions

All logs are natural.

| Feature | Definition |
|---|---|
| ret_1d_t | ln C_t − ln C_{t−1} |
| ret_1w_t | ln C_t − ln C_{t−5} |
| ret_1m_t | ln C_t − ln C_{t−21} |
| ret_1y_t | ln C_t − ln C_{t−252} |
| trend_mom_t | ln C_t − ln(mean of C_{t−199..t}) |
| log_vix_t | ln V_t |
| log_rv21_t | ln(√252 · sd(ret_1d_{t−20..t}, ddof=1) + 1e−8) |
| σ_t | V_t / 100 / √252 |
| z_1d_t | ret_1d_t / σ_{t−1}, where σ_{t−1} uses the previous S&P day's VIX close |
| baa_chg21_t | B*_t − B*_{t−21 rows}, in percentage points, where B*_t is the aligned BAA10Y value on row t (C5 only) |

---

## 3. Recipes

### Common to B0 and all candidates (identical to the live `fit_hmm`)

- **States and emissions:** 6 states with full-covariance Gaussian emissions, using hmmlearn 0.3.3 `GaussianHMM` with:
  - n_iter = 1000, tol = 1e−5, min_covar = 1e−4;
  - transmat_prior = startprob_prior = 1 + 1e−3;
  - init_params = "" and params = "stmc".
- **Starting values:** state means and covariances are initialised at the anchor buckets, with the transition matrix at diagonal 0.95 and 0.01 elsewhere, then row-normalised.
- **Mean prior:** MAP prior on the means, with means_prior = the anchor bucket means and means_weight = 3000.
- **Seeds:** 8 seeds (0-7). Seed 0 is the pure anchored start; seeds 1-7 add 0.15·N(0,1) jitter to the means. Keep the fit with the best training log-likelihood.
- **Anchors** (training rows only):
  - Bull = trend_mom > 0 and ret_1m > 0;
  - Bear = both < 0;
  - Sideways = otherwise;
  - Volatile = log_vix > the training median.
- State k is always REGIME_ORDER[k].
- **Refit gate:** every refit must pass the existing `label_states()` rank check plus these sign checks:
  - Bull means have trend_mom > 0 and ret_1m > 0; Bear means have both < 0;
  - each Volatile state's log_vix mean is above its Quiet twin's.

### Recipe differences

τ is the emission tempering in the **filter only**. Smoothing for hindsight targets always uses τ = 1.

| ID | Inputs (dimensions) | History / first row | Scaling and VIX-median rows | Clip (standardised) | τ |
|---|---|---|---|---|---|
| **B0** baseline | ret_1d, ret_1w, ret_1m, ret_1y, trend_mom, log_vix, log_rv21 (7) | Legacy: ^GSPC and ^VIX from 1990-01-01, inner-joined *before* features; first row 1990-12-31 | Its own training rows | none | 1 |
| **C1** Steady | same 7 as B0 | Features on the S&P calendar, then join VIX; first row 1990-01-02 | Training rows | ±5 | 0.5 |
| **C2** Longer memory | same 7 as B0 | C1 plus the bridged VIX for 1986-89; first row is the first with all inputs (about 1986-01-03); 1986-05-15 dropped | Training rows **from 1990-01-02 only** | ±5 | 0.5 |
| **C3** VIX-scaled | z_1d replaces ret_1d (7) | First row 1990-01-03 | Training rows | ±5 | 0.5 |
| **C4** VIX-scaled + longer memory | as C3 (7) | C3 plus the bridge; about 1986-01-03; 1986-05-15 and 05-16 dropped | Rows from 1990-01-03 only | ±5 | 0.5 |
| **C5** C4 + credit | C4 + baa_chg21 (8) | About 1986-02-04 | Rows from 1990-01-03 only | ±5 | 0.5 |

Rules that apply to every recipe:

**Clipping.** Clip after standardising, in both fitting and filtering. Scoring always uses **raw, unclipped** r_{t+1}.

**Tempered filter.** α̃_t ∝ (α̃_{t−1}·A) ⊙ exp(τ·log b(y_t)), followed by the predicted state P(s_{t+1} | data to t) = α̃_t·A.

**Return marginal for state k** (de-standardised):
- B0, C1, C2: m_k = μ_k[ret]·sd + mean and v_k = Σ_k[ret,ret]·sd².
- C3-C5: m_{k,t} = σ_t·(μ_k[z]·sd + mean) and v_{k,t} = σ_t²·Σ_k[z,z]·sd².
- State means are used as fitted. No flat-drift substitution.

**C5 staleness rule.** If Baa age exceeds 10 calendar days, score each state with its marginal Gaussian without the baa dimension. This never triggers historically (maximum age is 6 days); the same code path runs in backtest and live.

---

## 4. Selection walk-forward (development)

- Expanding window with refits every 1 January Y, for Y = 2001…2018: 18 fits per recipe.
- Fit Y trains on the recipe's first row through (Y−1)-12-31.
- For every day of year Y, the filter runs from the recipe's first row with fit Y's parameters, exactly as the live job does.
- Scored forecast origins run from 2001-01-02 to 2018-12-28 (targets through 2018-12-31). All recipes are scored on the identical set of days.
- Report the full window plus 2001-2010 and 2011-2018, and year-by-year win counts.
- Context benchmarks on the same windows (never decisive):
  - GARCH(1,1)-t;
  - climatology, i.e. the unconditional Student-t of the training window;
  - the no-HMM anchor rulebook;
  - the saved static 1990-2022 calibration.

---

## 5. Primary metric (single, pre-declared)

**LS1: next-day predictive log score, in nats/day.**

LS1 = mean over scored origins t of log Σ_k P(s_{t+1}=k | data to t) · N(r_{t+1}; m_{k,t}, v_{k,t})

- r_{t+1} is the raw close-to-close S&P 500 log return.
- The score is proper.
- It is comparable across feature sets because every recipe gives a density for the same observable, in the same units, with the Jacobian included.

**Paired inference.**
- Use daily differences d_t = LS1_cand,t − LS1_B0,t.
- Bootstrap with a circular moving-block bootstrap: 63-day blocks, 5,000 resamples, `numpy.random.default_rng(20261012)`.
- Report the mean, the two-sided 95% CI, and one-sided lower bounds at 90% and 95%.

---

## 6. Secondary metrics

All are computed on the same days with the same code for every recipe.

1. **21-day forward variance.**
   - h_t = (1/21) Σ_{j=0..20} Σ_k [α̃_t A^{j+1}]_k · (v_{k,t} + m_{k,t}²), with σ_t held fixed at the origin for z recipes.
   - y_t = mean of r² over t+1..t+21.
   - **QLIKE21** = mean(y/h − ln(y/h) − 1).
   - **Spearman21** = Spearman correlation of √h with √(252·y).
2. **Dip Brier.** D_t = 1 if min_{j=1..21} C_{t+j}/C_t − 1 ≤ −5%.
   - p_t = Σ_k F_{t,k}·q_k, where F is the tempered filtered probability.
   - q_k = (Σ F·D + 63·q̄) / (Σ F + 63), fitted on fit Y's training rows using only rows with t+21 ≤ train_end ("regime heads").
3. **Heads log score** (regime informativeness): per-regime Student-t heads for r_{t+1}, fitted by weighted ML on training rows, with a pooled fallback when total weight is below 250. Mixed with P(s_{t+1}). Use `evalkit.fit_t_weighted`.
4. **Honesty against a frozen hindsight rulebook** (`evalkit.truth`, vol median frozen on 1990-2018):
   - 6-way regime Brier, direction Brier and Volatile Brier;
   - accuracy and a reliability table;
   - the paired bootstrap as in section 5.
5. **Calibration.**
   - ECE of the filtered pmax against agreement with the model's own untempered smoothed path (bins 0, .5, .7, .9, .99, 1). Report it raw and after the isotonic map (section 9).
   - Share of days with pmax > 0.99.
6. **Stability** (`diagnostics.switching` on the filtered argmax): switches per year, flip-flops per year (A→B→A with B lasting under 5 days), share of spells under 5 days, and median spell.
7. **Timeliness** (`diagnostics.correction_episodes(close, 0.10, 0.10)` with peaks inside the window; `drawdown_lags` and `lag_summary` unchanged):
   - median trading days from the peak to the first any-Bear call and to the first Bear Volatile call;
   - misses;
   - lag relative to the −10% crossing.
8. **Label sanity** (`diagnostics.label_sanity`; a comparison counts only when both sides have at least 60 days):
   - forward 21-day vol: each Volatile regime above its Quiet twin, and Bear Volatile the highest;
   - P(dip) orderings;
   - forward 21-day returns are reported only;
   - regime shares and 6-way argmax agreement with B0.
9. **Tail coverage:** share of r_{t+1} below the 1% and above the 99% predictive quantiles.
10. **Context:** the gap to GARCH-t, climatology and the rulebook.

---

## 7. Development eligibility guards (vs B0, 2001-2018)

A candidate is eligible only if it passes all of G1-G6.

**G1 Labels.**
- Rank and sign checks pass on 18 of 18 fits.
- Zero forward-vol ordering failures.
- 6-way argmax agreement with B0 is at least 80%.
- Each regime's share is within ±5 percentage points of B0's.

**G2 Forward risk.**
- QLIKE21 ≤ 1.05 × B0.
- Spearman21 ≥ B0 − 0.01.
- Dip Brier is not significantly worse. Test: the 90th percentile of the bootstrap mean of (Brier_B0 − Brier_cand) must be ≥ 0.

**G3 Honesty.** The 6-way hindsight-rulebook Brier is not significantly worse than B0's (one-sided 90%).

**G4 Stability.** Switches per year, flip-flops per year and the share of spells under 5 days are each ≤ B0's.

**G5 Timeliness.**
- Median any-Bear lag ≤ B0 + 3 trading days.
- Any-Bear misses ≤ B0's.

**G6 Primary.** Mean ΔLS1 vs B0 > 0.

---

## 8. Selection rule (deterministic)

1. E = the set of eligible candidates. If E is empty, nothing goes to the holdout: the live model stays, and only the display fixes in section 11 ship.
2. c* = argmax of LS1 over E.
3. **Parsimony walk-down.** While c* has an eligible parent p with LS1(c*) − LS1(p) below the margin for the component that p removes, set c* ← p. Check parents in the order listed. Margins, in nats/day:

   | Step | Component removed | Margin |
   |---|---|---|
   | C5 → C4 | credit (live FRED dependency) | +0.003 **and** a one-sided 90% lower bound > 0 |
   | C4 → C3, then C4 → C2 | bridge / z_1d | +0.002 / +0.005 |
   | C2 → C1 | bridge | +0.002 |
   | C3 → C1 | z_1d | +0.005 |

4. Write `decision.json` with c*, every development metric for B0 and C1-C5, and the hashes listed in section 0.

---

## 9. Holdout run (once)

- Period: 2019-01-02 to the last complete trading day in the cache at run time (expected about 2026-10-09). Record that date in `decision.json`; the last target must exist.
- Recipes: **c* and B0 only.**
- Same annual walk-forward: fits Y = 2019…2026, each trained through (Y−1)-12-31, with the filter running from the first row.
- **Isotonic confidence map** (display, for c*):
  - Fit `sklearn IsotonicRegression(increasing=True, out_of_bounds='clip', y_min=0, y_max=1)` on development pairs: pmax_t against 1[filtered argmax = own smoothed argmax], over 2001-2018. Freeze it before the holdout.
  - Report calibrated ECE on the holdout.
- After the verdict is written, context-only benchmarks may also be computed: GARCH-t, climatology, the static saved calibration, and the 2023-latest sub-window.

---

## 10. Pass bar (holdout, c* vs B0)

All of H1-H7 must hold.

**H1 Primary.**
- Mean ΔLS1 > 0.
- The **one-sided 95%** block-bootstrap lower bound (5th percentile) is > 0.

**H2 Forward risk.**
- QLIKE21 ≤ 1.05 × B0.
- Spearman21 ≥ B0 − 0.02.
- Dip Brier is not significantly worse (one-sided 90%).

**H3 Labels.**
- Rank and sign checks pass on 8 of 8 fits.
- Zero forward-vol ordering failures.
- Dip-ordering failures ≤ B0's count.
- 6-way agreement with B0 ≥ 75%.

**H4 Honesty.** The hindsight-rulebook 6-way Brier is not significantly worse than B0's (one-sided 90%).

**H5 Stability.**
- Switches per year ≤ B0 + 1.0.
- Flip-flops per year ≤ B0 + 0.5.

**H6 Timeliness.**
- Median any-Bear lag ≤ B0 + 5 trading days.
- Any-Bear misses ≤ B0's.

**H7 Operations.**
- A dry-run of the 15-minute job on Actions takes ≤ 2 minutes.
- The backtest/live feature-parity test passes: maximum absolute difference ≤ 1e−12 on the last 300 complete rows.

**Calibrated ECE** is reported, not gated. If it is above 0.05, the map is refit on 2001-2025 before shipping; this changes the display only.

---

## 11. What ships

**PASS:** c* ships under the deployment plan.

**FAIL, or E empty:**
- The live model stays exactly as it is: same recipe and the saved 1990-2022 calibration.
- No other candidate gets a holdout attempt; that would be holdout shopping.
- Only model-independent display fixes ship:
  - verbal confidence calibrated from B0's walk-forward reliability;
  - per-regime statistics from walk-forward out-of-sample days only, with day counts;
  - "predicted tomorrow" removed (it equals naive persistence);
  - "Bear describes what the market has been doing, not a forecast".
- c* then runs as a shadow model and ships only if it passes the full H1-H7 bar on 12 prospective months.

---

## 12. Expected outcome (stated in advance, non-binding)

From the proposals' development evidence:

- **C3 is the likely c*.**
- C4 is expected to step down to C3: its LS1 is within 0.002.
- C5 is expected to fail G5: any-Bear lag of about 15 days vs 10.

If that happens, "more data" reaches Max through three routes:
1. 1990 history (+250 rows, including the 1990 bear market);
2. annual refits, which add every new year (the live fit stops at 2022);
3. VIX information in the return dimension.

The 1986 and credit options will have been tested fairly and declined on the evidence, and the page should say so plainly.

---

## 13. Deliverables (in `wf1/prereg/`)

- `PREREG.md` (this file) and `recipes.json`.
- `run_dev.py`, which reuses:
  - `hmm_regime.fit_hmm` for fitting;
  - `shmm.forward_tempered` / `smooth`;
  - from `evalkit`: `truth`, `regime_brier`, `fit_t_weighted`;
  - from `diagnostics`: `switching`, `correction_episodes`, `drawdown_lags`, `lag_summary`, `label_sanity`, `calibration_vs_hindsight`;
  - `availability.align`.
- `select.py` produces `decision.json`.
- `run_holdout.py` produces `holdout_results.json` and `verdict.json`.

Compute budget: about 108 development fits on 19 workers in under 15 minutes, and 16 holdout fits.