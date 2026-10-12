# Interpretation log for PREREG.md (sha256 8ae42ec6…41f2)

PREREG.md is not edited. These are the reading choices the code makes where the plan leaves room, written before any development-grid or holdout result exists.

## Data and features

1. **Loader.** `prlib.load_raw()` truncates every input series at 2018-12-31 before building any feature or target. Reading later data needs a key minted by `unlock_holdout()`. Only `run_holdout.py` can mint it, and only after `verify_decision()` passes. A later cut loads exactly the same dev-era rows: the fingerprint test (cut 2018 then sliced to 2010, against cut 2010) is identical.
2. **First rows**, as computed: B0 1990-12-31, C1 1990-01-02, C2 1986-01-02, C3 1990-01-03, C4 1986-01-03, C5 1986-02-04. C2 starts on 1986-01-02, the first row with all inputs; PREREG says "about 1986-01-03". The C5 Baa age is at most 6 days to 2018, so the staleness rule never triggers.
3. **Common origins.** Development scores the origins present in all six recipes. The holdout scores the origins present in c\* and B0, since only those two run there.
4. **Targets.** r_{t+1}, y21 and the dip flag come from the S&P calendar for every recipe, B0 included. B0 still filters on its legacy inner-joined rows. Origins whose 21-day window is incomplete are left out of QLIKE21, Spearman21 and Dip Brier only.

## Fitting and gates

5. A fit that fails the refit gate is still used in the walk-forward, scored as fitted, and counts as a failure in G1/H3. A fit that raises falls back to the previous year's fit, as `baseline/walkforward.py` does. It is flagged `fallback` and counts as a gate failure.
6. **Gate.** `label_states()` gets the first seven dimensions; C5's baa dimension is not ranked. The sign checks use the de-standardised state means.

## Metrics

7. **Smoothed path.** For ECE and the isotonic pairs, the smoothed path is the untempered smoother with fit Y's parameters over every row up to the evaluation run's end: 2018-12-31 in development, the decision's end date in the holdout. `baseline/walkforward.py` smooths to the end of year Y instead, and `run_dev.py --repro-b0` uses that end to match it.
8. **Regime heads.**
   - Student-t heads for r_{t+1} are weighted by the tempered predicted P(s_{t+1} | data to t). They use training rows with t+1 ≤ train_end, and fall back to the pooled fit when the weight is under 250.
   - Dip heads are weighted by the tempered filtered F. They use rows with t+21 ≤ train_end, and q̄ is the unweighted dip rate on those rows.
9. **"Not significantly worse (one-sided 90%)"** for G2, G3, H2 and H4 means the 90th percentile of the bootstrap mean of (B0 − candidate) is ≥ 0. Each bootstrap call starts a fresh `default_rng(20261012)`.
10. **Timeliness.** Correction episodes count only when their peak falls between the first and last labelled origin of the window. Otherwise, a sub-window or a smoke slice would count episodes it never labelled as misses. `drawdown_lags` and `lag_summary` are unchanged.
    - If neither side has a median any-Bear lag, the lag test passes.
    - If only the candidate lacks one, it fails.
    - If only B0 lacks one, the candidate passes.
    - Misses are always compared separately.
11. **Label checks.** "Bear Volatile the highest" is tested pairwise against the other five regimes. The P(dip) orderings tested are Bear ≥ Bull at each volatility level and Volatile ≥ Quiet within each direction. Every comparison needs at least 60 days on both sides.
12. **Context benchmarks.**
    - GARCH(1,1)-t and climatology are refit each year on returns from 1990-01-02 through Y−1.
    - The rulebook gives each anchor label (C1 features, training VIX median) a Student-t head, with a pooled fallback under 250 days.
    - The static benchmark is the repo's `model/hmm_params.json`, read only. It is in-sample through 2022.

## Selection (select.py)

13. **Walk-down.** A step down happens when ΔLS1 < margin; equality keeps the child. C5 is kept only if ΔLS1 ≥ 0.003 **and** the paired lb90 > 0. For C4, C3 is checked before C2.
14. **What decision.json hashes:**
    - PREREG.md and recipes.json;
    - prlib.py, run_dev.py, select.py, run_holdout.py;
    - baseline/hmm_regime_copy.py, diagnostics.py, harness.py; shmm.py, evalkit.py, availability.py;
    - the bridged-VIX CSV;
    - dev_results.json and every dev_daily and dev_fits file.

    `run_dev.py` records the script hashes it ran with, and `select.py` refuses if they have changed since.
15. **Holdout end date.** This is the last complete ^GSPC row at `select.py` time, read from the date column only. A row dated on or after the file's modification date is treated as partial. `run_holdout.py` loads exactly through the recorded date, even if the cache is refreshed later.

## Holdout (run_holdout.py)

16. **Order.**
    1. Check the one-shot status.
    2. Run `verify_decision`: complete, non-smoke, built on prereg/dev_results.json, c\* named, all hashes match.
    3. Check the dev-data fingerprint.
    4. Freeze the isotonic map on c\*'s 2001-2018 dev pairs to `confidence_map_frozen.json`.
    5. Unlock and load the data.
    6. Run 16 fits, score, write `holdout_results.json` and `verdict.json`.
    7. Only then compute context benchmarks and deployment artifacts.

    Nothing is written, not even `holdout_log.txt`, until steps 1-3 pass.
17. **Isotonic map.** It is applied as `np.interp(pmax, x, y)` on sklearn's thresholds, which equals `predict` with clip. The deployed `confidence_map.json` is the frozen map, unless the holdout calibrated ECE is above 0.05. In that case it is refit on dev pairs for 2001-2018 plus holdout pairs for 2019-2025; this changes the display only.
18. **Resuming after a crash.**
    - Before the verdict: `--fix-reason`, plus `--fix-diff` and `--accept-changed` if a file changed. Fits already written are reused.
    - After the verdict: only the context and deployment steps can be redone. The verdict is never recomputed.

    Each resume's reason and diff go into `holdout_log.txt`.
19. **Verdict.** It is PASS only when H1-H6 all pass; H7 is recorded as "deferred to deployment". Deployment artifacts are written on PASS and FAIL alike: c\* is the candidate on PASS and the shadow model on FAIL.
    - `hmm_params_<c*>_2026.json`: the repo's hmm_params.json shape plus `recipe`, `meta` and `deployment`;
    - `b0_shadow_params_2026.json`;
    - `wf_history.csv`: as-called tempered filtered probabilities, using dev fits for 2001-2018 and holdout fits for 2019 onward;
    - `confidence_map.json`;
    - `manifest.json`.
20. **Timestamp.** `fitted_at_utc` follows the repo's format (`YYYY-MM-DD HH:MM:SS+00:00`).
21. **Code-test hook.** `--dry-run-dev <smoke dir> [--dry-run-cstar C3]` runs the whole holdout pipeline on 2018 dev data. Its outputs go to `<smoke dir>/holdout_dryrun/`. It never unlocks, and `load_raw` still refuses anything after 2018.

## Engineering

22. **The `select.py` name.** It shadows the stdlib `select` module, which on this machine is `DLLs/select.pyd` (socket → selectors → select). The safeguards:
    - imported under that name, `select.py` loads and hands over the real module;
    - `run_dev.py` and `run_holdout.py` put prereg/ last on `sys.path` and `chdir` to wf1/, so spawned workers never see it.

## Process note

23. While checking the cache's last date, the last three ^GSPC rows (2026-10-07 to 10-09) were printed to the console. No value was used in any code, recipe, threshold or decision.

## Run log (full development grid, 2026-10-12)

24. **Scoring-pool crash, rerun with fewer workers.** The first `run_dev.py` run (default 19 workers) finished all 108 fits (all `ok`, all refit gates passed), then the scoring pool died: a worker exited abruptly (`BrokenProcessPool`, siblings `WinError 6`). This happened before any metric was computed or written; the log is kept as `dev_run_attempt1.log`. It was rerun as `run_dev.py --workers 8`. No code, recipe, threshold or metric changed. The 108 fit files were reused through `_reusable()` (same seeds, recipes and PREREG hashes), so the fits are identical. `select.py` then ran unchanged.
