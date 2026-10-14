# Market Regime HMM

A six-state hidden Markov model that sorts each trading day of the S&P 500 into one of six market moods (regimes), using trend momentum, the VIX, and daily, weekly, monthly and yearly returns.

| | Quiet | Volatile |
|---|---|---|
| **Bull** | Bull Quiet | Bull Volatile |
| **Sideways** | Sideways Quiet | Sideways Volatile |
| **Bear** | Bear Quiet | Bear Volatile |

**Live dashboard:** https://maxgilbertson.github.io/market-regime-hmm/

The page shows today's mood with a calibrated confidence word, the model's forecast of the size of tomorrow's move and the chance of a 5% fall within a month, what the model sees today, recent moods, the model's track record on data it never saw, and three context signals it does not use (credit stress, the VIX curve, small caps vs the S&P 500).

## The model (v2, live since 14 October 2026)

**Inputs.** Today's S&P 500 move measured against the daily move implied by yesterday's VIX (so a 1% fall counts for more on a calm day than in a panic); 5-day, 21-day and 252-day log returns; the gap to the 200-day average; the VIX; and 21-day realised volatility. S&P features are built on the full S&P calendar from 1985, and the VIX is joined without filling gaps, so the model's history starts on 3 January 1990. Inputs are standardised with the training rows only and clipped at ±5 sd.

**Calibration.** A Gaussian HMM with full covariances, fitted by Baum-Welch EM with 8 restarts. Each state starts from a rule-based bucket (direction from the signs of the 200-day trend and the 1-month return, Quiet or Volatile from the training VIX median), and a prior on the state means keeps each mood on its definition. The model is refit every 2 January on all data to 31 December (`refit.yml`). A new fit is kept only if every mood still means what its name says and it agrees with the previous fit on at least 85% of the last five years' days. Otherwise the old fit stays and an issue is opened.

**Inference.** A forward pass with the evidence tempered by 0.5 gives the filtered probabilities shown on the page. They are steadier and better calibrated than the raw filter. The top probability becomes a confidence word through an isotonic map fitted on 2001-2018 walk-forward days. On unseen 2019-2026 days its calibration error was 0.021, and "very sure" calls held up 97 times in 100. The model also gives:
- a forecast of the size of tomorrow's move: the mixture over moods of each mood's return spread, scaled by today's VIX;
- the chance of a 5% fall within 21 trading days: each mood's shrunk historical rate, weighted by today's probabilities.

## How it was chosen

In October 2026 five upgrades were tested against the previous model under a plan fixed in advance ([PREREG.md](docs/upgrade-2026-10/PREREG.md)): history back to 1986, credit spreads, VIX-scaled moves and steadier probabilities. An earlier screen also tried the yield curve, the dollar, small caps, VIX term structure and weekly macro data, which did not help or cut too much history. VIX-scaled moves won on 2001-2018. On unseen 2019-2026 data, against the previous model:
- it forecast daily moves better (95% lower bound of the gain +0.026 nats a day);
- it beat a standard GARCH volatility model;
- it changed mood about 10 times a year instead of 15;
- it caught all six falls of 10% or more.

It failed one pre-set label check: after "Bear Quiet" days a 5% fall followed more often than after "Bear Volatile" days. It went live because accuracy comes first, and the page says to read both Bear moods as "falling" rather than as a ranking of danger. Full record: [docs/upgrade-2026-10/RESULTS.md](docs/upgrade-2026-10/RESULTS.md). The previous model is kept in `model/legacy/`.

## Live updates

A GitHub Actions workflow runs every 15 minutes. It downloads fresh S&P 500 and VIX prices (falling back to the last good copy if Yahoo Finance is down), re-runs the filter with `model/hmm_params.json` in about 5 seconds, fetches the context signals, and republishes to GitHub Pages. During US market hours the latest day is a live, partial session, so the call can move intraday. Open pages check `data/summary.json` and redraw when new data arrives. GitHub runs scheduled jobs on a best-effort basis, so an update can land a few minutes late.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python hmm_regime.py          # downloads ^GSPC and ^VIX, runs the model, writes out/
python context_data.py        # optional context signals for the page
python build_dashboard.py     # bundles out/ and model/ into regime_dashboard.html
```

Other modes of `hmm_regime.py`: `--offline` (use cached prices), `--refit --train-end YYYY-12-31` (yearly recalibration with its checks), `--append-wf YYYY` (add a finished year's as-called moods to `model/wf_history.csv`).

## Files

| File | Contents |
|---|---|
| `model/hmm_params.json` | Live calibration (trained on 1990-2025) and per-mood dip rates |
| `model/confidence_map.json` | Top probability to confidence, with its unseen-data check |
| `model/wf_history.csv` | Moods 2001-2026 as called at the time, each year by a fit trained only on earlier years |
| `model/track_record.json` | The 2019-2026 test results shown on the page |
| `out/regime_history.csv` | Daily inputs, filtered and smoothed probabilities, moods, forecast move size (`pred_sd`) and 5% fall chance (`p_dip21`) |
| `out/summary.json` | Today's call |
| `out/context.json` | Context signals (not used by the model) |
| `out/transition_matrix.csv`, `out/state_means.csv`, `out/regime_stats.csv` | Model details |

## Constraints and drawbacks

- **Overlapping returns.** The weekly, monthly and yearly returns share inputs, which breaks the HMM's independence assumption. Tempering and the confidence map correct most of the resulting over-confidence.
- **Six moods are imposed.** The 3 x 2 grid is a convention, and "Bear Quiet" is rare and weakly identified. The two Bear moods are not a reliable ranking of danger.
- **Gaussian moods.** Crash days are still more extreme than the model expects. VIX scaling narrows the gap.
- **Calls lag.** A new mood usually shows up 1-3 trading days after it starts. Smoothed paths use hindsight and are not tradable.
- **Limited inputs.** The model uses only the S&P 500 (price index, no dividends) and the VIX, from free Yahoo Finance data. Extra series were tested and did not help.

This is a research tool, not investment advice.
