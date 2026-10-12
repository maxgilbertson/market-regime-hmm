# Market Regime HMM

A six-state hidden Markov model that classifies the S&P 500 into one of six market regimes each day, using trend momentum, VIX levels, and daily, weekly, monthly and yearly returns.

| | Quiet | Volatile |
|---|---|---|
| **Bull** | Bull Quiet | Bull Volatile |
| **Sideways** | Sideways Quiet | Sideways Volatile |
| **Bear** | Bear Quiet | Bear Volatile |

The model reports filtered regime probabilities, regime entropy, and smoothed and Viterbi paths. It is calibrated with Baum-Welch (forward-backward EM) on 1990-2022, and everything after 2022 is out-of-sample. The page turns the top probability into a plain confidence word using the model's walk-forward track record, and drops the next-day regime forecast, which was right no more often than guessing "same as today".

**Live dashboard:** https://maxgilbertson.github.io/market-regime-hmm/

## Live updates

A GitHub Actions workflow runs every 15 minutes. It downloads fresh S&P 500 and VIX prices, re-runs the filter with the saved calibration in `model/hmm_params.json`, and republishes the dashboard to GitHub Pages. During US market hours the latest day is a live, partial session, so the current regime call can move intraday. An open dashboard reloads itself every 15 minutes.

GitHub runs scheduled jobs on a best-effort basis, so an update can land a few minutes late. Raw outputs from the latest run are also published under `/data/`: `summary.json`, `regime_history.csv` and `data.json` (the compact data the page draws from). The page checks `summary.json` every few minutes and redraws itself when new data arrives.

To recalibrate, run `python hmm_regime.py --refit` locally and commit the new `model/hmm_params.json`. Changing any modelling flag also triggers a refit automatically.

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python hmm_regime.py          # downloads ^GSPC and ^VIX, fits the model, writes out/
python build_dashboard.py     # bundles out/ into regime_dashboard.html
```

Useful options for `hmm_regime.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--ticker` | `^GSPC` | Index to model |
| `--start` | `1990-01-01` | First date downloaded |
| `--train-end` | `2022-12-31` | End of the calibration window |
| `--seeds` | `8` | EM restarts |
| `--anchor` | `sign` | Regime anchor rule: `sign` or `tercile` |
| `--mean-weight` | `3000` | Prior strength holding state means to their anchors; `0` gives free EM |
| `--outdir` | `out` | Output folder |
| `--refit` | off | Ignore the saved calibration and fit again |
| `--offline` | off | Use cached prices instead of downloading |

## October 2026 upgrade test

Five upgrades were tested against this model under a plan fixed in advance: history back to 1986, credit spreads, VIX-scaled daily moves and steadier probabilities. The best, VIX-scaled moves, forecast better and switched mood far less often on unseen 2019-2026 data. It failed one pre-set check: "Bear Quiet" was followed by drops more often than "Bear Volatile". So this model stays, and the candidate (`model/shadow/`) is re-tested on new data until October 2027. Full record: [docs/upgrade-2026-10/RESULTS.md](docs/upgrade-2026-10/RESULTS.md).

| File | Contents |
|---|---|
| `model/hmm_params.json` | Live calibration (1990-2022) |
| `model/confidence_map.json` | Maps the top probability to how often such calls held up, 2001-2025 walk-forward |
| `model/wf_history.csv` | Walk-forward moods 2001-2026, each year from a fit trained only on earlier years; used for the page's "what usually happens next" figures |
| `model/shadow/` | Frozen upgrade candidate for the prospective test |

## Method

**Observations.** Log returns over 1, 5, 21 and 252 trading days; trend momentum as log(close / 200-day SMA); log VIX; log 21-day realised volatility. All features are standardised with training-window statistics only, so there is no look-ahead in the out-of-sample period.

**Calibration.** A Gaussian HMM with full covariances, fit by Baum-Welch EM. Each state is initialised from a rule-based bucket: direction from the sign of the 200-day trend and the 1-month return, and volatility from the VIX median. A Gaussian prior on the state means keeps each state on its definition during EM.

**Inference.** A log-space forward pass gives filtered probabilities P(state at t | data up to t). A forward-backward pass gives smoothed probabilities, cross-checked against hmmlearn to about 1e-11. Regime entropy is the Shannon entropy of the filtered distribution divided by log 6, so 0 means certain and 1 means uniform.

## Outputs

| File | Contents |
|---|---|
| `out/regime_history.csv` | Daily features, filtered and smoothed probabilities, labels and entropy |
| `out/summary.json` | Latest regime call and probabilities |
| `out/transition_matrix.csv` | Daily regime transition probabilities |
| `out/regime_stats.csv` | Per-regime next-day return, volatility, Sharpe and average VIX |
| `out/state_means.csv` | Fitted state centres in raw units |
| `out/regime_chart.png` | Static chart of the last 12 years |
| `out/report.txt` | Console report from the last run |

## Constraints and drawbacks

- **Overlapping returns.** The weekly, monthly and yearly returns share most of their inputs, which breaks the HMM's conditional-independence assumption. This inflates persistence and makes the filtered probabilities over-confident.
- **Gaussian emissions.** Returns and VIX are fat-tailed, so crash days can flip the regime abruptly.
- **Six states are imposed.** The 3 x 2 grid is a modelling convention, not a count selected by BIC or cross-validation.
- **Labels are partly imposed.** Free EM does not find a real Bear Quiet state, because persistent low-VIX bear markets barely exist in S&P history. The mean prior holds it in place at a cost of about 0.2 nats per observation.
- **Hindsight in design choices.** The anchor rule and prior weight were chosen after viewing full-sample results.
- **Filtered calls lag.** Switches typically take 2 to 5 sessions. Smoothed and Viterbi paths use future data and are not tradable.
- **Descriptive statistics only.** The regime statistics ignore costs and sizing and are not a backtest.
- **Limited data.** One price index without dividends and one volatility gauge, from free yfinance downloads.

This is a research tool, not investment advice.
