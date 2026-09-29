# Market Regime HMM

A six-state hidden Markov model that classifies the S&P 500 into one of six market regimes each day, using trend momentum, VIX levels, and daily, weekly, monthly and yearly returns.

| | Quiet | Volatile |
|---|---|---|
| **Bull** | Bull Quiet | Bull Volatile |
| **Sideways** | Sideways Quiet | Sideways Volatile |
| **Bear** | Bear Quiet | Bear Volatile |

The model reports filtered regime probabilities, regime entropy, smoothed and Viterbi paths, and a one-day-ahead regime forecast. It is calibrated with Baum-Welch (forward-backward EM) on 1990-2022, and everything after 2022 is out-of-sample.

**Live dashboard:** https://maxgilbertson.github.io/market-regime-hmm/

## Live updates

A GitHub Actions workflow runs every 15 minutes. It downloads fresh S&P 500 and VIX prices, re-runs the filter with the saved calibration in `model/hmm_params.json`, and republishes the dashboard to GitHub Pages. During US market hours the latest day is a live, partial session, so the current regime call can move intraday. An open dashboard reloads itself every 15 minutes.

GitHub runs scheduled jobs on a best-effort basis, so an update can land a few minutes late. Raw outputs from the latest run are also published under `/data/`, for example `summary.json` and `regime_history.csv`.

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

## Method

**Observations.** Log returns over 1, 5, 21 and 252 trading days; trend momentum as log(close / 200-day SMA); log VIX; log 21-day realised volatility. All features are standardised with training-window statistics only, so there is no look-ahead in the out-of-sample period.

**Calibration.** A Gaussian HMM with full covariances, fit by Baum-Welch EM. Each state is initialised from a rule-based bucket: direction from the sign of the 200-day trend and the 1-month return, and volatility from the VIX median. A Gaussian prior on the state means keeps each state on its definition during EM.

**Inference.** A log-space forward pass gives filtered probabilities P(state at t | data up to t). A forward-backward pass gives smoothed probabilities, cross-checked against hmmlearn to about 1e-11. Regime entropy is the Shannon entropy of the filtered distribution divided by log 6, so 0 means certain and 1 means uniform.

## Outputs

| File | Contents |
|---|---|
| `out/regime_history.csv` | Daily features, filtered and smoothed probabilities, labels and entropy |
| `out/summary.json` | Latest regime call, probabilities and forecast |
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
