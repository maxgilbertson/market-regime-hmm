# Model upgrade test, October 2026

**Verdict under the pre-set rules: FAIL** (one label check). **Update, 14 October 2026:** Max asked for the most accurate model, so C3 was made the live model anyway. It is the more accurate model on every accuracy measure in the test. The failed check is stated on the page: read "Bear Quiet" and "Bear Volatile" both as "falling", not as a ranking of danger.

The test plan in [PREREG.md](PREREG.md) was written and fingerprinted (SHA-256 `8ae42ec6…41f2`) before any candidate was scored on 2019–2026 data. [deviations.md](deviations.md) records every interpretation choice made while coding it.

## What was tested

Five candidates against the live model (B0), all keeping the same six moods and their definitions.

| Candidate | Change |
|---|---|
| C1 | Steadier probabilities (likelihood tempering) and the full 1990 history |
| C2 | C1 plus 1986–1989 history, bridging the old VXO index into the VIX to include the 1987 crash |
| C3 | C1 with each day's move measured against the VIX (VIX-scaled daily return) |
| C4 | C3 plus the 1986–1989 history |
| C5 | C4 plus credit spreads (Moody's Baa minus 10-year Treasury, with its real publication delay) |

An earlier screen also tried the yield curve, the US dollar, small caps vs the S&P 500, Nasdaq, NYSE and transport indices, VIX term structure, range-based volatility, drawdown measures and weekly financial-conditions and jobless-claims data. None helped inside the model, or they cut too much history, so they were not taken forward.

## Selection on 2001–2018

Each candidate was refit every January on all earlier data, then scored on the following year. The score is how well it forecast the next day's S&P 500 move (log score, higher is better).

| | Score | Eligible | Reason if not |
|---|---|---|---|
| B0 (live) | 3.210 | baseline | |
| C1 | 3.216 | yes | |
| C2 | 3.227 | no | mood calls drifted too far from the live model's (79% agreement, needs 80%) |
| **C3** | **3.282** | **yes, selected** | |
| C4 | 3.281 | no | mood drift, and worse agreement with hindsight |
| C5 | 3.286 | no | mood drift (73% agreement, one mood's share moved 5.2 points) |

## The one-shot test on 2019-01-02 to 2026-10-09 (C3 vs B0)

| Check | C3 | B0 | Pass? |
|---|---|---|---|
| H1 Next-day forecast score | 3.254 | 3.209 | yes (+0.045, 95% lower bound +0.026) |
| H2 Next-month volatility ranking | 0.61 | 0.56 | yes |
| H3 Mood meanings hold | 2 ordering slips | 1 | **no** |
| H4 Error against a hindsight rulebook (lower is better) | 1.154 | 1.167 | yes |
| H5 Mood switches per year | 10.2 | 15.2 | yes |
| H5 Quick back-and-forth flips per year | 0.1 | 2.2 | yes |
| H6 Bear calls in 6 corrections: median delay / missed | 20.5 days / 0 | 19 days / 1 | yes |

H3 failed because "Bear Quiet" days were followed by a 5% drop within a month more often than "Bear Volatile" days. The plan allowed no more such slips than the live model had (one), and C3 had two. Every other check favoured C3, but the rule was set in advance, so the live model stays.

For context, on the same 2019–2026 days a standard GARCH(1,1) volatility model with fat tails scored 3.243. C3 beat it and B0 did not. A simple rulebook (the six moods from fixed thresholds, with no hidden Markov model) scored 3.257, slightly above both.

## What shipped on 12 October (old model kept)

- Confidence words calibrated from the model's walk-forward record, so "100%" no longer appears.
- "What usually happens next" figures from walk-forward days only.
- The "predicted tomorrow" mood removed (it was right no more often than guessing "same as today").
- A plain note that Bear moods describe the recent past, not a forecast.

## What shipped on 14 October (C3 live)

- `model/hmm_params.json` is the holdout's 2026 fit of C3 (trained on 1990-01-03 to 2025-12-31), exactly as evaluated. The live code reproduces the test's 2026 probabilities to within 5e-11. The previous model is kept in `model/legacy/`.
- `model/confidence_map.json` is the isotonic map frozen on 2001-2018 before the test. On unseen 2019-2026 days its calibration error was 0.021, and "very sure" calls held up 97 times in 100.
- `model/wf_history.csv` holds C3's as-called walk-forward moods for 2001-2026.
- The model is refit every 2 January on all data to 31 December (`.github/workflows/refit.yml`), which is the cadence the test evaluated. A new fit is kept only if the mood labels still pass their checks and agree with the previous fit on at least 85% of the last five years' days. Otherwise the old fit stays and an issue is opened. Refitting on data to 2025-12-31 rebuilds the shipped calibration to within 1e-11.
- New page figures come straight from the model: the forecast size of tomorrow's move, and the chance of a 5% fall within 21 trading days (each mood's shrunk historical rate, weighted by today's probabilities).
- Credit stress, the VIX curve and small caps vs the S&P 500 are shown as context only. They are not model inputs, because they did not improve accuracy in the test.
