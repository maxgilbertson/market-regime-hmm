"""
Six-regime Hidden Markov Model for the US equity market.

Observations (daily):
    ret_1d, ret_1w, ret_1m, ret_1y   log returns over 1 / 5 / 21 / 252 trading days
    trend_mom                        log(close / SMA200)  (trend momentum)
    log_vix                          log of VIX close
    log_rv21                         log of 21-day realised vol (annualised)

Model:
    Gaussian HMM, 6 hidden states, full covariance.
    Parameters calibrated with Baum-Welch EM (forward-backward) on the
    training window; observations standardised with training statistics.

Inference:
    filtered  P(s_t | y_1..t)         forward pass   (causal, what you trade on)
    smoothed  P(s_t | y_1..T)         forward-backward (hindsight, for labelling)
    predicted P(s_{t+1} | y_1..t)     filtered @ A
    entropy   H(filtered_t) / log(6)  0 = certain, 1 = uniform
    viterbi   most likely joint path

State -> regime mapping:
    Each hidden state gets a volatility score (log_vix + log_rv21 state means)
    and a direction score (ret_1m + trend_mom state means).  The three lowest
    volatility states are "Quiet", the three highest "Volatile"; within each
    group the direction rank gives Bear / Sideways / Bull.

Usage:
    python hmm_regime.py [--train-end 2022-12-31] [--start 1990-01-01]
                         [--ticker ^GSPC] [--seeds 8] [--outdir out]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")
import logging  # noqa: E402
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # MAP-EM makes the likelihood monitor chatty


class Regime(str, Enum):
    BULL_QUIET = "Bull Quiet"
    BULL_VOLATILE = "Bull Volatile"
    SIDEWAYS_QUIET = "Sideways Quiet"
    SIDEWAYS_VOLATILE = "Sideways Volatile"
    BEAR_QUIET = "Bear Quiet"
    BEAR_VOLATILE = "Bear Volatile"


REGIME_ORDER = [
    Regime.BULL_QUIET, Regime.BULL_VOLATILE,
    Regime.SIDEWAYS_QUIET, Regime.SIDEWAYS_VOLATILE,
    Regime.BEAR_QUIET, Regime.BEAR_VOLATILE,
]

# hue = direction (green / blue / red), lightness = volatility (light = quiet)
REGIME_COLOR = {
    Regime.BULL_QUIET: "#5fbf5f",
    Regime.BULL_VOLATILE: "#006300",
    Regime.SIDEWAYS_QUIET: "#86b6ef",
    Regime.SIDEWAYS_VOLATILE: "#1c5cab",
    Regime.BEAR_QUIET: "#ec835a",
    Regime.BEAR_VOLATILE: "#b0201f",
}
REGIME_HATCH = {r: ("//" if "Volatile" in r.value else None) for r in REGIME_ORDER}

FEATURES = ["ret_1d", "ret_1w", "ret_1m", "ret_1y", "trend_mom", "log_vix", "log_rv21"]
ANCHOR_RULE = "sign"  # "sign" (200d MA + 1m return signs) or "tercile" (momentum composite terciles)


# --------------------------------------------------------------------------- data
def load_data(ticker: str, start: str, cache: Path) -> pd.DataFrame:
    if cache.exists():
        raw = pd.read_csv(cache, index_col=0, parse_dates=True)
        if raw.index[-1] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=4):
            return raw
    import time
    import yfinance as yf

    series = {}
    for name, sym in (("close", ticker), ("vix", "^VIX")):
        for attempt in range(4):
            s = yf.Ticker(sym).history(start=start, auto_adjust=True)["Close"]
            if len(s) > 1000:
                break
            time.sleep(2 + attempt * 2)
        else:
            raise RuntimeError(f"could not download {sym}")
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        series[name] = s
    px = pd.concat(series, axis=1).dropna()
    px.to_csv(cache)
    return px


def build_features(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    lc = np.log(df["close"])
    df["ret_1d"] = lc.diff(1)
    df["ret_1w"] = lc.diff(5)
    df["ret_1m"] = lc.diff(21)
    df["ret_1y"] = lc.diff(252)
    df["trend_mom"] = lc - np.log(df["close"].rolling(200).mean())
    df["log_vix"] = np.log(df["vix"])
    df["log_rv21"] = np.log(df["ret_1d"].rolling(21).std() * np.sqrt(252) + 1e-8)
    return df.dropna()


# --------------------------------------------------------------------------- HMM
def anchor_labels(train: pd.DataFrame) -> np.ndarray:
    """
    Rule-based pre-labelling of the training window into the 6-regime grid.
    direction : terciles of a composite momentum index (1m return + trend + 1y return)
    volatility: log VIX above / below its training median
    State index k corresponds to REGIME_ORDER[k] by construction.
    """
    def z(s):
        return (s - s.mean()) / s.std()

    if ANCHOR_RULE == "sign":
        # Bear: price below 200d MA and negative 1m return; Bull: above MA and positive 1m;
        # Sideways: mixed signals (pullback in uptrend / bounce in downtrend)
        bear = (train["trend_mom"] < 0) & (train["ret_1m"] < 0)
        bull = (train["trend_mom"] > 0) & (train["ret_1m"] > 0)
        d = np.where(bull, "Bull", np.where(bear, "Bear", "Sideways"))
    else:
        direction = z(train["ret_1m"]) + z(train["trend_mom"]) + z(train["ret_1y"])
        lo, hi = direction.quantile([1 / 3, 2 / 3])
        d = np.where(direction > hi, "Bull", np.where(direction < lo, "Bear", "Sideways"))
    v = np.where(train["log_vix"] > train["log_vix"].median(), "Volatile", "Quiet")
    names = [f"{a} {b}" for a, b in zip(d, v)]
    lookup = {r.value: i for i, r in enumerate(REGIME_ORDER)}
    return np.array([lookup[n] for n in names])


def fit_hmm(X: np.ndarray, anchors: np.ndarray, seeds: int, mean_weight: float = 0.0):
    """
    Baum-Welch EM with the six states initialised at the regime anchors
    (mean / covariance of each rule-labelled bucket, sticky transitions).
    seed 0 is the pure anchored start; other seeds jitter the means slightly.
    mean_weight > 0 adds a Gaussian prior on each state mean centred on its
    anchor with that many pseudo-observations (MAP-EM): the quant-rule
    definitions hold unless the data strongly disagrees.
    """
    from hmmlearn.hmm import GaussianHMM

    K, d = len(REGIME_ORDER), X.shape[1]
    mu0 = np.stack([X[anchors == k].mean(axis=0) for k in range(K)])
    cov0 = np.stack([np.cov(X[anchors == k].T) + 1e-3 * np.eye(d) for k in range(K)])
    A0 = np.full((K, K), 0.01)
    np.fill_diagonal(A0, 0.95)
    A0 /= A0.sum(axis=1, keepdims=True)

    best, best_ll = None, -np.inf
    prior = {"means_prior": mu0, "means_weight": mean_weight} if mean_weight > 0 else {}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        m = GaussianHMM(
            n_components=K, covariance_type="full", n_iter=1000, tol=1e-5,
            random_state=seed, min_covar=1e-4, init_params="", params="stmc",
            transmat_prior=1.0 + 1e-3, startprob_prior=1.0 + 1e-3,  # tiny pseudo-counts: no NaN rows
            **prior,
        )
        m.startprob_ = np.full(K, 1 / K)
        m.transmat_ = A0.copy()
        m.means_ = mu0 + (0.15 * rng.standard_normal(mu0.shape) if seed else 0.0)
        m.covars_ = cov0.copy()
        try:
            m.fit(X)
        except Exception as e:  # noqa: BLE001
            print(f"  seed {seed}: fit failed ({e})", file=sys.stderr)
            continue
        ll = m.score(X)
        conv = m.monitor_.converged
        print(f"  seed {seed}: loglik {ll:,.1f}  iters {m.monitor_.iter}  converged {conv}")
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("all HMM fits failed")
    return best, best_ll


def emission_logprob(model, X: np.ndarray) -> np.ndarray:
    """log N(y_t | mu_k, Sigma_k) for every t, k."""
    K = model.n_components
    out = np.empty((len(X), K))
    for k in range(K):
        out[:, k] = multivariate_normal.logpdf(X, model.means_[k], model.covars_[k], allow_singular=True)
    return out


def forward_backward(log_pi: np.ndarray, log_A: np.ndarray, logB: np.ndarray):
    """
    Returns filtered, smoothed, predicted (one step ahead), log-likelihood.
    All in log space; no scaling issues.
    """
    T, K = logB.shape
    log_alpha = np.empty((T, K))
    log_alpha[0] = log_pi + logB[0]
    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + logB[t]
    loglik = logsumexp(log_alpha[-1])

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        log_beta[t] = logsumexp(log_A + (logB[t + 1] + log_beta[t + 1])[None, :], axis=1)

    filtered = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    log_post = log_alpha + log_beta
    smoothed = np.exp(log_post - logsumexp(log_post, axis=1, keepdims=True))
    predicted = filtered @ np.exp(log_A)
    return filtered, smoothed, predicted, loglik


def viterbi(log_pi, log_A, logB) -> np.ndarray:
    T, K = logB.shape
    delta = np.empty((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = log_pi + logB[0]
    for t in range(1, T):
        cand = delta[t - 1][:, None] + log_A
        psi[t] = cand.argmax(axis=0)
        delta[t] = cand.max(axis=0) + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = delta[-1].argmax()
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]
    return path


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1])


# --------------------------------------------------------------------------- labelling
def label_states(model, mu: np.ndarray, sd: np.ndarray) -> dict[int, Regime]:
    """
    Consistency check on the anchored states after EM.
    Re-derives the regime grid from the fitted state means (volatility rank
    splits Quiet / Volatile, direction rank inside each half gives
    Bear / Sideways / Bull) and warns if EM drifted away from the anchors.
    The anchored identity mapping is always used; this is diagnostic.
    """
    means = model.means_ * sd + mu  # back to raw feature units
    f = {name: means[:, i] for i, name in enumerate(FEATURES)}

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-12)

    vol_score = z(f["log_vix"]) + z(f["log_rv21"])
    dir_score = z(f["ret_1m"]) + z(f["trend_mom"]) + z(f["ret_1w"])

    K = len(vol_score)
    by_vol = np.argsort(vol_score)
    quiet, volatile = by_vol[: K // 2], by_vol[K // 2:]
    mapping: dict[int, Regime] = {}
    for group, names in (
        (quiet, [Regime.BEAR_QUIET, Regime.SIDEWAYS_QUIET, Regime.BULL_QUIET]),
        (volatile, [Regime.BEAR_VOLATILE, Regime.SIDEWAYS_VOLATILE, Regime.BULL_VOLATILE]),
    ):
        for s, name in zip(group[np.argsort(dir_score[group])], names):
            mapping[int(s)] = name
    return mapping


# --------------------------------------------------------------------------- plotting
def make_chart(df: pd.DataFrame, probs: pd.DataFrame, regime_cols, outfile: Path,
               years: int = 12, train_end=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.patches import Patch

    d = df[df.index >= df.index[-1] - pd.DateOffset(years=years)]
    p = probs.loc[d.index]

    fig, axes = plt.subplots(
        4, 1, figsize=(15, 13), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 1.2, 1.2], "hspace": 0.08},
    )
    fig.patch.set_facecolor("#fcfcfb")
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"

    # panel 1: price with filtered-regime shading
    ax = axes[0]
    ax.plot(d.index, d["close"], color=ink, lw=1.2)
    ax.set_yscale("log")
    reg = d["regime_filtered"].values
    start = 0
    for i in range(1, len(d) + 1):
        if i == len(d) or reg[i] != reg[start]:
            r = Regime(reg[start])
            ax.axvspan(d.index[start], d.index[min(i, len(d) - 1)], color=REGIME_COLOR[r],
                       alpha=0.28, lw=0, hatch=REGIME_HATCH[r])
            start = i
    ax.set_ylabel("S&P 500 (log)", color=ink)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("HMM market regimes (filtered, argmax) - shading = regime, hatch = volatile",
                 loc="left", color=ink, fontsize=12)
    if train_end is not None and d.index[0] < train_end < d.index[-1]:
        for a in axes:
            a.axvline(train_end, color=muted, lw=1, ls="--")
        ax.annotate("out-of-sample ->", (train_end, d["close"].min()), xytext=(4, 4),
                    textcoords="offset points", color=muted, fontsize=8)
    handles = [Patch(facecolor=REGIME_COLOR[r], alpha=0.5, hatch=REGIME_HATCH[r], label=r.value)
               for r in REGIME_ORDER]
    ax.legend(handles=handles, ncol=6, loc="upper left", fontsize=8, frameon=False)

    # panel 2: stacked filtered probabilities
    ax = axes[1]
    ax.stackplot(p.index, [p[c].values for c in regime_cols],
                 colors=[REGIME_COLOR[Regime(c)] for c in regime_cols], lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Filtered P(regime)", color=ink)

    # panel 3: entropy
    ax = axes[2]
    ax.plot(d.index, d["entropy_filtered"], color="#2a78d6", lw=1)
    ax.fill_between(d.index, 0, d["entropy_filtered"], color="#2a78d6", alpha=0.15, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Regime entropy", color=ink)

    # panel 4: VIX
    ax = axes[3]
    ax.plot(d.index, d["vix"], color="#eb6834", lw=1)
    ax.set_ylabel("VIX", color=ink)

    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=muted, labelsize=9)
    fig.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--train-end", default="2022-12-31",
                    help="EM calibration uses data up to this date; later dates are out-of-sample")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--anchor", choices=["sign", "tercile"], default="sign",
                    help="rule used to define the six regime anchors on the training window")
    ap.add_argument("--mean-weight", type=float, default=3000.0,
                    help="pseudo-observations pulling each state mean toward its anchor (0 = free EM). "
                         "Free EM fits ~0.2 nats/obs better but lets 'Bear Quiet' drift into a "
                         "positive-return dip state; 3000 keeps all six regimes on their definitions.")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    global ANCHOR_RULE
    ANCHOR_RULE = args.anchor

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("Loading data ...")
    px = load_data(args.ticker, args.start, outdir / "prices.csv")
    df = build_features(px)
    print(f"  {len(df):,} observations {df.index[0].date()} -> {df.index[-1].date()}")

    train = df.loc[: args.train_end]
    mu, sd = train[FEATURES].mean().values, train[FEATURES].std().values
    X_all = ((df[FEATURES] - mu) / sd).values
    X_tr = ((train[FEATURES] - mu) / sd).values
    print(f"  training window {train.index[0].date()} -> {train.index[-1].date()} ({len(train):,} obs)")

    anchors = anchor_labels(train)
    print("  anchor bucket sizes:",
          {r.value: int((anchors == i).sum()) for i, r in enumerate(REGIME_ORDER)})

    print(f"Calibrating 6-state Gaussian HMM with Baum-Welch (forward-backward EM), {args.seeds} restarts ...")
    model, ll_train = fit_hmm(X_tr, anchors, args.seeds, mean_weight=args.mean_weight)

    mapping = {k: REGIME_ORDER[k] for k in range(6)}          # anchored by construction
    check = label_states(model, mu, sd)                         # rank-based re-derivation
    drift = [f"{mapping[k].value} -> looks like {check[k].value}" for k in range(6) if check[k] != mapping[k]]
    if drift:
        print("  WARNING: EM drifted from anchors on:", "; ".join(drift))
    else:
        print("  post-EM state ordering consistent with anchors (vol split + direction rank)")

    print("Running forward (filter) and forward-backward (smoother) over full history ...")
    log_pi = np.log(model.startprob_ + 1e-300)
    log_A = np.log(model.transmat_ + 1e-300)
    logB = emission_logprob(model, X_all)
    filt, smooth, pred, ll_all = forward_backward(log_pi, log_A, logB)
    vit = viterbi(log_pi, log_A, logB)

    # cross-check smoother against hmmlearn's own forward-backward
    ref = model.predict_proba(X_all)
    print(f"  smoother max |diff| vs hmmlearn: {np.abs(ref - smooth).max():.2e}")
    oos = df.index > pd.Timestamp(args.train_end)
    print(f"  loglik/obs  in-sample {ll_train/len(X_tr):.4f}   "
          f"full {ll_all/len(X_all):.4f}")

    # -------- assemble output table (columns ordered by regime)
    order = sorted(range(6), key=lambda k: REGIME_ORDER.index(mapping[k]))
    regime_cols = [mapping[k].value for k in order]
    P_f = pd.DataFrame(filt[:, order], index=df.index, columns=regime_cols)
    P_s = pd.DataFrame(smooth[:, order], index=df.index, columns=regime_cols)
    P_p = pd.DataFrame(pred[:, order], index=df.index, columns=regime_cols)

    out = df[["close", "vix"] + FEATURES].copy()
    out["regime_filtered"] = P_f.idxmax(axis=1)
    out["regime_smoothed"] = P_s.idxmax(axis=1)
    out["regime_viterbi"] = [mapping[int(s)].value for s in vit]
    out["regime_predicted_next"] = P_p.idxmax(axis=1)
    out["p_max_filtered"] = P_f.max(axis=1)
    out["entropy_filtered"] = entropy(P_f.values)
    out["entropy_smoothed"] = entropy(P_s.values)
    out["out_of_sample"] = oos
    for c in regime_cols:
        out[f"pf_{c}"] = P_f[c]
    for c in regime_cols:
        out[f"ps_{c}"] = P_s[c]
    out.to_csv(outdir / "regime_history.csv", float_format="%.6f")

    # -------- transition matrix in regime order, expected durations
    A = model.transmat_[np.ix_(order, order)]
    A_df = pd.DataFrame(A, index=regime_cols, columns=regime_cols)
    A_df.to_csv(outdir / "transition_matrix.csv", float_format="%.4f")
    durations = {c: float(1 / (1 - A[i, i])) for i, c in enumerate(regime_cols)}

    # -------- per-regime realised statistics (filtered labelling, next-day return)
    fwd = out["ret_1d"].shift(-1)
    stats = []
    for c in regime_cols:
        m = out["regime_filtered"] == c
        r = fwd[m].dropna()
        stats.append({
            "regime": c,
            "days": int(m.sum()),
            "share": float(m.mean()),
            "ann_return_next_day": float(r.mean() * 252),
            "ann_vol_next_day": float(r.std() * np.sqrt(252)),
            "sharpe_next_day": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
            "mean_vix": float(out.loc[m, "vix"].mean()),
            "expected_duration_days": durations[c],
        })
    stats_df = pd.DataFrame(stats).set_index("regime")
    stats_df.to_csv(outdir / "regime_stats.csv", float_format="%.4f")

    # -------- state means in raw units
    means_raw = pd.DataFrame(model.means_[order] * sd + mu, index=regime_cols, columns=FEATURES)
    means_raw.to_csv(outdir / "state_means.csv", float_format="%.5f")

    # -------- current snapshot
    last = out.index[-1]
    recent = out.iloc[-10:]
    summary = {
        "as_of": str(last.date()),
        "ticker": args.ticker,
        "train_end": args.train_end,
        "n_obs": int(len(out)),
        "close": float(out.loc[last, "close"]),
        "vix": float(out.loc[last, "vix"]),
        "regime_filtered": out.loc[last, "regime_filtered"],
        "regime_smoothed": out.loc[last, "regime_smoothed"],
        "regime_viterbi": out.loc[last, "regime_viterbi"],
        "regime_predicted_tomorrow": out.loc[last, "regime_predicted_next"],
        "filtered_probs": {c: round(float(P_f.loc[last, c]), 4) for c in regime_cols},
        "predicted_probs_tomorrow": {c: round(float(P_p.loc[last, c]), 4) for c in regime_cols},
        "entropy_filtered": round(float(out.loc[last, "entropy_filtered"]), 4),
        "entropy_filtered_20d_mean": round(float(out["entropy_filtered"].iloc[-20:].mean()), 4),
        "features_last": {f: round(float(out.loc[last, f]), 5) for f in FEATURES},
        "expected_duration_days": {k: round(v, 1) for k, v in durations.items()},
        "loglik_per_obs_train": round(float(ll_train / len(X_tr)), 4),
        "loglik_per_obs_full": round(float(ll_all / len(X_all)), 4),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("Rendering chart ...")
    make_chart(out, P_f, regime_cols, outdir / "regime_chart.png",
               train_end=pd.Timestamp(args.train_end))

    # -------- console report
    pd.set_option("display.width", 200)
    print("\n=== CURRENT REGIME  (as of", summary["as_of"], ") ===")
    print(f"  close {summary['close']:,.2f}   VIX {summary['vix']:.2f}")
    print(f"  filtered : {summary['regime_filtered']}   (p={out.loc[last,'p_max_filtered']:.3f}, "
          f"entropy={summary['entropy_filtered']:.3f})")
    print(f"  smoothed : {summary['regime_smoothed']}")
    print(f"  viterbi  : {summary['regime_viterbi']}")
    print(f"  tomorrow : {summary['regime_predicted_tomorrow']}")
    print("\n  filtered probabilities:")
    for c, v in summary["filtered_probs"].items():
        print(f"    {c:<18} {v:6.1%}   {'#' * int(v * 40)}")
    print("\n=== LAST 10 DAYS ===")
    print(recent[["close", "vix", "regime_filtered", "p_max_filtered", "entropy_filtered"]].round(3).to_string())
    print("\n=== TRANSITION MATRIX (rows = from) ===")
    print(A_df.round(3).to_string())
    print("\n=== REGIME STATS (filtered labels, next-day returns) ===")
    print(stats_df.round(3).to_string())
    print("\n=== STATE MEANS (raw units) ===")
    print(means_raw.round(4).to_string())
    print(f"\nOutputs written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
