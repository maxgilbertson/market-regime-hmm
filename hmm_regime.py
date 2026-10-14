"""
Six-regime hidden Markov model for the S&P 500, model v2 (October 2026).

Regimes: Bull / Sideways / Bear  x  Quiet / Volatile.

Recipe (C3 "VIX-scaled", chosen in the October 2026 pre-registered test, docs/upgrade-2026-10/):
    z_1d       today's S&P 500 log return divided by the daily move implied by YESTERDAY's VIX
    ret_1w     5-day log return
    ret_1m     21-day log return
    ret_1y     252-day log return
    trend_mom  log(close / 200-day average)
    log_vix    log VIX
    log_rv21   log of 21-day realised volatility (annualised)
S&P features are built on the full S&P calendar (history from 1985) and VIX is joined without
forward-filling, so rows start 1990-01-03. Inputs are standardised with the training rows' mean
and sd, then clipped at +-5 sd.

Model: 6-state Gaussian HMM, full covariance, Baum-Welch EM started at rule-based anchors
(direction from the signs of trend_mom and ret_1m, Quiet/Volatile from the training VIX median)
with a MAP prior holding each state mean near its anchor, 8 restarts.

Inference:
    filtered   P(s_t | data to t), with the emission log-likelihood tempered by tau = 0.5
               (steadier, better-calibrated probabilities); this is what the page shows
    predicted  P(s_t+1 | data to t) = filtered @ A
    smoothed   P(s_t | all data), untempered; hindsight only
    pred_sd    the model's forecast of the size of tomorrow's move (mixture over regimes)
    p_dip21    chance of a 5% fall within 21 trading days, from per-regime "heads" fitted on
               training rows (each regime's shrunk historical dip rate)

Modes:
    python hmm_regime.py                         live run with model/hmm_params.json
    python hmm_regime.py --refit --train-end 2026-12-31
                                                 yearly recalibration; writes the new params only
                                                 if every label check passes (exit code 2 if not)
    python hmm_regime.py --append-wf 2026        append that year's as-called moods (from the
                                                 current out/regime_history.csv) to model/wf_history.csv
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import warnings
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # MAP-EM makes the likelihood monitor chatty

SQ252 = float(np.sqrt(252))


class Regime(str, Enum):
    BULL_QUIET = "Bull Quiet"
    BULL_VOLATILE = "Bull Volatile"
    SIDEWAYS_QUIET = "Sideways Quiet"
    SIDEWAYS_VOLATILE = "Sideways Volatile"
    BEAR_QUIET = "Bear Quiet"
    BEAR_VOLATILE = "Bear Volatile"


REGIME_ORDER = list(Regime)
REGS = [r.value for r in REGIME_ORDER]

RECIPE = {   # the evaluated C3 recipe; a params file may carry its own copy
    "id": "C3", "label": "VIX-scaled",
    "features": ["z_1d", "ret_1w", "ret_1m", "ret_1y", "trend_mom", "log_vix", "log_rv21"],
    "clip": 5.0, "tau": 0.5, "seeds": 8, "mean_weight": 3000.0,
    "dip_horizon": 21, "dip_threshold": -0.05, "dip_prior_weight": 63.0,
}
SPX_START, VIX_START = "1985-01-01", "1990-01-01"


# --------------------------------------------------------------------------- data
def _download(sym: str, start: str) -> pd.Series:
    import time
    import yfinance as yf

    for attempt in range(4):
        try:
            s = yf.Ticker(sym).history(start=start, auto_adjust=True)["Close"]
        except Exception:  # noqa: BLE001
            s = pd.Series(dtype=float)
        if len(s) > 1000:
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            return s[~s.index.duplicated(keep="last")].dropna()
        time.sleep(2 + attempt * 3)
    raise RuntimeError(f"could not download {sym}")


def load_data(cache: Path, offline: bool = False) -> pd.DataFrame:
    """S&P 500 calendar (master) from 1985 with the VIX joined on it (never forward-filled).
    Downloads fresh prices each run (the last row is today's live bar while US markets are open)
    and falls back to the cache if the download fails."""
    if not offline:
        try:
            spx, vix = _download("^GSPC", SPX_START), _download("^VIX", VIX_START)
            raw = pd.DataFrame({"close": spx, "vix": vix.reindex(spx.index)}, index=spx.index)
            if raw["close"].iloc[-1] <= 0 or raw.index[-1] < pd.Timestamp.today().normalize() - pd.Timedelta(days=10):
                raise RuntimeError("downloaded prices look stale or invalid")
            raw.to_csv(cache)
            return raw
        except Exception as e:  # noqa: BLE001
            print(f"  download failed ({e}); using cached prices", file=sys.stderr)
    if not cache.exists():
        raise RuntimeError(f"no cached prices at {cache}")
    raw = pd.read_csv(cache, index_col=0, parse_dates=True)
    if "close" not in raw:   # old cache format (inner-joined from 1990) cannot rebuild the 1990 rows
        raise RuntimeError(f"{cache} is in the old format; delete it and run online once")
    return raw


def build_features(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Recipe features (identical to the evaluation code). 'sig' = VIX/100/sqrt(252)."""
    df = raw[["close"]].copy()
    lc = np.log(df["close"])
    df["ret_1d"] = lc.diff(1)
    df["ret_1w"] = lc.diff(5)
    df["ret_1m"] = lc.diff(21)
    df["ret_1y"] = lc.diff(252)
    df["trend_mom"] = lc - np.log(df["close"].rolling(200).mean())
    df["log_rv21"] = np.log(df["ret_1d"].rolling(21).std() * SQ252 + 1e-8)
    df["vix"] = raw["vix"]
    df["log_vix"] = np.log(df["vix"])
    df["z_1d"] = df["ret_1d"] / (df["vix"].shift(1) / 100 / SQ252)   # previous S&P day's VIX
    df = df.dropna(subset=list(features) + ["vix"])
    df["sig"] = df["vix"] / 100 / SQ252
    return df


def zmat(df: pd.DataFrame, feats, mu, sd, clip) -> np.ndarray:
    Z = ((df[list(feats)] - mu) / sd).values
    return np.clip(Z, -clip, clip) if clip else Z


# --------------------------------------------------------------------------- fitting
def anchor_labels(train: pd.DataFrame, vix_med: float) -> np.ndarray:
    """Sign anchors: k = 2*direction + volatile (Bull/Sideways/Bear x Quiet/Volatile)."""
    bull = (train["trend_mom"] > 0) & (train["ret_1m"] > 0)
    bear = (train["trend_mom"] < 0) & (train["ret_1m"] < 0)
    d = np.where(bull, 0, np.where(bear, 2, 1))
    return 2 * d + (train["log_vix"] > vix_med).values.astype(int)


def fit_hmm(X: np.ndarray, anchors: np.ndarray, seeds: int, mean_weight: float = 0.0):
    """Baum-Welch EM started at the anchor buckets, sticky transitions; seed 0 is the pure anchored
    start, other seeds jitter the means. mean_weight > 0 adds a Gaussian prior on each state mean
    centred on its anchor (MAP-EM)."""
    from hmmlearn.hmm import GaussianHMM

    K, d = 6, X.shape[1]
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
            transmat_prior=1.0 + 1e-3, startprob_prior=1.0 + 1e-3, **prior,
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
        print(f"  seed {seed}: loglik {ll:,.1f}  iters {m.monitor_.iter}  converged {m.monitor_.converged}")
        if ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("all HMM fits failed")
    return best, best_ll


def label_gate(means, mu, sd, feats) -> dict:
    """Checks that the fitted states still mean what their names say."""
    raw = np.asarray(means) * np.asarray(sd) + np.asarray(mu)
    f = {name: raw[:, i] for i, name in enumerate(feats)}

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-12)

    vol, dirn = z(f["log_vix"]) + z(f["log_rv21"]), z(f["ret_1m"]) + z(f["trend_mom"]) + z(f["ret_1w"])
    by_vol = np.argsort(vol)
    mapping = {}
    for group, ks in ((by_vol[:3], [4, 2, 0]), (by_vol[3:], [5, 3, 1])):
        for s, k in zip(group[np.argsort(dirn[group])], ks):
            mapping[int(s)] = k
    i_tm, i_1m, i_v = feats.index("trend_mom"), feats.index("ret_1m"), feats.index("log_vix")
    g = {"rank_ok": all(mapping[k] == k for k in range(6)),
         "bull_sign_ok": all(raw[k, i_tm] > 0 and raw[k, i_1m] > 0 for k in (0, 1)),
         "bear_sign_ok": all(raw[k, i_tm] < 0 and raw[k, i_1m] < 0 for k in (4, 5)),
         "vol_twin_ok": all(raw[k, i_v] > raw[k - 1, i_v] for k in (1, 3, 5)),
         "bear_volatile_highest_vix": bool(raw[5, i_v] == raw[:, i_v].max())}
    g = {k: bool(v) for k, v in g.items()}
    g["pass"] = all(g.values())
    return g


# --------------------------------------------------------------------------- inference
def emission_logprob(means, covars, X: np.ndarray) -> np.ndarray:
    out = np.empty((len(X), len(means)))
    for k in range(len(means)):
        out[:, k] = multivariate_normal.logpdf(X, means[k], covars[k], allow_singular=True)
    return out


def forward_tempered(logB, pi, A, tau=1.0):
    """Filtered P(s_t | data to t) and predicted P(s_t+1 | data to t), emission log-lik scaled by tau."""
    from hmmlearn import _hmmc

    _, fwd = _hmmc.forward_log(pi, A, tau * logB)
    filt = np.exp(fwd - logsumexp(fwd, axis=1, keepdims=True))
    return filt, filt @ A


def smooth(logB, pi, A):
    from hmmlearn import _hmmc

    _, fwd = _hmmc.forward_log(pi, A, logB)
    bwd = _hmmc.backward_log(pi, A, logB)
    lg = fwd + bwd
    return np.exp(lg - logsumexp(lg, axis=1, keepdims=True))


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


def return_moments(P: dict, df: pd.DataFrame, feats) -> tuple[np.ndarray, np.ndarray]:
    """Per-day, per-regime mean and variance of the next day's raw log return."""
    i = feats.index("z_1d") if "z_1d" in feats else feats.index("ret_1d")
    m0 = P["means"][:, i] * P["feature_std"][i] + P["feature_mean"][i]
    v0 = P["covars"][:, i, i] * P["feature_std"][i] ** 2
    if feats[i] == "z_1d":                      # scaled by today's VIX-implied daily move
        s = df["sig"].values[:, None]
        return s * m0, s ** 2 * v0
    return np.tile(m0, (len(df), 1)), np.tile(v0, (len(df), 1))


def dip_heads(F: np.ndarray, df: pd.DataFrame, close: pd.Series, train_end, R: dict) -> np.ndarray:
    """Each regime's chance of a >=5% fall within 21 trading days, from training rows whose 21-day
    window ends by train_end, weighted by the filtered probabilities and shrunk to the overall rate."""
    H, w = R["dip_horizon"], R["dip_prior_weight"]
    fmin = close[::-1].rolling(H).min()[::-1].shift(-1)
    dip = ((fmin / close - 1) <= R["dip_threshold"]).astype(float).where(fmin.notna())
    d21 = pd.Series(close.index, index=close.index).shift(-H)
    ok = ((d21 <= pd.Timestamp(train_end)).reindex(df.index).fillna(False).values
          & np.isfinite(dip.reindex(df.index).values))
    Dv = dip.reindex(df.index).values
    qbar = Dv[ok].mean()
    return (F[ok].T @ Dv[ok] + w * qbar) / (F[ok].sum(axis=0) + w)


def load_params(path: Path) -> dict:
    blob = json.loads(path.read_text())
    R = {**RECIPE, **{k: v for k, v in (blob.get("recipe") or {}).items() if k in RECIPE}}
    R["features"] = blob["config"]["features"]
    P = {k: np.asarray(blob[k], float) for k in
         ("startprob", "transmat", "means", "covars", "feature_mean", "feature_std")}
    return {"blob": blob, "R": R, "P": P}


def infer(M: dict, df: pd.DataFrame):
    R, P = M["R"], M["P"]
    X = zmat(df, R["features"], P["feature_mean"], P["feature_std"], R["clip"])
    logB = emission_logprob(P["means"], P["covars"], X)
    F, Pn = forward_tempered(logB, P["startprob"], P["transmat"], R["tau"])
    S = smooth(logB, P["startprob"], P["transmat"])
    V = viterbi(np.log(P["startprob"] + 1e-300), np.log(P["transmat"] + 1e-300), logB)
    return F, Pn, S, V


# --------------------------------------------------------------------------- modes
def refit(args, raw: pd.DataFrame) -> int:
    """Yearly recalibration on all rows up to --train-end. Writes params only if every check passes."""
    R = dict(RECIPE)
    feats = R["features"]
    df = build_features(raw, feats)
    te = pd.Timestamp(args.train_end)
    train = df.loc[:te]
    mu, sd = train[feats].mean().values, train[feats].std().values
    X = zmat(train, feats, mu, sd, R["clip"])
    vmed = float(train["log_vix"].median())
    anc = anchor_labels(train, vmed)
    print(f"Refitting {R['id']} on {train.index[0].date()} -> {train.index[-1].date()} ({len(train):,} rows)")
    model, ll = fit_hmm(X, anc, R["seeds"], mean_weight=R["mean_weight"])
    gate = label_gate(model.means_, mu, sd, feats)
    blob = {
        "config": {"ticker": "^GSPC", "start": str(df.index[0].date()), "train_end": str(te.date()),
                   "seeds": R["seeds"], "anchor": "sign", "mean_weight": R["mean_weight"], "features": feats,
                   "refit": "yearly"},
        "fitted_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00"),
        "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
        "startprob": model.startprob_.tolist(), "transmat": model.transmat_.tolist(),
        "means": model.means_.tolist(), "covars": model.covars_.tolist(),
        "recipe": {k: R[k] for k in ("id", "label", "clip", "tau")},
        "meta": {"fit_year": te.year + 1, "train_last_row": str(train.index[-1].date()), "n_train": int(len(train)),
                 "train_loglik_per_obs": float(ll / len(X)), "vix_median_log": vmed, "gate": gate},
    }
    M = {"blob": blob, "R": {**R}, "P": {k: np.asarray(blob[k], float) for k in
                                         ("startprob", "transmat", "means", "covars", "feature_mean", "feature_std")}}
    F, _, _, _ = infer(M, df.loc[:te])
    blob["dip_heads"] = dip_heads(F, df.loc[:te], raw["close"], te, R).tolist()

    # stability check: the new fit's calls over the last 5 years should mostly agree with the old fit's
    agree = None
    path = Path(args.model_file)
    if path.exists():
        old = load_params(path)
        recent = df.loc[te - pd.DateOffset(years=5):te]
        Fo, _, _, _ = infer(old, df.loc[:te])
        Fn, _, _, _ = infer(M, df.loc[:te])
        k = df.loc[:te].index.isin(recent.index)
        agree = float((Fo[k].argmax(1) == Fn[k].argmax(1)).mean())
    share = pd.Series(F.argmax(1)).iloc[-2520:].value_counts(normalize=True).reindex(range(6), fill_value=0)
    checks = {**gate, "every_mood_seen_1pct_last_10y": bool((share >= 0.01).all()),
              "agrees_with_previous_fit_85pct_last_5y": agree is None or agree >= 0.85}
    checks["pass"] = all(v for k, v in checks.items() if k != "pass")
    print("  checks:", checks, f"(agreement {agree:.3f})" if agree is not None else "")
    status = {"train_end": str(te.date()), "checked_utc": blob["fitted_at_utc"], "checks": checks, "agreement": agree}
    (path.parent / "refit_status.json").write_text(json.dumps(status, indent=1))
    if not checks["pass"]:
        print("REFIT REJECTED: keeping the previous calibration", file=sys.stderr)
        return 2
    path.write_text(json.dumps(blob, indent=1))
    print(f"  saved {path}")
    return 0


def append_wf(year: int, out: Path) -> int:
    """Append a finished year's as-called moods (from the live run's history) to model/wf_history.csv."""
    h = pd.read_csv(out / "regime_history.csv", index_col=0, parse_dates=True)
    rows = h[h.index.year == year]
    if rows.empty:
        print(f"no rows for {year}", file=sys.stderr)
        return 1
    wf_path = Path("model/wf_history.csv")
    wf = pd.read_csv(wf_path, index_col=0, parse_dates=True) if wf_path.exists() else pd.DataFrame()
    add = pd.DataFrame({"model_year": year, "regime": rows["regime_filtered"], "p_max": rows["p_max_filtered"].round(6)})
    for r in REGS:
        add[f"p_{r}"] = rows[f"pf_{r}"].round(6)
    add.index.name = "Date"
    wf = pd.concat([wf[wf.index.year != year], add]).sort_index() if len(wf) else add
    wf.to_csv(wf_path, float_format="%.6f", date_format="%Y-%m-%d")
    print(f"appended {len(add)} rows for {year} to {wf_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--model-file", default="model/hmm_params.json")
    ap.add_argument("--offline", action="store_true", help="use cached prices instead of downloading")
    ap.add_argument("--refit", action="store_true", help="yearly recalibration (needs --train-end)")
    ap.add_argument("--train-end", default=None)
    ap.add_argument("--append-wf", type=int, default=None, metavar="YEAR")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(exist_ok=True)
    if args.append_wf:
        return append_wf(args.append_wf, out)

    updated_at = pd.Timestamp.now(tz="UTC")
    print("Loading data ...")
    raw = load_data(out / "prices.csv", offline=args.offline)
    if args.refit:
        if not args.train_end:
            ap.error("--refit needs --train-end")
        return refit(args, raw)

    M = load_params(Path(args.model_file))
    R, P, blob = M["R"], M["P"], M["blob"]
    feats = R["features"]
    df = build_features(raw, feats)
    print(f"  {len(df):,} rows {df.index[0].date()} -> {df.index[-1].date()}; "
          f"calibration trained to {blob['config']['train_end']} ({R.get('id', '?')}, tau {R['tau']})")

    F, Pn, S, V = infer(M, df)
    Mk, Vk = return_moments(P, df, feats)
    pm = (Pn * Mk).sum(1)
    pred_sd = np.sqrt((Pn * (Vk + Mk ** 2)).sum(1) - pm ** 2)
    q = np.asarray(blob.get("dip_heads") or dip_heads(F, df, raw["close"], blob["config"]["train_end"], R))
    p_dip = F @ q

    train_end = pd.Timestamp(blob["config"]["train_end"])
    outdf = df[["close", "vix", "ret_1d"] + [f for f in feats if f != "ret_1d"]].copy()
    outdf["regime_filtered"] = [REGS[i] for i in F.argmax(1)]
    outdf["regime_smoothed"] = [REGS[i] for i in S.argmax(1)]
    outdf["regime_viterbi"] = [REGS[i] for i in V]
    outdf["p_max_filtered"] = F.max(1)
    outdf["entropy_filtered"] = entropy(F)
    outdf["entropy_smoothed"] = entropy(S)
    outdf["pred_sd"] = pred_sd
    outdf["p_dip21"] = p_dip
    outdf["out_of_sample"] = df.index > train_end
    for k, r in enumerate(REGS):
        outdf[f"pf_{r}"] = F[:, k]
    for k, r in enumerate(REGS):
        outdf[f"ps_{r}"] = S[:, k]
    outdf.to_csv(out / "regime_history.csv", float_format="%.6f")

    A = P["transmat"]
    pd.DataFrame(A, index=REGS, columns=REGS).to_csv(out / "transition_matrix.csv", float_format="%.4f")
    pd.DataFrame(P["means"] * P["feature_std"] + P["feature_mean"], index=REGS, columns=feats).to_csv(
        out / "state_means.csv", float_format="%.5f")
    durations = {r: float(1 / (1 - A[k, k])) for k, r in enumerate(REGS)}
    fwd = outdf["ret_1d"].shift(-1)
    stats = []
    for r in REGS:
        m = outdf["regime_filtered"] == r
        x = fwd[m].dropna()
        stats.append({"regime": r, "days": int(m.sum()), "share": float(m.mean()),
                      "ann_return_next_day": float(x.mean() * 252), "ann_vol_next_day": float(x.std() * SQ252),
                      "mean_vix": float(outdf.loc[m, "vix"].mean()), "expected_duration_days": durations[r]})
    pd.DataFrame(stats).set_index("regime").to_csv(out / "regime_stats.csv", float_format="%.4f")

    last = outdf.index[-1]
    summary = {
        "as_of": str(last.date()),
        "updated_at_utc": updated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_fitted_at_utc": blob.get("fitted_at_utc"),
        "model": {"recipe": R.get("id"), "label": R.get("label"), "tau": R["tau"], "clip": R["clip"],
                  "features": feats, "train_start": blob["config"]["start"], "train_end": blob["config"]["train_end"]},
        "train_end": blob["config"]["train_end"],
        "ticker": "^GSPC",
        "n_obs": int(len(outdf)),
        "close": float(outdf.loc[last, "close"]),
        "vix": float(outdf.loc[last, "vix"]),
        "regime_filtered": outdf.loc[last, "regime_filtered"],
        "regime_smoothed": outdf.loc[last, "regime_smoothed"],
        "regime_viterbi": outdf.loc[last, "regime_viterbi"],
        "filtered_probs": {r: round(float(F[-1, k]), 4) for k, r in enumerate(REGS)},
        "entropy_filtered": round(float(outdf.loc[last, "entropy_filtered"]), 4),
        "pred_sd": round(float(pred_sd[-1]), 6),
        "p_dip21": round(float(p_dip[-1]), 4),
        "dip_heads": {r: round(float(q[k]), 4) for k, r in enumerate(REGS)},
        "features_last": {f: round(float(outdf.loc[last, f]), 5) for f in feats},
        "expected_duration_days": {k: round(v, 1) for k, v in durations.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    if not np.all(np.isfinite(F)) or not (0.002 <= pred_sd[-1] <= 0.15):
        print("SANITY CHECK FAILED: non-finite probabilities or implausible forecast", file=sys.stderr)
        return 3

    print(f"\n=== {last.date()}  S&P {summary['close']:,.2f}  VIX {summary['vix']:.2f}")
    print(f"  mood: {summary['regime_filtered']} (p={F[-1].max():.3f}, entropy {summary['entropy_filtered']:.3f})")
    print(f"  forecast daily move +-{pred_sd[-1]*100:.2f}%   chance of a 5% fall within a month {p_dip[-1]:.1%}")
    for r in REGS:
        print(f"    {r:<18} {summary['filtered_probs'][r]:6.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
