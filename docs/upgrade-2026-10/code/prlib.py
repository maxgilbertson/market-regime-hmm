"""
Shared library for the pre-registered model-v2 evaluation (wf1/prereg/PREREG.md, recipes.json).

* Data: cached files only. load_raw() hard-truncates every series at 2018-12-31 BEFORE any feature
  or target is built. A later cut needs the key minted by unlock_holdout(), which only
  run_holdout.py may call and which first checks decision.json against the files on disk.
* Fitting: hmm_regime.fit_hmm (verbatim copy). Filtering / smoothing: shmm.forward_tempered / smooth.
  Heads, hindsight truth, Brier: evalkit. Stability, timeliness, label sanity, calibration: diagnostics.
  Point-in-time BAA10Y: availability.align.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import logsumexp  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent            # wf1/prereg
WF1 = HERE.parent
SCR = WF1.parent
for _sub in ("baseline", "propose-structure-first", "propose-evaluation-first", "data-auditor"):
    if str(WF1 / _sub) not in sys.path:
        sys.path.append(str(WF1 / _sub))

import hmm_regime_copy as M  # noqa: E402
import shmm  # noqa: E402
import evalkit as EK  # noqa: E402
import diagnostics as D  # noqa: E402
from availability import RULES, align  # noqa: E402

PREREG = HERE / "PREREG.md"
PREREG_SHA = "8ae42ec6a4acf149e36f175fc49ea1d91b19cb60cd9c60209e72b678931d41f2"
RECIPES_PATH = HERE / "recipes.json"
CFG = json.loads(RECIPES_PATH.read_text())
MC = CFG["metrics"]
REGIMES = CFG["regimes"]
assert REGIMES == [r.value for r in M.REGIME_ORDER]
RIDS = list(CFG["recipes"])
DEV_CUT = pd.Timestamp(CFG["dev"]["cut"])
SQ252 = np.sqrt(252.0)
PF = [f"pf_{r}" for r in REGIMES]
PS = [f"ps_{r}" for r in REGIMES]

EVAL_SCRIPTS = [HERE / "prlib.py", HERE / "run_dev.py", HERE / "select.py", HERE / "run_holdout.py",
                WF1 / "baseline" / "hmm_regime_copy.py", WF1 / "baseline" / "diagnostics.py",
                WF1 / "baseline" / "harness.py", WF1 / "propose-structure-first" / "shmm.py",
                WF1 / "propose-evaluation-first" / "evalkit.py", WF1 / "data-auditor" / "availability.py",
                SCR / CFG["data"]["bridge"]]


# --------------------------------------------------------------------------- hashing / json
def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(p) -> str:
    return Path(p).resolve().relative_to(WF1).as_posix()


def clean(o):
    """JSON-safe copy: numpy -> python, NaN -> None, timestamps -> ISO dates."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (pd.Timestamp, _dt.date)):
        return None if pd.isna(o) else str(pd.Timestamp(o).date())
    if isinstance(o, pd.DataFrame):
        return clean(o.reset_index().to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return clean(o.to_dict())
    if o is None or isinstance(o, (str, int)):
        return o
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def dump(obj, path):
    Path(path).write_text(json.dumps(clean(obj), indent=1))


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc_repo() -> str:
    """Timestamp in the repo's model/hmm_params.json style ('2026-09-29 10:37:15+00:00')."""
    return str(_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0))


# --------------------------------------------------------------------------- holdout lock
_HOLDOUT_KEY = None


def verify_decision(decision: dict, accept_changed: tuple = ()) -> None:
    """Raise PermissionError unless decision.json is a complete, non-smoke selection naming c*, and
    every hashed file (PREREG.md, recipes.json, every evaluation script, dev outputs) is unchanged
    (paths in accept_changed excepted: run_holdout passes those only after logging a fix diff)."""
    if decision.get("smoke") or not decision.get("complete"):
        raise PermissionError("decision.json is a smoke / incomplete selection")
    dev_meta = decision.get("dev_results", {}).get("meta", {})
    if (decision.get("dev_dir") != rel(HERE) or not dev_meta.get("complete")
            or rel(HERE / "dev_results.json") not in decision.get("hashes", {})):
        raise PermissionError("decision.json is not built on the complete prereg/dev_results.json grid")
    if not decision.get("c_star"):
        raise PermissionError("decision.json selects no candidate (E empty): nothing goes to the holdout")
    if sha256(PREREG) != PREREG_SHA or decision.get("hashes", {}).get(rel(PREREG)) != PREREG_SHA:
        raise PermissionError("PREREG.md hash mismatch")
    need = {rel(p) for p in EVAL_SCRIPTS} | {rel(RECIPES_PATH)}
    missing = sorted(need - set(decision["hashes"]))
    if missing:
        raise PermissionError(f"decision.json does not hash: {missing}")
    bad = [p for p, h in decision["hashes"].items()
           if p not in accept_changed and (not (WF1 / p).exists() or sha256(WF1 / p) != h)]
    if bad:
        raise PermissionError(f"files changed since decision.json: {bad}")


def unlock_holdout(decision: dict, accept_changed: tuple = ()) -> object:
    """Mint the key that lets load_raw() read 2019+ (only from run_holdout.py, only after
    verify_decision passes)."""
    global _HOLDOUT_KEY
    if Path(sys.argv[0]).name != "run_holdout.py":
        raise PermissionError("only run_holdout.py may unlock the holdout")
    verify_decision(decision, accept_changed)
    _HOLDOUT_KEY = object()
    return _HOLDOUT_KEY


# --------------------------------------------------------------------------- data
def _read(relpath, col="Close") -> pd.Series:
    d = pd.read_csv(SCR / relpath, index_col=0, parse_dates=True)
    d.index = pd.to_datetime(d.index).normalize()
    d = d[~d.index.duplicated(keep="last")]
    return pd.to_numeric(d[col] if col else d.iloc[:, 0], errors="coerce")


def load_raw(cut=DEV_CUT, key=None) -> pd.DataFrame:
    """S&P calendar (master) from 1985-01-02: close, real VIX (joined, never filled), bridged VIX,
    point-in-time BAA10Y and its age, S&P log return. Everything is truncated at `cut` first."""
    cut = None if cut is None else pd.Timestamp(cut)
    if cut is None or cut > DEV_CUT:
        if key is None or key is not _HOLDOUT_KEY:
            raise PermissionError("data after 2018-12-31 is reserved for run_holdout.py")
    dc = CFG["data"]
    spx = _read(dc["spx"]).loc[dc["spx_start"]:].dropna()
    vix = _read(dc["vix"]).dropna()
    baa = _read(dc["baa"], col=None).dropna()
    if cut is not None:
        spx, vix, baa = spx.loc[:cut], vix.loc[:cut], baa.loc[:cut]
    cal = spx.index
    raw = pd.DataFrame({"close": spx, "vix": vix.reindex(cal)}, index=cal)
    br = pd.read_csv(SCR / dc["bridge"], index_col=0, parse_dates=True)[dc["bridge_col"]]
    br.index = pd.to_datetime(br.index).normalize()
    raw["vix_bridge"] = br.loc[: dc["bridge_until"]].reindex(cal).combine_first(raw["vix"])
    b, age = align(baa, cal, RULES[dc["baa_rule"]])
    raw["baa"], raw["baa_age"] = b.values, age.values
    raw["ret"] = np.log(raw["close"]).diff()
    return raw


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def cache_last_complete_day() -> pd.Timestamp:
    """Date of the last complete ^GSPC row in the cache (dates only; no values are read)."""
    p = SCR / CFG["data"]["spx"]
    lines = p.read_text().strip().splitlines()
    last, prev = (pd.Timestamp(x.split(",")[0]) for x in (lines[-1], lines[-2]))
    fetched = pd.Timestamp(_dt.datetime.fromtimestamp(p.stat().st_mtime)).normalize()
    return last if last < fetched else prev


def features(raw: pd.DataFrame, rid: str) -> pd.DataFrame:
    """Recipe feature table (PREREG section 2); rows with any missing input dropped; 'sig' = V/100/sqrt(252)."""
    r = CFG["recipes"][rid]
    if r["history"] == "legacy":       # B0: inner join from 1990-01-01 BEFORE features (live load_data)
        px = raw[["close", "vix"]].dropna().loc[CFG["data"]["legacy_start"]:]
        df = M.build_features(px)
    else:                              # S&P features on the full S&P calendar, then join VIX
        df = raw[["close"]].copy()
        lc = np.log(df["close"])
        df["ret_1d"] = lc.diff(1)
        df["ret_1w"] = lc.diff(5)
        df["ret_1m"] = lc.diff(21)
        df["ret_1y"] = lc.diff(252)
        df["trend_mom"] = lc - np.log(df["close"].rolling(200).mean())
        df["log_rv21"] = np.log(df["ret_1d"].rolling(21).std() * SQ252 + 1e-8)
        df["vix"] = raw["vix_bridge"] if r["bridge"] else raw["vix"]
        df["log_vix"] = np.log(df["vix"])
        df["z_1d"] = df["ret_1d"] / (df["vix"].shift(1) / 100 / SQ252)   # previous S&P day's VIX
        if "baa_chg21" in r["features"]:
            df["baa_chg21"] = raw["baa"] - raw["baa"].shift(21)
            df["baa_age"] = raw["baa_age"]
        df = df.dropna(subset=r["features"] + ["vix"])
    df["sig"] = df["vix"] / 100 / SQ252
    return df


def targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Per S&P origin t: r_{t+1}, mean r^2 over t+1..t+21, 5% dip within 21 days, dates of t+1 and t+21."""
    H = MC["fwd_horizon"]
    c, r = raw["close"], raw["ret"]
    T = pd.DataFrame(index=raw.index)
    T["r_next"] = r.shift(-1)
    T["y21"] = (r ** 2)[::-1].rolling(H).mean()[::-1].shift(-1)
    fmin = c[::-1].rolling(H).min()[::-1].shift(-1)
    T["dip"] = ((fmin / c - 1) <= MC["dip_threshold"]).astype(float).where(fmin.notna())
    dates = pd.Series(raw.index, index=raw.index)
    T["d1"], T["d21"] = dates.shift(-1), dates.shift(-H)
    return T


def common_origins(raw: pd.DataFrame, start, end, rids=None) -> pd.DatetimeIndex:
    """Origins scored for every recipe in `rids` (default all six): rows present in all of them,
    start <= t, next S&P day <= end."""
    idx = raw.index
    for rid in (rids or RIDS):
        idx = idx.intersection(features(raw, rid).index)
    nxt = pd.Series(raw.index, index=raw.index).shift(-1).reindex(idx)
    keep = (idx >= pd.Timestamp(start)) & (nxt <= pd.Timestamp(end)).values
    return idx[keep]


# --------------------------------------------------------------------------- fitting
def zmat(df, feats, mu, sd, clip):
    Z = ((df[feats] - mu) / sd).values
    return np.clip(Z, -clip, clip) if clip else Z


def anchor_labels(train: pd.DataFrame, vix_med: float) -> np.ndarray:
    """Sign anchors (live rule) with an explicit VIX median: k = 2*direction + volatile."""
    bull = (train["trend_mom"] > 0) & (train["ret_1m"] > 0)
    bear = (train["trend_mom"] < 0) & (train["ret_1m"] < 0)
    d = np.where(bull, 0, np.where(bear, 2, 1))
    return 2 * d + (train["log_vix"] > vix_med).values.astype(int)


def refit_gate(means, mu, sd, feats) -> dict:
    means, mu, sd = np.asarray(means), np.asarray(mu), np.asarray(sd)
    raw = means * sd + mu
    mp = M.label_states(SimpleNamespace(means_=means[:, :7]), mu[:7], sd[:7])
    i_tm, i_1m, i_v = feats.index("trend_mom"), feats.index("ret_1m"), feats.index("log_vix")
    g = {"rank_ok": all(mp[k] == M.REGIME_ORDER[k] for k in range(6)),
         "bull_sign_ok": all(raw[k, i_tm] > 0 and raw[k, i_1m] > 0 for k in (0, 1)),
         "bear_sign_ok": all(raw[k, i_tm] < 0 and raw[k, i_1m] < 0 for k in (4, 5)),
         "vol_twin_ok": all(raw[k, i_v] > raw[k - 1, i_v] for k in (1, 3, 5))}
    g = {k: bool(v) for k, v in g.items()}
    g["pass"] = all(g.values())
    return g


def fit_one(raw: pd.DataFrame, rid: str, year: int, seeds: int | None = None) -> dict:
    """Fit Y: recipe rows from its first row through (Y-1)-12-31. Returns a params blob shaped like
    the repo's model/hmm_params.json plus recipe metadata."""
    r = CFG["recipes"][rid]
    feats = r["features"]
    seeds = int(seeds or CFG["common"]["seeds"])
    df = features(raw, rid)
    te = pd.Timestamp(f"{year - 1}-12-31")
    train = df.loc[:te]
    sr = train.loc[r["scale_from"]:] if r["scale_from"] else train
    mu, sd = sr[feats].mean().values, sr[feats].std().values
    X = zmat(train, feats, mu, sd, r["clip"])
    vmed = float(sr["log_vix"].median())
    anc = anchor_labels(train, vmed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model, ll = M.fit_hmm(X, anc, seeds, mean_weight=CFG["common"]["mean_weight"])
    return {
        "config": {"ticker": "^GSPC", "start": CFG["data"]["legacy_start"] if r["history"] == "legacy"
                   else str(df.index[0].date()), "train_end": str(te.date()), "seeds": seeds,
                   "anchor": CFG["common"]["anchor_rule"], "mean_weight": CFG["common"]["mean_weight"],
                   "features": feats},
        "fitted_at_utc": now_utc_repo(),
        "feature_mean": mu.tolist(), "feature_std": sd.tolist(),
        "startprob": model.startprob_.tolist(), "transmat": model.transmat_.tolist(),
        "means": model.means_.tolist(), "covars": model.covars_.tolist(),
        "recipe": {"id": rid, **r, "recipes_sha256": sha256(RECIPES_PATH), "prereg_sha256": PREREG_SHA},
        "meta": {"fit_year": year, "first_row": str(df.index[0].date()), "train_last_row": str(train.index[-1].date()),
                 "n_train": int(len(train)), "train_loglik_per_obs": float(ll / len(X)),
                 "vix_median_log": vmed, "scale_rows_from": str(sr.index[0].date()),
                 "anchor_sizes": {REGIMES[k]: int((anc == k).sum()) for k in range(6)},
                 "gate": refit_gate(model.means_, mu, sd, feats), "seeds": seeds,
                 "fit_log": buf.getvalue()[-2000:]},
    }


def params(blob):
    return {k: np.asarray(blob[k], float) for k in
            ("startprob", "transmat", "means", "covars", "feature_mean", "feature_std")}


# --------------------------------------------------------------------------- inference
def log_emission(blob, df, rid) -> np.ndarray:
    r = CFG["recipes"][rid]
    feats = r["features"]
    p = params(blob)
    X = zmat(df, feats, p["feature_mean"], p["feature_std"], r["clip"])
    logB = M.emission_logprob(SimpleNamespace(n_components=6, means_=p["means"], covars_=p["covars"]), X)
    if r.get("baa_stale_days") is not None:        # C5 staleness rule: drop the baa dimension
        stale = (df["baa_age"] > r["baa_stale_days"]).values
        if stale.any():
            keep = [i for i, f in enumerate(feats) if f != "baa_chg21"]
            sub = SimpleNamespace(n_components=6, means_=p["means"][:, keep],
                                  covars_=p["covars"][:, keep][:, :, keep])
            logB[stale] = M.emission_logprob(sub, X[stale][:, keep])
    return logB


def _dip_heads(F, Dv, ok):
    m = ok & np.isfinite(Dv)
    w = MC["dip_prior_weight"]
    qbar = Dv[m].mean()
    return (F[m].T @ Dv[m] + w * qbar) / (F[m].sum(axis=0) + w)


def _t_heads(P, x, ok):
    m = ok & np.isfinite(x)
    pooled = EK.fit_t_weighted(x[m], np.ones(m.sum()))
    out = [pooled if P[m, k].sum() < MC["heads_pooled_below_weight"] else EK.fit_t_weighted(x[m], P[m, k])
           for k in range(6)]
    return np.array(out, float)


def year_block(blob, df, rid, T, year, end, light=False) -> pd.DataFrame:
    """Days of `year` (<= end) under fit `blob`: tempered filter from the recipe's first row, untempered
    smoother over all rows <= end, and the per-origin scores of PREREG sections 5-6."""
    r = CFG["recipes"][rid]
    p = params(blob)
    d = df.loc[: pd.Timestamp(end)]
    logB = log_emission(blob, d, rid)
    pi, A = p["startprob"], p["transmat"]
    F, P = shmm.forward_tempered(logB, pi, A, r["tau"])
    S = shmm.smooth(logB, pi, A)
    rows = d.index.year == year
    out = pd.DataFrame(np.hstack([F[rows], S[rows]]), index=d.index[rows], columns=PF + PS)
    out["model_year"] = int(blob["meta"]["fit_year"])
    if light:
        return out
    te = pd.Timestamp(blob["config"]["train_end"])
    Tt = T.reindex(d.index)
    tr = (d.index <= te)
    q = _dip_heads(F[tr], Tt["dip"].values[tr], (Tt["d21"] <= te).values[tr])
    th = _t_heads(P[tr], Tt["r_next"].values[tr], (Tt["d1"] <= te).values[tr])
    mu, sd = p["feature_mean"][0], p["feature_std"][0]
    m0 = p["means"][:, 0] * sd + mu
    v0 = p["covars"][:, 0, 0] * sd ** 2
    n = int(rows.sum())
    if r["features"][0] == "z_1d":                 # sigma_t fixed at the origin
        s = d["sig"].values[rows][:, None]
        Mk, Vk = s * m0, s ** 2 * v0
    else:
        Mk, Vk = np.tile(m0, (n, 1)), np.tile(v0, (n, 1))
    Py, Fy = P[rows], F[rows]
    x = Tt["r_next"].values[rows]
    logP = np.log(np.clip(Py, 1e-300, 1))
    out["ls1"] = logsumexp(logP + D.norm_logpdf(x[:, None], Mk, Vk), axis=1)
    m2 = Vk + Mk ** 2
    acc, Pj = np.zeros(n), Py.copy()
    for _ in range(MC["fwd_horizon"]):
        acc += (Pj * m2).sum(axis=1)
        Pj = Pj @ A
    out["h21"] = acc / MC["fwd_horizon"]
    out["y21"] = Tt["y21"].values[rows]
    out["dip"] = Tt["dip"].values[rows]
    out["pdip"] = Fy @ q
    lt = np.stack([EK.t_logpdf(x, *th[k]) for k in range(6)], axis=1)
    out["heads_ls"] = logsumexp(logP + lt, axis=1)
    out["pit"] = (Py * norm.cdf((x[:, None] - Mk) / np.sqrt(Vk))).sum(axis=1)
    out["r_next"] = x
    out.loc[~np.isfinite(x), ["ls1", "heads_ls", "pit"]] = np.nan
    return out


def walk_forward(raw, rid, fits: dict, years, end, light=False) -> pd.DataFrame:
    """Stitch year blocks; a year whose fit is missing (fit raised) reuses the previous year's fit."""
    df, T = features(raw, rid), targets(raw)
    parts, prev = [], None
    for y in years:
        blob = fits.get(y) or prev
        if blob is None:
            continue
        b = year_block(blob, df, rid, T, y, end, light=light)
        b["fallback"] = fits.get(y) is None
        parts.append(b)
        prev = blob
    return pd.concat(parts)


# --------------------------------------------------------------------------- metrics
def block_boot(d) -> dict:
    """Circular moving-block bootstrap of the mean (63-day blocks, 5000 resamples, fresh rng 20261012)."""
    cfg = MC["bootstrap"]
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T, B, n = len(d), cfg["block"], cfg["n"]
    if T == 0:
        return {"n": 0}
    rng = np.random.default_rng(cfg["seed"])
    nb = -(-T // B)
    means = np.empty(n)
    for i0 in range(0, n, 250):
        k = min(250, n - i0)
        st = rng.integers(0, T, size=(k, nb))
        idx = ((st[:, :, None] + np.arange(B)) % T).reshape(k, -1)[:, :T]
        means[i0:i0 + k] = d[idx].mean(axis=1)
    q = np.percentile(means, [2.5, 5, 10, 90, 95, 97.5])
    return {"n": T, "mean": d.mean(), "ci95": [q[0], q[5]], "lb95": q[1], "lb90": q[2],
            "ub90": q[3], "ub95": q[4]}


def ece(pm, agree) -> float:
    bins = MC["calibration_bins"]
    pm, agree = pd.Series(np.asarray(pm, float)), pd.Series(np.asarray(agree, float))
    cat = pd.cut(pm, bins, right=False)
    g = pd.DataFrame({"p": pm, "a": agree, "c": cat}).groupby("c", observed=True)
    t = pd.DataFrame({"p": g["p"].mean(), "a": g["a"].mean(), "n": g.size()})
    return float((t["n"] * (t["p"] - t["a"]).abs()).sum() / t["n"].sum())


def reliability(pm, hit) -> list:
    pm, hit = pd.Series(np.asarray(pm, float)), pd.Series(np.asarray(hit, float))
    cat = pd.cut(pm, MC["calibration_bins"], right=False)
    g = pd.DataFrame({"p": pm, "h": hit, "c": cat.astype(str)}).groupby("c")
    return [{"bin": k, "mean_p": v["p"].mean(), "hit": v["h"].mean(), "days": len(v)} for k, v in g]


def iso_fit(pm, agree) -> dict:
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(**MC["isotonic"]).fit(np.asarray(pm, float), np.asarray(agree, float))
    return {"kind": "isotonic (sklearn IsotonicRegression, increasing, clip, y in [0,1]); apply with "
                    "np.interp(pmax, x, y)", "x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist(),
            "n_pairs": int(len(pm))}


def iso_apply(m: dict, pm) -> np.ndarray:
    return np.interp(np.asarray(pm, float), m["x"], m["y"])


def cal_pairs(daily: pd.DataFrame):
    """(pmax of filtered, 1[filtered argmax == own smoothed argmax])."""
    F, S = daily[PF].values, daily[PS].values
    return F.max(axis=1), (F.argmax(axis=1) == S.argmax(axis=1)).astype(float)


def truth_labels(raw) -> pd.Series:
    sp = pd.DataFrame({"close": raw["close"], "ret_1d": raw["ret"]})
    return EK.truth(sp, vol_median_end=MC["truth_vol_median_end"])


def truth_brier_daily(daily, truth) -> pd.Series:
    tl = truth.reindex(daily.index)
    ok = tl.notna().values
    Y = np.zeros((len(daily), 6))
    Y[np.where(ok)[0], [REGIMES.index(x) for x in tl[ok]]] = 1
    b = ((daily[PF].values - Y) ** 2).sum(axis=1)
    return pd.Series(np.where(ok, b, np.nan), index=daily.index)


def labels(daily) -> pd.Series:
    return pd.Series(np.array(REGIMES)[daily[PF].values.argmax(axis=1)], index=daily.index)


def label_checks(tab: pd.DataFrame) -> dict:
    nmin = MC["label_min_days"]

    def ok(*rs):
        return all(pd.notna(tab.loc[x, "days"]) and tab.loc[x, "days"] >= nmin for x in rs)
    fv, fd = [], []
    vol = tab["fwd21_vol_ann"]
    pairs = {(f"{d} Volatile", f"{d} Quiet") for d in ("Bull", "Sideways", "Bear")}
    pairs |= {("Bear Volatile", o) for o in REGIMES if o != "Bear Volatile"}
    for hi, lo in sorted(pairs):
        if ok(hi, lo) and not vol[hi] > vol[lo]:
            fv.append(f"{hi} fwd21 vol {vol[hi]:.3f} <= {lo} {vol[lo]:.3f}")
    pd5 = tab["p_dd5_within21"]
    for v in ("Quiet", "Volatile"):
        if ok(f"Bear {v}", f"Bull {v}") and pd5[f"Bear {v}"] < pd5[f"Bull {v}"]:
            fd.append(f"Bear {v} P(dip) < Bull {v}")
    for d in ("Bull", "Sideways", "Bear"):
        if ok(f"{d} Quiet", f"{d} Volatile") and pd5[f"{d} Quiet"] > pd5[f"{d} Volatile"]:
            fd.append(f"{d} Quiet P(dip) > {d} Volatile")
    return {"fwdvol_failures": fv, "dip_failures": fd}


def window_metrics(daily, raw, truth, win) -> dict:
    """All section 5-6 metrics on the rows of `daily` (already restricted to scored origins in `win`)."""
    a, b = (pd.Timestamp(x) for x in win)
    a, b = max(a, daily.index[0]), min(b, daily.index[-1])   # episodes only where labels exist
    F, S = daily[PF].set_axis(REGIMES, axis=1), daily[PS].set_axis(REGIMES, axis=1)
    lab = labels(daily)
    pm = F.max(axis=1)
    out = {"n": int(len(daily)), "first": daily.index[0], "last": daily.index[-1],
           "ls1": daily["ls1"].mean(), "ls1_by_year": daily["ls1"].groupby(daily.index.year).mean().to_dict(),
           "heads_ls": daily["heads_ls"].mean()}
    ok = daily["y21"].notna() & (daily["h21"] > 0)
    q = daily["y21"][ok] / daily["h21"][ok]
    out["qlike21"] = float((q - np.log(q) - 1).mean())
    out["spearman21"] = float(spearmanr(np.sqrt(daily["h21"][ok]), np.sqrt(252 * daily["y21"][ok])).correlation)
    okd = daily["dip"].notna()
    out["dip_brier"] = float(((daily["pdip"][okd] - daily["dip"][okd]) ** 2).mean())
    out["dip_base_rate"] = float(daily["dip"][okd].mean())
    hon = EK.regime_brier(F, truth)
    tl = truth.reindex(daily.index)
    hit = (lab == tl)[tl.notna()]
    hon["reliability"] = reliability(pm[tl.notna()], hit)
    out["honesty"] = hon
    pmx, agree = cal_pairs(daily)
    ctab, cal = D.calibration_vs_hindsight(F, S)
    cal.update({"ece_raw": ece(pmx, agree), "share_pmax_gt_0.99": float((pm > 0.99).mean()),
                "table": ctab.set_axis(ctab.index.astype(str))})
    out["calibration"] = cal
    out["stability"] = D.switching(lab)
    eps = [e for e in D.correction_episodes(raw["close"], MC["correction"]["thr"], MC["correction"]["reset"])
           if a <= e["peak"] <= b]
    L = D.drawdown_lags(eps, lab, raw.index) if eps else pd.DataFrame()
    out["timeliness"] = D.lag_summary(L) if len(L) else {"n_episodes": 0}
    out["timeliness_episodes"] = L
    tab, flags = D.label_sanity(lab, pd.DataFrame({"close": raw["close"], "ret_1d": raw["ret"]}))
    out["label"] = {**label_checks(tab), "flags_diagnostics": flags,
                    "shares": lab.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0).to_dict(),
                    "table": tab}
    pit = daily["pit"].dropna()
    out["tails"] = {"below_q01": float((pit < 0.01).mean()), "above_q99": float((pit > 0.99).mean())}
    return out


def paired(cand, base, truth) -> dict:
    """Paired comparisons of candidate vs base on identical rows (both already restricted)."""
    idx = cand.index.intersection(base.index)
    c, b = cand.loc[idx], base.loc[idx]
    okd = c["dip"].notna() & b["dip"].notna()
    bc = (b["pdip"] - b["dip"]) ** 2 - (c["pdip"] - c["dip"]) ** 2
    tb = truth_brier_daily(b, truth) - truth_brier_daily(c, truth)
    lc, lb = labels(c), labels(b)
    yc, yb = c["ls1"].groupby(idx.year).mean(), b["ls1"].groupby(idx.year).mean()
    sc = lc.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0)
    sb = lb.value_counts(normalize=True).reindex(REGIMES, fill_value=0.0)
    return {"dls1": block_boot((c["ls1"] - b["ls1"]).values),
            "dip_brier_base_minus_cand": block_boot(bc[okd].values),
            "truth_brier6_base_minus_cand": block_boot(tb.dropna().values),
            "agree_argmax": float((lc == lb).mean()),
            "share_diff_pp": ((sc - sb) * 100).to_dict(), "max_abs_share_diff_pp": float((sc - sb).abs().max() * 100),
            "year_wins": int((yc > yb).sum()), "years": int(len(yc))}


# --------------------------------------------------------------------------- context benchmarks
def static_daily(raw, end) -> pd.DataFrame:
    """The live saved 1990-2022 calibration (repo model/hmm_params.json, read only), B0 features, tau=1."""
    blob = json.loads((Path(D.H.SAVED_PARAMS)).read_text())
    blob = {**blob, "meta": {"fit_year": 0}}
    df, T = features(raw, "B0"), targets(raw)
    d = df.loc[: pd.Timestamp(end)]
    p = params(blob)
    logB = log_emission(blob, d, "B0")
    F, P = shmm.forward_tempered(logB, p["startprob"], p["transmat"], 1.0)
    m0 = p["means"][:, 0] * p["feature_std"][0] + p["feature_mean"][0]
    v0 = p["covars"][:, 0, 0] * p["feature_std"][0] ** 2
    x = T["r_next"].reindex(d.index).values
    ls = logsumexp(np.log(np.clip(P, 1e-300, 1)) + D.norm_logpdf(x[:, None], m0, v0), axis=1)
    return pd.DataFrame({"static_saved": np.where(np.isfinite(x), ls, np.nan)}, index=d.index)


def bench_year(raw, year, end) -> pd.DataFrame:
    """GARCH(1,1)-t, climatology (Student-t) and the no-HMM anchor rulebook, refit for fit year Y."""
    te = pd.Timestamp(f"{year - 1}-12-31")
    r = raw["ret"].loc["1990-01-02": pd.Timestamp(end)].dropna()
    rtr = r.loc[:te]
    T = targets(raw)
    days = r.index[(r.index.year == year)]
    x = T["r_next"].reindex(days).values
    g = D.fit_garch(rtr.values, "t")
    h = pd.Series(D.garch_var(r.values, g["mu"], g["omega"], g["alpha"], g["beta"], g["h0"]), index=r.index)
    hn = h.shift(-1).reindex(days).values
    out = pd.DataFrame(index=days)
    out["garch_t"] = D.t_logpdf_var(x, g["mu"], hn, g["nu"])
    out["climatology"] = EK.t_logpdf(x, *EK.fit_t_weighted(rtr.values, np.ones(len(rtr))))
    df1 = features(raw, "C1")
    tr = df1.loc[:te]
    lab = pd.Series(anchor_labels(df1, float(tr["log_vix"].median())), index=df1.index)
    xr = T["r_next"].reindex(tr.index)
    okt = (T["d1"].reindex(tr.index) <= te) & xr.notna()
    pooled = EK.fit_t_weighted(xr[okt].values, np.ones(int(okt.sum())))
    heads = {}
    for k in range(6):
        m = okt & (lab.reindex(tr.index) == k)
        heads[k] = EK.fit_t_weighted(xr[m].values, np.ones(int(m.sum()))) \
            if m.sum() >= MC["heads_pooled_below_weight"] else pooled
    lk = lab.reindex(days)
    out["rulebook"] = [EK.t_logpdf(xi, *heads[int(k)]) if pd.notna(k) else np.nan for xi, k in zip(x, lk)]
    return out
