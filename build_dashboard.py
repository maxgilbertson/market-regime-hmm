"""Bundle the HMM outputs (out/, model/) into the self-contained regime_dashboard.html.

Reads only saved outputs; it never runs the model.
Optional inputs, used when present:
  model/confidence_map.json  {"x": [raw top prob...], "y": [calibrated...], "since": "2001",
                              "very_sure_hit": 0.93, "method": "isotonic, walk-forward"}
  model/wf_history.csv       Date + "regime" (walk-forward, out-of-sample label per day);
                             the per-mood statistics then use those days only.
  model/track_record.json    facts from the 2019-2026 unseen-data test (shown as "Track record").
  out/context.json           display-only "other signals" (not used by the model); skipped when missing.
--site DIR also writes DIR/index.html and DIR/data/{summary.json, regime_history.csv, data.json, context.json}.
"""
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, MODEL = ROOT / "out", ROOT / "model"
REGS = ["Bull Quiet", "Bull Volatile", "Sideways Quiet", "Sideways Volatile", "Bear Quiet", "Bear Volatile"]
DIR_FAM, VOL_FAM = ["Bull", "Bull", "Sideways", "Sideways", "Bear", "Bear"], ["Quiet", "Volatile"] * 3

# plain wording per model input: label, note, phrase, noun (plural?), transform, unit, signed, dp, (low, high) side words
FEAT = {
    "ret_1d": ("Last day", "S&P 500, latest day", "the last day's move", "the last day's move", "pct", "%", True, 1, ("weak", "strong")),
    "z_1d": ("Last day vs VIX", "in VIX-sized units", "the last day's move sized against the VIX", "the last day's move", "id", "×", True, 1, ("weak", "strong")),
    "ret_1w": ("Past week", "last 5 trading days", "the past week's move", "the past week's move", "pct", "%", True, 1, ("weak", "strong")),
    "ret_1m": ("Past month", "last 21 trading days", "the past month's move", "the past month's move", "pct", "%", True, 1, ("weak", "strong")),
    "ret_1y": ("Past year", "last 252 trading days", "the past year's move", "the past year's move", "pct", "%", True, 0, ("weak", "strong")),
    "trend_mom": ("Trend", "vs its 200-day average", "the trend", "the trend", "pct", "%", True, 1, ("weak", "strong")),
    "log_vix": ("Fear gauge (VIX)", "priced-in swings", "the fear gauge", "the fear gauge", "exp", "", False, 1, ("calm", "high")),
    "log_rv21": ("Actual swings", "past month, yearly rate", "how much prices have swung", "price swings*", "vol", "%", False, 0, ("calm", "high")),
}
DAILY = {"ret_1d", "z_1d"}   # one-day moves: noisy, so worded by size and kept out of the headline
HEAD = ["trend_mom", "log_vix", "ret_1m", "log_rv21", "ret_1y", "ret_1w", "z_1d", "ret_1d"]
TX = {"pct": lambda x: (np.exp(x) - 1) * 100, "vol": lambda x: np.exp(x) * 100, "exp": np.exp, "id": lambda x: x}


def jload(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


h = pd.read_csv(OUT / "regime_history.csv", index_col=0, parse_dates=True)
S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
P = jload(MODEL / "hmm_params.json") or {}
CMAP = jload(MODEL / "confidence_map.json")
cfg = P.get("config", {})
n, asof, d0 = len(h), h.index[-1], h.index[0]
reg = np.array([REGS.index(r) for r in h["regime_filtered"]])
cur = int(reg[-1])
close = h["close"].astype(float)

if "z_1d" not in h and "ret_1d" in h and "vix" in h:   # VIX-scaled move, sigma from the previous day's VIX
    h["z_1d"] = h["ret_1d"] / (h["vix"].shift(1) / 100 / math.sqrt(252))
features = [f for f in (cfg.get("features") or list(FEAT)) if f in h]


def calib(p):
    p = np.asarray(p, float)
    return np.interp(p, CMAP["x"], CMAP["y"]) if CMAP and CMAP.get("x") else p


# today's probabilities, top one calibrated, the rest rescaled to fill the remainder
raw = np.array([float(S["filtered_probs"].get(r, 0)) for r in REGS])
top = int(raw.argmax())
probs = raw.copy()
probs[top] = float(calib(raw[top]))
if raw[top] < 1:
    others = np.arange(6) != top   # rescale the rest to fill the remainder, never above the headline mood
    probs[others] = np.minimum(raw[others] * (1 - probs[top]) / max(1e-12, 1 - raw[top]), probs[top])

# ---------- what usually happened next (walk-forward labels when supplied) ----------
fwd = close.shift(-21) / close - 1
fmin = close[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1) / close - 1
train_end = S.get("train_end") or cfg.get("train_end")
labels = h["regime_filtered"]
period_note = f"live model; {d0:%Y}–{str(train_end)[:4]} in-sample" if train_end else "live model"
wf = None
if (MODEL / "wf_history.csv").exists():
    wf = pd.read_csv(MODEL / "wf_history.csv", index_col=0, parse_dates=True)
    col = next((c for c in ("regime", "regime_filtered", "regime_wf") if c in wf), None)
    if col:
        labels, period_note = wf[col].reindex(h.index), "walk-forward (each year from a model trained only on earlier years)"
    else:
        wf = None
lab = labels.dropna()
ok = fwd.notna() & fmin.notna()


def runs(s):
    """[(regime, length)] for consecutive equal labels."""
    v = s.values
    out, j = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or v[i] != v[j]:
            out.append((v[j], i - j))
            j = i
    return out


rr = runs(lab)
outlook = {}
for k, r in enumerate(REGS):
    m = (labels == r) & ok
    lens = [L for g, L in rr if g == r]
    nxt = [rr[i + 1][0] for i in range(len(rr) - 1) if rr[i][0] == r]
    to = max(set(nxt), key=nxt.count) if nxt else None
    days = int((lab == r).sum())
    outlook[r] = {
        "days": days, "share": round(days / max(1, len(lab)), 4), "rare": days / max(1, len(lab)) < 0.1,
        "spell": round(float(np.median(lens)), 1) if lens else None,
        "move": round(float(fwd[m].abs().median()), 4) if m.sum() > 20 else None,
        "fall5": round(float((fmin[m] <= -0.05).mean()), 4) if m.sum() > 20 else None,
        "to": to, "to_share": round(nxt.count(to) / len(nxt), 3) if to else None,
    }
mall = labels.notna() & ok
outlook_all = {"move": round(float(fwd[mall].abs().median()), 4), "fall5": round(float((fmin[mall] <= -0.05).mean()), 4)}
period = f"{lab.index[0]:%Y}–{lab.index[-1]:%Y}, {period_note}"

# the same figures on days after the training period only (live model, no walk-forward file)
outlook_oos = None
if wf is None and "out_of_sample" in h:
    oos = h["out_of_sample"].astype(str).str.lower().isin(["true", "1"]) & ok
    if oos.sum() >= 120:
        outlook_oos = {"since": f"{h.index[oos.values.argmax()]:%Y}",
                       "all": {"move": round(float(fwd[oos].abs().median()), 4),
                               "fall5": round(float((fmin[oos] <= -0.05).mean()), 4), "days": int(oos.sum())}}
        for r in REGS:
            m = (labels == r) & oos
            if m.sum() >= 60:
                outlook_oos[r] = {"move": round(float(fwd[m].abs().median()), 4),
                                  "fall5": round(float((fmin[m] <= -0.05).mean()), 4), "days": int(m.sum())}

# ---------- what the model sees today ----------
mu = {f: h.groupby(reg)[f].mean() for f in features}
sd = {f: h.groupby(reg)[f].std() for f in features}


def push(f, x):
    """Which family this one reading points to on its own (flat prior, per-mood normal fit)."""
    m, s = mu[f].reindex(range(6)).values, sd[f].reindex(range(6)).values
    if np.isnan(m).any() or np.isnan(s).any() or np.isnan(x):
        return None
    lk = np.exp(-0.5 * ((x - m) / s) ** 2) / s
    if lk.sum() <= 0:
        return None
    p = lk / lk.sum()
    fam, base = AXIS[f]
    tot = {}
    for k in range(6):
        tot[fam[k]] = tot.get(fam[k], 0) + p[k]
    g = max(tot, key=tot.get)
    return {"toward": g, "strength": round(float((tot[g] - base) / (1 - base)), 2), "backs": g in REGS[cur].split()}


def axis_for(f):
    """The split (direction or calm/jittery) this input separates best: share of variance between families."""
    x, best = h[f], None
    for fam, base in ((DIR_FAM, 1 / 3), (VOL_FAM, 1 / 2)):
        g = pd.Series([fam[k] for k in reg], index=h.index)
        eta = (x.groupby(g).transform("mean").var() or 0) / (x.var() or 1)
        if best is None or eta > best[0]:
            best = (eta, (fam, base))
    return best[1]


AXIS = {f: axis_for(f) for f in features}


def level(pct, lo="lower", hi="higher"):
    """Plain verdict for where a value sits in its own history (pct = share of days below it)."""
    return (f"Much {lo} than usual" if pct < 0.1 else f"{lo.capitalize()} than usual" if pct < 0.3 else
            f"A bit {lo} than usual" if pct < 0.45 else "About usual" if pct <= 0.55 else
            f"A bit {hi} than usual" if pct <= 0.7 else f"{hi.capitalize()} than usual" if pct <= 0.9 else
            f"Much {hi} than usual")


def weekly(s):
    """Past year, one point a week, always ending on the latest day."""
    return s.loc[s.index[-1] - pd.DateOffset(years=1):].iloc[::-5][::-1]


readings, NOUN = [], {}
for f in features:
    lab_, note, phrase, noun, tx, unit, signed, dp, sides = FEAT.get(f, (f, "", f, f, "id", "", False, 2, ("low", "high")))
    s = pd.Series(TX[tx](h[f].astype(float).values), index=h.index).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 300:
        continue
    v = float(s.iloc[-1])
    yr = s.loc[asof - pd.DateOffset(years=1):]
    lo1, hi1 = float(yr.min()), float(yr.max())
    pct = float((s.values < v).mean())
    p30, p70 = float(np.percentile(s, 30)), float(np.percentile(s, 70))
    pos = (v - lo1) / (hi1 - lo1) if hi1 > lo1 else 0.5
    if f in DAILY:   # judged by size, worded by direction
        a = float((s.abs().values < abs(v)).mean())
        d = "down" if v < 0 else "up"
        vd = (f"Small {d} day" if a < 0.3 else f"Normal-sized {d} day" if a <= 0.7 else
              f"Bigger {d} day than usual" if a <= 0.95 else f"Unusually big {d} day")
        unusual = a > 0.7
    else:
        if pos >= 0.95 and pct > 0.5:
            vd = "Near its 1-year high"
        elif pos <= 0.05 and pct < 0.5:
            vd = "Near its 1-year low"
        else:
            vd = level(pct)
        unusual = not (0.3 <= pct <= 0.7)
    spark = yr.iloc[::-5][::-1]  # weekly, always ending on today
    q = dp + 1
    NOUN[f] = (noun.rstrip("*"), noun.endswith("*"),
               "about usual" if 0.45 <= pct <= 0.55 else f"on the {sides[0] if pct < 0.5 else sides[1]} side")
    readings.append({
        "key": f, "label": lab_, "note": note, "phrase": phrase, "unit": unit, "signed": signed, "dp": dp,
        "value": round(v, q), "p30": round(p30, q), "p70": round(p70, q), "pct": round(pct, 3),
        "verdict": vd, "unusual": unusual, "push": push(f, float(h[f].iloc[-1])),
        "spark": [round(float(x), q) for x in spark.values],
    })
headline = [k for k in HEAD if k in {r["key"] for r in readings}][:5]

R = {r["key"]: r for r in readings}
pro = sorted([R[k] for k in headline if R[k]["push"] and R[k]["push"]["backs"] and R[k]["push"]["strength"] >= 0.3],
             key=lambda r: -r["push"]["strength"])
con = sorted([R[k] for k in headline if R[k]["push"] and not R[k]["push"]["backs"] and R[k]["push"]["strength"] >= 0.3],
             key=lambda r: -r["push"]["strength"])


def clause(keys):
    """'the fear gauge and price swings are on the calm side' (grouped by side, nouns de-duplicated)."""
    groups = {}
    for k in keys:
        noun, plural, side = NOUN[k]
        g = groups.setdefault(side, [])
        if noun not in [x[0] for x in g]:
            g.append((noun, plural))
    parts = []
    for side, g in groups.items():
        verb = "are" if len(g) > 1 or g[0][1] else "is"
        parts.append(f"{' and '.join(x[0] for x in g)} {verb} {side}")
    return " and ".join(parts) or None


why = clause([r["key"] for r in pro[:2]])
why_against = f"{clause([con[0]['key']])}, which points to {con[0]['push']['toward']}" if con and why else None

# ---------- current spell, detection lag ----------
j = n - 1
while j > 0 and reg[j - 1] == cur:
    j -= 1
lag = None
if "regime_smoothed" in h:
    sm, fl, lags, i = h["regime_smoothed"].values, h["regime_filtered"].values, [], 1
    while i < n:
        if sm[i] != sm[i - 1]:
            e = i
            while e + 1 < n and sm[e + 1] == sm[i]:
                e += 1
            if e - i + 1 >= 10:
                hit = next((t - i for t in range(i, min(e + 1, i + 40)) if fl[t] == sm[i]), None)
                if hit is not None:
                    lags.append(hit)
            i = e + 1
        else:
            i += 1
    if len(lags) >= 10:
        lag = [int(round(x)) for x in np.percentile(lags, [25, 75])]

# ---------- risk ahead: the model's own forecasts (pred_sd, p_dip21), each vs its full history ----------
def rnd(xs, q):
    return [round(float(x), q) if np.isfinite(x) else None for x in xs]


risk = []
if "pred_sd" in h:
    s = (h["pred_sd"].astype(float) * 100).replace([np.inf, -np.inf], np.nan).dropna()
    act = (h["ret_1d"].astype(float).rolling(21).std() * 100) if "ret_1d" in h else None
    if len(s) > 300 and np.isfinite(s.iloc[-1]):
        v, sp = float(s.iloc[-1]), weekly(s)
        pct = float((s.values < v).mean())
        risk.append({
            "key": "move", "label": "Expected daily move", "note": "for a typical day",
            "value": round(v, 2), "p30": round(float(s.quantile(0.3)), 2), "p70": round(float(s.quantile(0.7)), 2),
            "pct": round(pct, 3), "verdict": level(pct, "calmer", "bumpier"), "unusual": pct > 0.7,
            "since": f"{s.index[0]:%Y}", "spark": rnd(sp.values, 3),
            "spark2": rnd(act.reindex(sp.index).values, 3) if act is not None else None,
            "actual": round(float(act.iloc[-1]), 2) if act is not None and np.isfinite(act.iloc[-1]) else None,
        })
if "p_dip21" in h:
    s = h["p_dip21"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) > 300 and np.isfinite(s.iloc[-1]):
        v, sp = float(s.iloc[-1]), weekly(s)
        pct = float((s.values < v).mean())
        risk.append({
            "key": "dip", "label": "Chance of a 5% fall", "note": "within a month",
            "value": round(v, 4), "p30": round(float(s.quantile(0.3)) * 100, 2), "p70": round(float(s.quantile(0.7)) * 100, 2),
            "pct": round(pct, 3), "verdict": level(pct), "unusual": pct > 0.7, "since": f"{s.index[0]:%Y}",
            "spark": rnd(sp.values * 100, 2), "base": round(float((fmin[ok] <= -0.05).mean()), 4),
        })

# ---------- recent mood spells (live model, filtered) ----------
spells, fr, i = [], h["regime_filtered"].values, n - 1
while i >= 0 and len(spells) < 8:
    a = i
    while a > 0 and fr[a - 1] == fr[i]:
        a -= 1
    c0 = close.iloc[a - 1] if a > 0 else close.iloc[a]
    spells.append({"k": REGS.index(fr[i]), "a": f"{h.index[a]:%Y-%m-%d}", "b": f"{h.index[i]:%Y-%m-%d}",
                   "n": int(i - a + 1), "chg": round(float(close.iloc[i] / c0 - 1), 4)})
    i = a - 1

# ---------- track record (2019-2026 unseen-data test) ----------
track = None
TR = jload(MODEL / "track_record.json")
if TR and TR.get("facts"):
    track = {"facts": [{k: f.get(k) for k in ("value", "label", "detail")} for f in TR["facts"] if f.get("value")][:5],
             "caveat": TR.get("caveat"), "record": TR.get("record"), "period": TR.get("period")}

# ---------- other signals: display only, not used by the model ----------
signals = []
try:
    CX = jload(OUT / "context.json") or {}
    for g in CX.get("signals") or []:
        sp = [x for x in (g.get("spark") or []) if x is not None and np.isfinite(x)]
        v, pct = g.get("value"), g.get("pct")
        if v is None or pct is None or not np.isfinite(v) or len(sp) < 2:
            continue
        dp, toward = int(g.get("dp", 2)), (pct > 0.5) == (g.get("stress") == "high")
        vd = level(pct)
        if not 0.3 <= pct <= 0.7:
            vd += ": a stress sign" if toward else ": calm"
        signals.append({
            "key": g.get("key"), "label": g.get("label"), "note": str(g.get("note") or "").split(";")[0],
            "more": str(g.get("note") or "").partition(";")[2].strip(), "source": g.get("source"),
            "unit": {"pp": "pp", "pct_pts": "pp", "ratio": ""}.get(g.get("unit"), g.get("unit") or ""),
            "signed": bool((g.get("p2") or 0) < 0), "dp": dp, "value": round(float(v), dp + 1),
            "p30": g.get("p30"), "p70": g.get("p70"), "pct": round(float(pct), 3), "verdict": vd,
            "unusual": toward and not 0.3 <= pct <= 0.7, "as_of": g.get("as_of"), "since": g.get("since"),
            "spark": [round(float(x), dp + 1) for x in sp],
        })
except (ValueError, TypeError, AttributeError) as e:
    print(f"context.json skipped: {e}")
    signals = []

# ---------- compact, tiered series: daily for 5 years, weekly before ----------
cut = asof - pd.DateOffset(years=5)
old = h.loc[h.index < cut]
tier = pd.concat([old.groupby(old.index.to_period("W")).tail(1), h.loc[h.index >= cut]])
tdays = (tier.index - d0).days.values
ptop = calib(h[[f"pf_{r}" for r in REGS]].max(axis=1).values)
tp = pd.Series(ptop, index=h.index).reindex(tier.index).values
alld = (h.index - d0).days.values
spans = [[int(alld[i]), int(reg[i])] for i in range(n) if i == 0 or reg[i] != reg[i - 1]]
low, unsure, i = ptop < 0.5, [], 0
while i < n:
    if low[i]:
        e = i
        while e + 1 < n and low[e + 1]:
            e += 1
        unsure.append([int(alld[i]), int(alld[e])])
        i = e + 1
    else:
        i += 1

# 'train' describes the model that draws today's mood and the ribbon; the outlook figures may come from elsewhere
train = {"start": f"{d0:%Y-%m-%d}", "end": train_end, "refit": cfg.get("refit", "fixed")}
model_info = {k: (S.get("model") or {}).get(k) for k in ("label", "tau")} if S.get("model") else None
summary = {
    "as_of": S["as_of"], "updated_at_utc": S.get("updated_at_utc"), "session": S.get("session"),
    "close": round(float(S.get("close", close.iloc[-1])), 2),
    "close_chg_1d": round(float(close.iloc[-1] / close.iloc[-2] - 1), 4),
    "vix": round(float(S.get("vix", h["vix"].iloc[-1])), 2),
    "regime": REGS[cur], "probs": {r: round(float(p), 4) for r, p in zip(REGS, probs)},
    "spell_start": f"{h.index[j]:%Y-%m-%d}", "spell_days": int(n - j),
    "why": why, "why_against": why_against, "lag_days": lag, "train": train, "model": model_info,
    "stats_source": "walk-forward" if wf is not None else "live",
    "inputs": list(dict.fromkeys(R[f]["phrase"] for f in features if f in R)),
    "calibration": ({"since": CMAP.get("since"), "hit": CMAP.get("very_sure_hit"), "method": CMAP.get("method")}
                    if CMAP else None),
}
summary["data_version"] = hashlib.sha1(json.dumps([summary["as_of"], summary["updated_at_utc"], summary["probs"]])
                                       .encode()).hexdigest()[:12]
data = {
    "v": 3, "regimes": REGS, "d0": f"{d0:%Y-%m-%d}", "summary": summary,
    "series": {"dd": np.diff(tdays, prepend=0).astype(int).tolist(),
               "c": [int(round(x)) for x in tier["close"].values],
               "v": [int(round(x * 10)) for x in tier["vix"].values],
               "p": [int(round(x * 100)) for x in tp]},
    "spans": spans, "unsure": unsure,
    "outlook": outlook, "outlook_all": outlook_all, "outlook_oos": outlook_oos, "period": period,
    "readings": readings, "headline": headline,
    "risk": risk, "spells": spells, "track": track, "signals": signals,
}

blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
template = (ROOT / "regime_dashboard.template.html").read_text(encoding="utf-8")
assert "/*__DATA__*/null" in template
html = template.replace("/*__DATA__*/null", blob.replace("</", "<\\/"))
(ROOT / "regime_dashboard.html").write_text(html, encoding="utf-8")
print(f"wrote regime_dashboard.html ({len(html.encode()) / 1e3:.0f} KB, data {len(blob.encode()) / 1e3:.0f} KB, "
      f"{len(tier)} rows, {len(spans)} spans)")

if "--site" in sys.argv:
    site = Path(sys.argv[sys.argv.index("--site") + 1])
    (site / "data").mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text(html, encoding="utf-8")
    raw_s = {k: v for k, v in S.items() if "tomorrow" not in k}   # the 'predicted tomorrow' call is not shown
    raw_s["data_version"] = summary["data_version"]
    (site / "data" / "summary.json").write_text(json.dumps(raw_s, indent=2), encoding="utf-8")
    (site / "data" / "data.json").write_text(blob, encoding="utf-8")
    # full history without the 'predicted next' column (that call is not shown); values copied as text
    with open(OUT / "regime_history.csv", newline="", encoding="utf-8") as fi, \
            open(site / "data" / "regime_history.csv", "w", newline="", encoding="utf-8") as fo:
        rd, wr = csv.reader(fi), csv.writer(fo, lineterminator="\n")
        hdr = next(rd)
        keep = [i for i, c in enumerate(hdr) if c != "regime_predicted_next"]
        wr.writerow([hdr[i] for i in keep])
        wr.writerows([row[i] for i in keep] for row in rd)
    for name in ("transition_matrix.csv", "regime_stats.csv", "state_means.csv", "regime_chart.png", "context.json"):
        if (OUT / name).exists():   # keep the older data links working
            shutil.copy2(OUT / name, site / "data" / name)
    (site / ".nojekyll").write_text("")
    print(f"wrote site to {site}/")
