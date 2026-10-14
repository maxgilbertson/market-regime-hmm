"""
Display-only market signals for the page's "Other signals" section. The model does NOT use them:
they were tested as model inputs in October 2026 and did not make it more accurate
(docs/upgrade-2026-10/RESULTS.md), but they are useful context.

    credit      Moody's Baa corporate bond yield minus the 10-year Treasury yield (FRED BAA10Y), % points
    vix_curve   VIX / 3-month VIX; above 1 means traders fear the next month more than the months after
    small_caps  Russell 2000 minus S&P 500, % change over the past 63 trading days

Writes out/context.json. Each signal is independent: one that fails to download is left out, and a
failure never stops the model run. Raw downloads are cached in out/context_cache/ (FRED is refreshed
at most every 6 hours; it only updates once a day, a day or two after the date it reports).
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("out")
CACHE = OUT / "context_cache"


def _yahoo(sym: str, start: str) -> pd.Series:
    import yfinance as yf

    for attempt in range(3):
        try:
            s = yf.Ticker(sym).history(start=start, auto_adjust=True)["Close"]
            if len(s) > 200:
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                return s[~s.index.duplicated(keep="last")].dropna()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2 + 3 * attempt)
    raise RuntimeError(f"no data for {sym}")


def _fred(sid: str, max_age_h: float = 6) -> pd.Series:
    path = CACHE / f"{sid}.csv"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_h * 3600:
        txt = path.read_text()
    else:
        req = urllib.request.Request(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                                     headers={"User-Agent": "market-regime-hmm (github.com/maxgilbertson)"})
        txt = urllib.request.urlopen(req, timeout=30).read().decode()
        if sid not in txt.splitlines()[0]:
            raise RuntimeError(f"unexpected FRED response for {sid}")
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(txt)
    d = pd.read_csv(io.StringIO(txt), na_values=".")
    s = pd.Series(pd.to_numeric(d.iloc[:, 1], errors="coerce").values, index=pd.to_datetime(d.iloc[:, 0]))
    return s.dropna()


def describe(s: pd.Series, key: str, label: str, note: str, unit: str, dp: int, stress: str, source: str) -> dict:
    """Today's value, its usual range over the whole history, and the last year for a mini chart.
    stress = 'high' if a higher value means more market stress, 'low' if a lower value does."""
    s = s.dropna()
    v = float(s.iloc[-1])
    q = s.quantile([0.02, 0.30, 0.70, 0.98])
    yr = s.loc[s.index[-1] - pd.DateOffset(years=1):]
    spark = yr.iloc[::-5][::-1]                       # weekly, ending today
    return {
        "key": key, "label": label, "note": note, "unit": unit, "dp": dp, "stress": stress, "source": source,
        "as_of": f"{s.index[-1]:%Y-%m-%d}", "since": f"{s.index[0]:%Y}",
        "value": round(v, 4), "p2": round(float(q[0.02]), 4), "p30": round(float(q[0.30]), 4),
        "p70": round(float(q[0.70]), 4), "p98": round(float(q[0.98]), 4),
        "pct": round(float((s < v).mean()), 3),
        "yr_lo": round(float(yr.min()), 4), "yr_hi": round(float(yr.max()), 4),
        "spark": [round(float(x), 4) for x in spark.values],
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    signals, errors = [], []

    try:
        baa = _fred("BAA10Y")
        signals.append(describe(baa, "credit", "Credit stress",
                                "extra yield on riskier company bonds over US Treasuries",
                                "pp", 2, "high", "FRED BAA10Y (Moody's Baa minus 10-year Treasury)"))
    except Exception as e:  # noqa: BLE001
        errors.append(f"credit: {e}")

    try:
        v, v3 = _yahoo("^VIX", "2006-07-01"), _yahoo("^VIX3M", "2006-07-01")
        spx = _yahoo("^GSPC", "2006-07-01")
        ratio = (v / v3).reindex(spx.index).dropna()   # S&P trading days only
        signals.append(describe(ratio, "vix_curve", "Near-term fear",
                                "VIX ÷ 3-month VIX; above 1 = more fear about the next month than later",
                                "ratio", 2, "high", "Cboe VIX and VIX3M via Yahoo Finance"))
    except Exception as e:  # noqa: BLE001
        errors.append(f"vix_curve: {e}")

    try:
        rut, spx = _yahoo("^RUT", "1987-09-01"), _yahoo("^GSPC", "1987-09-01")
        both = pd.concat({"rut": rut, "spx": spx}, axis=1, sort=True).dropna()
        rel = (np.log(both["rut"]).diff(63) - np.log(both["spx"]).diff(63)) * 100
        signals.append(describe(rel, "small_caps", "Small caps vs the S&P 500",
                                "Russell 2000 minus S&P 500, past 3 months",
                                "pct_pts", 1, "low", "Russell 2000 and S&P 500 via Yahoo Finance"))
    except Exception as e:  # noqa: BLE001
        errors.append(f"small_caps: {e}")

    (OUT / "context.json").write_text(json.dumps({
        "updated_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "used_by_model": False, "signals": signals, "errors": errors}, indent=1))
    print(f"context.json: {len(signals)} signals" + (f"; skipped {errors}" if errors else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
