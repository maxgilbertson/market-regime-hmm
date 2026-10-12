"""
Development walk-forward (PREREG sections 4-7). Data is hard-truncated at 2018-12-31.
Fits Y = 2001..2018 for B0 and C1-C5 in parallel (dev_fits/<rid>/<Y>.json), scores every recipe on
the identical origin set (dev_daily/<rid>.csv.gz) and writes dev_results.json.

  python run_dev.py                                                    # full grid, 19 workers
  python run_dev.py --recipes B0 C3 --years 2001 2002 --seeds 2 --out smoke   # smoke slice
  python run_dev.py --repro-b0 2001                                    # B0 vs baseline/walkforward.py
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # select.py lives here: keep this folder LAST on sys.path
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE] + [str(_HERE)]

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prlib as L  # noqa: E402

CFG = L.CFG


def _fit_job(raw, rid, year, seeds, path):
    t0 = time.time()
    try:
        blob = L.fit_one(raw, rid, year, seeds)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        L.dump(blob, path)
        return rid, year, "ok", round(time.time() - t0, 1), blob["meta"]["gate"]
    except Exception as e:  # noqa: BLE001
        return rid, year, f"failed: {e!r}", round(time.time() - t0, 1), None


def _eval_job(raw, rid, fits, years, end):
    return rid, L.walk_forward(raw, rid, fits, years, end)


def _bench_job(raw, year, end):
    return year, L.bench_year(raw, year, end)


def _reusable(p: Path, seeds: int) -> bool:
    if not p.exists():
        return False
    b = json.loads(p.read_text())
    return (b["meta"]["seeds"] == seeds and b["recipe"]["recipes_sha256"] == L.sha256(L.RECIPES_PATH)
            and b["recipe"]["prereg_sha256"] == L.PREREG_SHA)


def repro_b0(year: int):
    """B0 fit Y with the full 8 seeds vs baseline/wf_params/<Y>.json and baseline/wf_probs.csv."""
    raw = L.load_raw()
    t0 = time.time()
    mine = L.fit_one(raw, "B0", year)
    base = json.loads((L.WF1 / "baseline" / "wf_params" / f"{year}.json").read_text())
    keys = ("feature_mean", "feature_std", "startprob", "transmat", "means", "covars")
    out = {"year": year, "seeds": mine["meta"]["seeds"], "fit_seconds": round(time.time() - t0, 1),
           "max_abs_param_diff": {k: float(np.max(np.abs(np.asarray(mine[k]) - np.asarray(base[k])))) for k in keys},
           "train_loglik_per_obs": [mine["meta"]["train_loglik_per_obs"], base.get("train_loglik_per_obs")],
           "gate": mine["meta"]["gate"]}
    wf = pd.read_csv(L.WF1 / "baseline" / "wf_probs.csv", index_col=0, parse_dates=True)
    wf = wf[wf["model_year"] == year]
    df, T = L.features(raw, "B0"), L.targets(raw)
    end = f"{year}-12-31"
    a = L.year_block(mine, df, "B0", T, year, end, light=True)
    b = L.year_block({**base, "meta": {"fit_year": year}}, df, "B0", T, year, end, light=True)
    idx = wf.index.intersection(a.index)
    out.update({"days": int(len(idx)), "days_baseline": int(len(wf)),
                "max_abs_filtered_diff_vs_wf_probs": float(np.abs(a.loc[idx, L.PF].values - wf.loc[idx, L.PF].values).max()),
                "max_abs_smoothed_diff_vs_wf_probs": float(np.abs(a.loc[idx, L.PS].values - wf.loc[idx, L.PS].values).max()),
                "max_abs_filtered_diff_baseline_params_this_pipeline":
                    float(np.abs(b.loc[idx, L.PF].values - wf.loc[idx, L.PF].values).max()),
                "argmax_agreement": float((L.labels(a.loc[idx]) == wf.loc[idx, "regime_filtered"]).mean())})
    # LS1: this pipeline vs the baseline's formula on wf_probs' predicted probabilities (same days)
    full = L.year_block(mine, df, "B0", T, year, end)
    x = full.loc[idx, "r_next"].values
    ok = np.isfinite(x)
    m0 = np.asarray(mine["means"])[:, 0] * mine["feature_std"][0] + mine["feature_mean"][0]
    v0 = np.asarray(mine["covars"])[:, 0, 0] * mine["feature_std"][0] ** 2
    pp = wf.loc[idx, [c.replace("pf_", "pp_") for c in L.PF]].values
    ls_base = L.logsumexp(np.log(np.clip(pp, 1e-300, 1)) + L.D.norm_logpdf(x[:, None], m0, v0), axis=1)
    by_year = pd.read_csv(L.WF1 / "baseline" / "logscore_by_year.csv", index_col=0)
    out.update({"ls1_this_pipeline": float(full.loc[idx, "ls1"].values[ok].mean()),
                "ls1_baseline_formula_on_wf_probs": float(ls_base[ok].mean()),
                "ls1_baseline_logscore_by_year_hmm_wf": float(by_year.loc[year, "hmm_wf"]),
                "max_abs_daily_ls1_diff": float(np.abs(full.loc[idx, "ls1"].values[ok] - ls_base[ok]).max())})
    print(json.dumps(L.clean(out), indent=1))
    return out


def main():
    os.chdir(L.WF1)   # spawn workers put the cwd on sys.path: keep select.py's folder out of it
    ap = argparse.ArgumentParser()
    y0, y1 = CFG["dev"]["fit_years"]
    ap.add_argument("--recipes", nargs="+", default=L.RIDS)
    ap.add_argument("--years", nargs="+", type=int, default=list(range(y0, y1 + 1)))
    ap.add_argument("--seeds", type=int, default=CFG["common"]["seeds"])
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--out", default=".")
    ap.add_argument("--repro-b0", type=int, default=None)
    a = ap.parse_args()
    if a.repro_b0:
        return repro_b0(a.repro_b0)
    out = Path(a.out) if Path(a.out).is_absolute() else _HERE / a.out
    rids = [r for r in L.RIDS if r in a.recipes]
    years = sorted(a.years)
    complete = rids == L.RIDS and years == list(range(y0, y1 + 1)) and a.seeds == CFG["common"]["seeds"]
    t0 = time.time()
    raw = L.load_raw()                           # hard-truncated at 2018-12-31
    assert raw.index[-1] <= L.DEV_CUT
    end = L.DEV_CUT

    # ---------------- fits
    paths = {(r, y): out / "dev_fits" / r / f"{y}.json" for r in rids for y in years}
    todo = sorted([k for k, p in paths.items() if not _reusable(p, a.seeds)],
                  key=lambda k: (-k[1], not CFG["recipes"][k[0]]["bridge"]))
    fit_log = {}
    if todo:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(todo))) as ex:
            futs = [ex.submit(_fit_job, raw, r, y, a.seeds, paths[(r, y)]) for r, y in todo]
            for f in as_completed(futs):
                r, y, st, sec, gate = f.result()
                fit_log[(r, y)] = st
                print(f"fit {r} {y}: {st} {sec}s gate={None if gate is None else gate['pass']}", flush=True)
    fits = {r: {} for r in rids}
    for (r, y), p in paths.items():
        if p.exists() and fit_log.get((r, y), "ok") == "ok":
            fits[r][y] = json.loads(p.read_text())
    print(f"fits done {time.time() - t0:.0f}s", flush=True)

    # ---------------- walk-forward scoring (+ context benchmarks)
    common = L.common_origins(raw, CFG["dev"]["score_start"], end)
    common = common[common.year.isin(years)]
    daily, bench = {}, []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(rids) + len(years))) as ex:
        ev = [ex.submit(_eval_job, raw, r, fits[r], years, end) for r in rids]
        bj = [ex.submit(_bench_job, raw, y, end) for y in years]
        for f in as_completed(ev):
            r, dd = f.result()
            daily[r] = dd.loc[dd.index.intersection(common)]
        bench = pd.concat([f.result()[1] for f in bj]).sort_index()
    bench = bench.join(L.static_daily(raw, end)).reindex(common)
    truth = L.truth_labels(raw)
    print(f"scoring done {time.time() - t0:.0f}s", flush=True)

    ddir = out / "dev_daily"
    ddir.mkdir(parents=True, exist_ok=True)
    for r, dd in daily.items():
        dd.to_csv(ddir / f"{r}.csv.gz", float_format="%.10g")
    bench.to_csv(ddir / "benchmarks.csv.gz", float_format="%.10g")

    # ---------------- metrics
    wins = {w: v for w, v in CFG["dev"]["windows"].items() if len(common[(common >= v[0]) & (common <= v[1])])}
    res = {"meta": {"written_utc": L.now_utc(), "complete": complete, "recipes": rids, "years": years,
                    "seeds": a.seeds, "n_origins": int(len(common)), "first_origin": common[0], "last_origin": common[-1],
                    "dev_data_fingerprint": L.fingerprint(raw), "dev_last_row": raw.index[-1],
                    "prereg_sha256": L.sha256(L.PREREG), "recipes_sha256": L.sha256(L.RECIPES_PATH),
                    "script_hashes": {L.rel(p): L.sha256(p) for p in L.EVAL_SCRIPTS
                                      if p.exists() and p.name not in ("select.py", "run_holdout.py")},
                    "windows": wins},
           "recipes": {}, "vs_B0": {}, "parents": {}, "context": {}}
    for r in rids:
        fl = {}
        for y in years:
            b = fits[r].get(y)
            fl[y] = ({"status": "ok", "gate": b["meta"]["gate"], "n_train": b["meta"]["n_train"],
                      "train_loglik_per_obs": b["meta"]["train_loglik_per_obs"],
                      "anchor_sizes": b["meta"]["anchor_sizes"]} if b else
                     {"status": fit_log.get((r, y), "missing"), "gate": {"pass": False}})
        rr = {"fits": fl, "gates_passed": sum(v["gate"]["pass"] for v in fl.values()), "gates_total": len(years),
              "first_row": L.features(raw, r).index[0], "windows": {}}
        for w, (wa, wb) in wins.items():
            rr["windows"][w] = L.window_metrics(daily[r].loc[wa:wb], raw, truth, (wa, wb))
        pm, ag = L.cal_pairs(daily[r])
        rr["windows"][CFG["dev"]["primary_window"]]["calibration"]["ece_isotonic_insample"] = \
            L.ece(L.iso_apply(L.iso_fit(pm, ag), pm), ag)
        res["recipes"][r] = rr
    if "B0" in daily:
        for r in rids:
            if r != "B0":
                res["vs_B0"][r] = {w: L.paired(daily[r].loc[wa:wb], daily["B0"].loc[wa:wb], truth)
                                   for w, (wa, wb) in wins.items()}
    pw = CFG["dev"]["primary_window"]
    for child, plist in CFG["parents"].items():
        for p, *_ in plist:
            if child in daily and p in daily and pw in wins:
                wa, wb = wins[pw]
                res["parents"][f"{child}-{p}"] = L.paired(daily[child].loc[wa:wb], daily[p].loc[wa:wb], truth)["dls1"]
    for w, (wa, wb) in wins.items():
        bm = bench.loc[wa:wb].mean().to_dict()
        res["context"][w] = {"ls1": bm, "gap": {r: {k: res["recipes"][r]["windows"][w]["ls1"] - v for k, v in bm.items()}
                                               for r in rids}}
    L.dump(res, out / "dev_results.json")

    # ---------------- short console summary
    print(f"\n{'rid':4} {'LS1':>8} {'dLS1':>8} {'lb90':>8} {'QLIKE':>6} {'sw/yr':>6} {'ff/yr':>6} {'bearLag':>7} gates")
    for r in rids:
        m = res["recipes"][r]["windows"][pw] if pw in wins else next(iter(res["recipes"][r]["windows"].values()))
        v = res["vs_B0"].get(r, {}).get(pw, {}).get("dls1", {})
        print(f"{r:4} {m['ls1']:8.4f} {v.get('mean', 0):8.4f} {v.get('lb90', 0):8.4f} {m['qlike21']:6.3f} "
              f"{m['stability']['switches_per_year']:6.1f} {m['stability']['flipflops_per_year']:6.1f} "
              f"{m['timeliness'].get('any_bear_median_days_from_peak', float('nan')):7.1f} "
              f"{res['recipes'][r]['gates_passed']}/{len(years)}")
    print(f"wrote {out / 'dev_results.json'} ({time.time() - t0:.0f}s, complete={complete})")


if __name__ == "__main__":
    main()
