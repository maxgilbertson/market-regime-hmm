"""
One-shot holdout (PREREG sections 9-11). The only script that may read 2019+ data.

Refuses (and writes nothing) unless prereg/decision.json is a complete, non-smoke selection naming c*
whose sha256 hashes match PREREG.md, recipes.json, every evaluation script and the dev outputs on disk,
the dev-era data fingerprint is unchanged, and no holdout verdict exists yet.

Order: verify -> freeze the isotonic map on c*'s 2001-2018 dev pairs (confidence_map_frozen.json)
-> unlock -> load data through decision.holdout.end -> fit c* and B0 for Y = 2019..2026 (16 fits)
-> score 2019-01-02..end -> H1-H6 (H7 deferred to deployment) -> verdict.json + holdout_results.json
-> context benchmarks (after the verdict) -> deployment artifacts for c* in deploy/.

  python run_holdout.py
  python run_holdout.py --fix-reason "crash in X" [--fix-diff fix.diff --accept-changed prereg/run_holdout.py]
      # only to resume a started run that crashed before any holdout metric existed; the reason and the
      # diff go into holdout_log.txt (outcome-unrelated fixes only: paths, dtypes, crashes)
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # select.py lives here: keep this folder LAST on sys.path
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE] + [str(_HERE)]

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import prlib as L  # noqa: E402

CFG = L.CFG
DEC = _HERE / "decision.json"
LOG = _HERE / "holdout_log.txt"
RES = _HERE / "holdout_results.json"
VER = _HERE / "verdict.json"
MAP0 = _HERE / "confidence_map_frozen.json"
HFITS = _HERE / "holdout_fits"
HDAILY = _HERE / "holdout_daily"
DEPLOY = _HERE / "deploy"
SEEDS = CFG["common"]["seeds"]
_LOG_OPEN = False


def _set_root(root: Path, seeds: int, dec_path: Path):
    """Dry run only: send every output to `root` (never prereg/ itself)."""
    global DEC, LOG, RES, VER, MAP0, HFITS, HDAILY, DEPLOY, SEEDS
    DEC = dec_path
    LOG, RES, VER = root / "holdout_log.txt", root / "holdout_results.json", root / "verdict.json"
    MAP0, HFITS = root / "confidence_map_frozen.json", root / "holdout_fits"
    HDAILY, DEPLOY, SEEDS = root / "holdout_daily", root / "deploy", seeds


def log(msg: str):
    line = f"[{L.now_utc()}] {msg}"
    print(line, flush=True)
    if _LOG_OPEN:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def refuse(msg: str):
    print(f"REFUSED: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def _fit_job(raw, rid, year, path, seeds):
    t0 = time.time()
    try:
        blob = L.fit_one(raw, rid, year, seeds)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        L.dump(blob, path)
        return rid, year, "ok", round(time.time() - t0, 1), blob["meta"]["gate"]
    except Exception as e:  # noqa: BLE001
        return rid, year, f"failed: {e!r}", round(time.time() - t0, 1), None


def _eval_job(raw, rid, fits, years, end, light=False):
    return rid, L.walk_forward(raw, rid, fits, years, end, light=light)


def _bench_job(raw, year, end):
    return year, L.bench_year(raw, year, end)


def _reusable(p: Path, year: int) -> bool:
    if not p.exists():
        return False
    b = json.loads(p.read_text())
    return (b["meta"]["seeds"] == SEEDS and b["config"]["train_end"] == f"{year - 1}-12-31"
            and b["recipe"]["recipes_sha256"] == L.sha256(L.RECIPES_PATH)
            and b["recipe"]["prereg_sha256"] == L.PREREG_SHA)


def _le(a, b, extra=0.0):
    """a <= b + extra; None = no lag value (all missed / no episode)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return a is not None
    return a <= b + extra


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def pass_bar(c, Mc, Mb, P, gates_c) -> dict:
    H = CFG["pass_bar"]
    out = {}
    d = P["dls1"]
    out["H1"] = {"dls1_mean": d["mean"], "lb95": d["lb95"], "ci95": d["ci95"],
                 "pass": d["mean"] > H["H1"]["dls1_mean_gt"] and d["lb95"] > H["H1"]["lb95_gt"]}
    db = P["dip_brier_base_minus_cand"]
    out["H2"] = {"qlike21": Mc["qlike21"], "qlike21_b0": Mb["qlike21"], "spearman21": Mc["spearman21"],
                 "spearman21_b0": Mb["spearman21"], "dip_brier_b0_minus_cand_mean": db["mean"],
                 "dip_brier_b0_minus_cand_p90": db["ub90"]}
    out["H2"]["pass"] = (Mc["qlike21"] <= H["H2"]["qlike_ratio_max"] * Mb["qlike21"]
                         and Mc["spearman21"] >= Mb["spearman21"] - H["H2"]["spearman_drop_max"]
                         and db["ub90"] >= 0)
    fv, fdc, fdb = Mc["label"]["fwdvol_failures"], Mc["label"]["dip_failures"], Mb["label"]["dip_failures"]
    out["H3"] = {"gates_passed": sum(gates_c.values()), "gates_total": len(gates_c), "fwdvol_failures": fv,
                 "dip_failures": fdc, "dip_failures_b0": fdb, "agree_b0": P["agree_argmax"]}
    out["H3"]["pass"] = (all(gates_c.values()) and len(fv) <= H["H3"]["max_fwdvol_failures"]
                         and len(fdc) <= len(fdb) and P["agree_argmax"] >= H["H3"]["min_agree_b0"])
    tb = P["truth_brier6_base_minus_cand"]
    out["H4"] = {"brier6": Mc["honesty"]["brier6"], "brier6_b0": Mb["honesty"]["brier6"],
                 "b0_minus_cand_mean": tb["mean"], "b0_minus_cand_p90": tb["ub90"], "pass": tb["ub90"] >= 0}
    sc, sb = Mc["stability"], Mb["stability"]
    out["H5"] = {"switches_per_year": [sc["switches_per_year"], sb["switches_per_year"]],
                 "flipflops_per_year": [sc["flipflops_per_year"], sb["flipflops_per_year"]]}
    out["H5"]["pass"] = (sc["switches_per_year"] <= sb["switches_per_year"] + H["H5"]["switches_extra_max"]
                         and sc["flipflops_per_year"] <= sb["flipflops_per_year"] + H["H5"]["flipflops_extra_max"])
    tc, t0 = Mc["timeliness"], Mb["timeliness"]
    lag, lag0 = _f(tc.get("any_bear_median_days_from_peak")), _f(t0.get("any_bear_median_days_from_peak"))
    mis, mis0 = tc.get("any_bear_missed", 0), t0.get("any_bear_missed", 0)
    out["H6"] = {"any_bear_median_lag": lag, "any_bear_median_lag_b0": lag0, "any_bear_missed": mis,
                 "any_bear_missed_b0": mis0, "n_episodes": tc.get("n_episodes", 0)}
    out["H6"]["pass"] = _le(lag, lag0, H["H6"]["anybear_lag_extra_max"]) and mis <= mis0
    out["H7"] = {"status": H["H7"], "pass": None}
    return out


def freeze_map(dec, c):
    """Isotonic confidence map on c*'s 2001-2018 development pairs, frozen before any 2019+ data is read."""
    p = L.WF1 / dec["dev_dir"] / "dev_daily" / f"{c}.csv.gz"
    dd = pd.read_csv(p, index_col=0, parse_dates=True).loc["2001-01-01":"2018-12-31"]
    pm, ag = L.cal_pairs(dd)
    m = {**L.iso_fit(pm, ag), "recipe": c, "fitted_on": "development pairs 2001-2018 (pmax vs filtered argmax == "
         "own smoothed argmax)", "dev_daily": L.rel(p), "dev_daily_sha256": L.sha256(p),
         "ece_raw_dev": L.ece(pm, ag), "ece_isotonic_dev_insample": L.ece(L.iso_apply(L.iso_fit(pm, ag), pm), ag)}
    if MAP0.exists():
        old = json.loads(MAP0.read_text())
        if old["x"] != L.clean(m["x"]) or old["y"] != L.clean(m["y"]):
            refuse("confidence_map_frozen.json exists and differs from the dev pairs")
        log(f"isotonic map already frozen: {MAP0.name} sha256 {L.sha256(MAP0)}")
        return old
    m["frozen_utc"] = L.now_utc()
    L.dump(m, MAP0)
    log(f"isotonic map frozen BEFORE unlock: {MAP0.name} sha256 {L.sha256(MAP0)} ({m['n_pairs']} dev pairs)")
    return json.loads(MAP0.read_text())


def dry_run(a):
    """Code test only: the whole pipeline on a pseudo-holdout INSIDE the development data (fit Y=2018,
    origins 2018-01-02..2018-12-28) from a smoke decision. Never unlocks; load_raw refuses > 2018-12-31."""
    global _LOG_OPEN
    ddir = Path(a.dry_run_dev) if Path(a.dry_run_dev).is_absolute() else _HERE / a.dry_run_dev
    if ddir.resolve() == _HERE:
        refuse("dry run needs a smoke folder, not prereg/ itself")
    dec = json.loads((ddir / "decision.json").read_text())
    c = dec.get("c_star") or a.dry_run_cstar      # test hook: a smoke slice rarely has an eligible c*
    if not c:
        refuse("smoke decision selects no candidate; pass --dry-run-cstar for the code test")
    root = ddir / "holdout_dryrun"
    root.mkdir(exist_ok=True)
    _set_root(root, int(dec["dev_results"]["meta"]["seeds"]), ddir / "decision.json")
    if VER.exists() and RES.exists() and a.fix_reason:
        _LOG_OPEN = True
        log(f"DRY RUN post-verdict resume test: {a.fix_reason}")
        return resume_post_verdict(dec, (), a, dry_c=c)
    if VER.exists() or RES.exists():
        refuse(f"{root} already holds a dry-run verdict; delete the folder to repeat")
    _LOG_OPEN = True
    log(f"DRY RUN (development data only, not the holdout)  c*={c} (decision: {dec.get('c_star')})  seeds={SEEDS}")
    run(dec, c, pd.Timestamp("2018-01-02"), L.DEV_CUT, [2018], a, (), dry=True)


def main():
    global _LOG_OPEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fix-reason", default=None)
    ap.add_argument("--fix-diff", default=None)
    ap.add_argument("--accept-changed", nargs="*", default=[])
    ap.add_argument("--dry-run-dev", default=None, help="smoke folder: code test on 2018 dev data, no unlock")
    ap.add_argument("--dry-run-cstar", default=None, help="dry run only: candidate to exercise")
    a = ap.parse_args()
    if a.fix_diff:
        a.fix_diff = str(Path(a.fix_diff).resolve())
    os.chdir(L.WF1)   # spawn workers put the cwd on sys.path: keep select.py's folder out of it
    if a.dry_run_dev:
        return dry_run(a)

    # ------------------------------------------------ refusals (nothing is written before these pass)
    if VER.exists() or RES.exists():
        if not (VER.exists() and RES.exists() and a.fix_reason):
            refuse("holdout metrics already exist: the holdout runs once (only the post-verdict steps can be "
                   "resumed, with --fix-reason)")
        dec = json.loads(DEC.read_text())
        accept = tuple(a.accept_changed)
        if accept and not a.fix_diff:
            refuse("--accept-changed needs --fix-diff (the diff is logged)")
        try:
            L.verify_decision(dec, accept)
        except PermissionError as e:
            refuse(str(e))
        _LOG_OPEN = True
        diff = Path(a.fix_diff).read_text(encoding="utf-8") if a.fix_diff else "(no file changed)"
        log(f"RESUME POST-VERDICT steps after outcome-unrelated fix: {a.fix_reason}\naccepted changed files: "
            f"{list(accept)}\n--- diff ---\n{diff}\n--- end diff ---")
        try:
            resume_post_verdict(dec, accept, a)
        except Exception:  # noqa: BLE001
            log("CRASH in post-verdict resume (verdict unchanged):\n" + traceback.format_exc())
            raise
        return log("post-verdict resume done")
    if not DEC.exists():
        refuse("decision.json missing: run select.py on the complete development grid first")
    dec = json.loads(DEC.read_text())
    accept = tuple(a.accept_changed)
    if LOG.exists() and not a.fix_reason:
        refuse("a previous holdout run started (holdout_log.txt exists); resume only with --fix-reason")
    if not LOG.exists() and (a.fix_reason or accept):
        refuse("--fix-reason / --accept-changed are only for resuming a started run")
    if accept and not a.fix_diff:
        refuse("--accept-changed needs --fix-diff (the diff is logged)")
    try:
        L.verify_decision(dec, accept)
    except PermissionError as e:
        refuse(str(e))
    c = dec["c_star"]
    end = pd.Timestamp(dec["holdout"]["end"])
    start = pd.Timestamp(dec["holdout"]["start"])
    if L.fingerprint(L.load_raw()) != dec["dev_data_fingerprint"]:
        refuse("development data (<= 2018-12-31) changed since run_dev.py")

    # ------------------------------------------------ the run starts here
    _LOG_OPEN = True
    if a.fix_reason:
        diff = Path(a.fix_diff).read_text(encoding="utf-8") if a.fix_diff else "(no file changed)"
        log(f"RESUME after outcome-unrelated fix: {a.fix_reason}\naccepted changed files: {list(accept)}\n"
            f"--- diff ---\n{diff}\n--- end diff ---")
    else:
        log(f"HOLDOUT RUN START  c*={c}  B0  end={end.date()}  decision.json sha256 {L.sha256(DEC)}")
    log(f"verified {len(dec['hashes'])} hashes (PREREG {L.PREREG_SHA[:12]}..), dev fingerprint OK; "
        f"cache last complete day now {L.cache_last_complete_day().date()}")
    t0 = time.time()
    y0, y1 = CFG["holdout"]["fit_years"]
    try:
        run(dec, c, start, end, list(range(y0, y1 + 1)), a, accept)
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        log("CRASH (no verdict written):\n" + traceback.format_exc())
        raise
    log(f"done in {time.time() - t0:.0f}s")


def run(dec, c, start, end, years, a, accept, dry=False):
    m0 = freeze_map(dec, c)
    key = None if dry else L.unlock_holdout(dec, accept)
    raw = L.load_raw(cut=end, key=key)
    if raw.index[-1] != end or not np.isfinite(raw["ret"].iloc[-1]):
        raise RuntimeError(f"last complete day {end.date()} missing from the cache (last row {raw.index[-1].date()})")
    same_dev = L.fingerprint(raw.loc[:L.DEV_CUT]) == dec["dev_data_fingerprint"]
    log(f"loaded {raw.index[0].date()}..{raw.index[-1].date()}; dev slice identical to development data: {same_dev}")
    if not same_dev:
        raise RuntimeError("dev-era slice of the holdout load differs from the development data")

    # ------------------------------------------------ fits Y = 2019..2026 for c* and B0
    rids = [c, "B0"]
    paths = {(r, y): HFITS / r / f"{y}.json" for r in rids for y in years}
    todo = sorted([k for k, p in paths.items() if not _reusable(p, k[1])], key=lambda k: -k[1])
    for k in set(paths) - set(todo):
        log(f"fit {k[0]} {k[1]}: reused from the crashed run ({L.sha256(paths[k])[:12]}..)")
    status = {}
    if todo:
        with ProcessPoolExecutor(max_workers=min(a.workers, len(todo))) as ex:
            futs = [ex.submit(_fit_job, raw, r, y, paths[(r, y)], SEEDS) for r, y in todo]
            for f in as_completed(futs):
                r, y, st, sec, gate = f.result()
                status[(r, y)] = st
                log(f"fit {r} {y}: {st} {sec}s gate={gate}")
    fits = {r: {y: json.loads(paths[(r, y)].read_text()) for y in years
                if paths[(r, y)].exists() and status.get((r, y), "ok") == "ok"} for r in rids}
    gates = {r: {y: bool(fits[r][y]["meta"]["gate"]["pass"]) if y in fits[r] else False for y in years}
             for r in rids}

    # ------------------------------------------------ scoring 2019-01-02..end
    common = L.common_origins(raw, start, end, rids=rids)
    daily = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        for f in as_completed([ex.submit(_eval_job, raw, r, fits[r], years, end) for r in rids]):
            r, dd = f.result()
            daily[r] = dd.loc[dd.index.intersection(common)]
    HDAILY.mkdir(exist_ok=True)
    for r, dd in daily.items():
        dd.to_csv(HDAILY / f"{r}.csv.gz", float_format="%.10g")
    truth = L.truth_labels(raw)
    win = (str(start.date()), str(end.date()))
    Mx = {r: L.window_metrics(daily[r], raw, truth, win) for r in rids}
    P = L.paired(daily[c], daily["B0"], truth)
    pm, ag = L.cal_pairs(daily[c])
    cal = {"ece_raw": L.ece(pm, ag), "ece_calibrated_frozen_map": L.ece(L.iso_apply(m0, pm), ag),
           "reliability_calibrated": L.reliability(L.iso_apply(m0, pm), ag),
           "map_sha256": L.sha256(MAP0)}
    H = pass_bar(c, Mx[c], Mx["B0"], P, gates[c])
    ok = all(H[k]["pass"] for k in ("H1", "H2", "H3", "H4", "H5", "H6"))
    verdict = {"written_utc": L.now_utc(), "c_star": c, "verdict": "PASS" if ok else "FAIL",
               "H7": CFG["pass_bar"]["H7"], "checks": {k: H[k]["pass"] for k in H},
               "ships": (f"{c} ships under the deployment plan once H7 (operations) passes" if ok else
                         "live model stays as is (saved 1990-2022 calibration); only the model-independent display "
                         f"fixes ship; {c} runs as a shadow model and needs the full H1-H7 bar on 12 prospective "
                         "months"),
               "holdout": {"start": win[0], "end": win[1], "n_origins": int(len(common)),
                           "first_origin": common[0], "last_origin": common[-1]},
               "decision_sha256": L.sha256(DEC), "prereg_sha256": L.PREREG_SHA}
    results = {"meta": verdict["holdout"] | {"written_utc": L.now_utc(), "c_star": c, "recipes": rids,
                                             "fit_status": {f"{r} {y}": status.get((r, y), "reused") for r, y in paths},
                                             "gates": gates, "map_frozen": m0},
               "pass_bar": H, "recipes": Mx, "c_vs_B0": P, "calibration": cal}
    L.dump(results, RES)
    L.dump(verdict, VER)
    log(f"VERDICT {verdict['verdict']}  " + "  ".join(f"{k}={v}" for k, v in verdict["checks"].items()))

    post_verdict(dec, c, raw, end, years, fits, daily, m0, verdict, results, a)


def post_verdict(dec, c, raw, end, years, fits, daily, m0, verdict, results, a):
    """Context benchmarks and deployment artifacts. Runs after verdict.json exists and never changes it."""
    rids = [c, "B0"]
    common = daily[c].index
    truth = L.truth_labels(raw)
    win = (results["meta"]["start"], results["meta"]["end"])
    # ------------------------------------------------ context only (after the verdict)
    with ProcessPoolExecutor(max_workers=min(a.workers, len(years))) as ex:
        bench = pd.concat([f.result()[1] for f in [ex.submit(_bench_job, raw, y, end) for y in years]]).sort_index()
    bench = bench.join(L.static_daily(raw, end)).reindex(common)
    bench.to_csv(HDAILY / "benchmarks.csv.gz", float_format="%.10g")
    ctx = {}
    sub = CFG["holdout"]["sub_window"]
    for w, (wa, wb) in {"holdout": win, "2023-latest": (sub[0], win[1])}.items():
        if not len(daily[c].loc[wa:wb]):
            continue
        bm = bench.loc[wa:wb].mean().to_dict()
        ctx[w] = {"benchmarks_ls1": bm,
                  "gap": {r: {k: daily[r].loc[wa:wb, "ls1"].mean() - v for k, v in bm.items()} for r in rids}}
        if w != "holdout":
            ctx[w]["recipes"] = {r: L.window_metrics(daily[r].loc[wa:wb], raw, truth, (wa, wb)) for r in rids}
            ctx[w]["c_vs_B0"] = L.paired(daily[c].loc[wa:wb], daily["B0"].loc[wa:wb], truth)
    results["context_after_verdict"] = ctx
    L.dump(results, RES)
    log("context benchmarks written (GARCH-t, climatology, rulebook, static saved, 2023-latest)")

    # ------------------------------------------------ deployment artifacts for c*
    deploy(dec, c, raw, end, fits, daily, m0, results["calibration"], verdict)


def resume_post_verdict(dec, accept, a, dry_c=None):
    """After a crash in the post-verdict steps only: reload the saved fits, daily scores and verdict
    (nothing is refit or rescored) and redo the context benchmarks and deployment artifacts."""
    verdict, results = json.loads(VER.read_text()), json.loads(RES.read_text())
    m0 = json.loads(MAP0.read_text())
    if dry_c:
        c, end, years, key = dry_c, L.DEV_CUT, [2018], None
    else:
        c, end = dec["c_star"], pd.Timestamp(dec["holdout"]["end"])
        y0, y1 = CFG["holdout"]["fit_years"]
        years = list(range(y0, y1 + 1))
        key = L.unlock_holdout(dec, accept)
    raw = L.load_raw(cut=end, key=key)
    fits = {r: {y: json.loads((HFITS / r / f"{y}.json").read_text()) for y in years
                if (HFITS / r / f"{y}.json").exists()} for r in (c, "B0")}
    daily = {r: pd.read_csv(HDAILY / f"{r}.csv.gz", index_col=0, parse_dates=True) for r in (c, "B0")}
    log(f"post-verdict resume: verdict {verdict['verdict']} (unchanged), {sum(map(len, fits.values()))} fits reloaded")
    post_verdict(dec, c, raw, end, years, fits, daily, m0, verdict, results, a)


def deploy(dec, c, raw, end, fits, daily, m0, cal, verdict):
    DEPLOY.mkdir(exist_ok=True)
    dy = CFG["holdout"]["deploy_year"]
    role = "candidate live model (pending H7)" if verdict["verdict"] == "PASS" else "shadow model"
    files = {}

    # confidence map: frozen dev map unless the holdout calibrated ECE is above 0.05 -> refit on 2001-2025
    cmap = {**m0, "source": "frozen on 2001-2018 development pairs"}
    if cal["ece_calibrated_frozen_map"] > CFG["metrics"]["ece_refit_above"]:
        r0, r1 = CFG["metrics"]["ece_refit_years"]
        dd = pd.read_csv(L.WF1 / dec["dev_dir"] / "dev_daily" / f"{c}.csv.gz", index_col=0, parse_dates=True)
        pairs = pd.concat([dd.loc[f"{r0}-01-01":"2018-12-31"], daily[c].loc["2019-01-01":f"{r1}-12-31"]])
        pm, ag = L.cal_pairs(pairs)
        cmap = {**L.iso_fit(pm, ag), "recipe": c, "source": f"refit on {r0}-{r1} pairs (holdout calibrated ECE "
                f"{cal['ece_calibrated_frozen_map']:.4f} > {CFG['metrics']['ece_refit_above']}); display only",
                "frozen_map_sha256": L.sha256(MAP0)}
    cmap["fitted_utc"] = L.now_utc()
    L.dump(cmap, DEPLOY / "confidence_map.json")
    files["confidence_map.json"] = "isotonic pmax -> P(agrees with hindsight); np.interp(pmax, x, y)"

    # Y = deploy-year params for c* (repo hmm_params.json shape + recipe metadata) and B0 shadow params
    for rid, name, what in ((c, f"hmm_params_{c}_{dy}.json", f"{c} fit Y={dy} ({role})"),
                            ("B0", f"b0_shadow_params_{dy}.json", f"B0 fit Y={dy} (shadow / comparison)")):
        yy = max(fits[rid]) if fits[rid] else None
        if yy is None:
            log(f"no {rid} holdout fit available: {name} not written")
            continue
        blob = {**fits[rid][yy], "deployment": {"role": role if rid == c else "shadow", "verdict": verdict["verdict"],
                                                "fit_year": yy, "requested_year": dy,
                                                "confidence_map": "confidence_map.json" if rid == c else None,
                                                "decision_sha256": verdict["decision_sha256"]}}
        L.dump(blob, DEPLOY / name)
        files[name] = what + ("" if yy == dy else f" (fallback: fit {yy})")

    # as-called filtered history 2001..end for c*: dev fits 2001-2018 + holdout fits 2019..
    d0, d1 = CFG["dev"]["fit_years"]
    allfits = {y: json.loads((L.WF1 / dec["dev_dir"] / "dev_fits" / c / f"{y}.json").read_text())
               for y in range(d0, d1 + 1) if (L.WF1 / dec["dev_dir"] / "dev_fits" / c / f"{y}.json").exists()}
    allfits.update(fits[c])
    years = list(range(CFG["holdout"]["wf_history_from"], end.year + 1))
    hist = L.walk_forward(raw, c, allfits, years, end, light=True)
    F = hist[L.PF].values
    out = pd.DataFrame(index=hist.index.rename("Date"))
    out["model_year"] = hist["model_year"]
    out["fallback_fit"] = hist["fallback"]
    out["regime"] = np.array(L.REGIMES)[F.argmax(axis=1)]
    out["p_max"] = F.max(axis=1)
    out["confidence"] = L.iso_apply(cmap, out["p_max"].values)
    for k, rname in enumerate(L.REGIMES):
        out[f"p_{rname}"] = F[:, k]
    out.to_csv(DEPLOY / "wf_history.csv", float_format="%.6f")
    files["wf_history.csv"] = (f"{c} as-called tempered filtered probabilities {out.index[0].date()}.."
                               f"{out.index[-1].date()}, fit Y used on each day of year Y")
    man = {"written_utc": L.now_utc(), "c_star": c, "verdict": verdict["verdict"], "role": role,
           "H7": CFG["pass_bar"]["H7"], "recipe": CFG["recipes"][c],
           "files": {k: {"what": v, "sha256": L.sha256(DEPLOY / k)} for k, v in files.items()}}
    L.dump(man, DEPLOY / "manifest.json")
    log("deployment artifacts: " + ", ".join(f"{k} {man['files'][k]['sha256'][:12]}.." for k in files))


if __name__ == "__main__":
    main()
