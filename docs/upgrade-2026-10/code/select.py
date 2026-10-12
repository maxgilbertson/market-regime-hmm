"""
Deterministic selection (PREREG sections 7-8). Reads dev_results.json (2001-2018 only), applies the
eligibility guards G1-G6 vs B0, takes c* = argmax LS1 over the eligible set, walks down the parents by
the pre-declared margins and writes decision.json with c*, every development metric, the holdout end
date and the sha256 of PREREG.md, recipes.json, every evaluation script and the dev outputs that
run_holdout.py relies on. No 2019+ value is read: the holdout end date comes from the cache's date column.

  python select.py                # prereg/dev_results.json -> prereg/decision.json
  python select.py --dev smoke    # smoke slice -> smoke/decision.json (marked smoke: cannot unlock)
"""
if __name__ != "__main__":
    # This file shares its name with the stdlib 'select' module (socket -> selectors -> select). If it is
    # ever imported under that name, load the real module and hand it over; never run the selection.
    import importlib.machinery as _ilm
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    _d = _os.path.dirname(_os.path.abspath(__file__))
    _spec = _ilm.PathFinder.find_spec(__name__, [p for p in _sys.path if _os.path.abspath(p or ".") != _d])
    if _spec is None:
        raise ImportError(f"{__name__}: stdlib module not found outside {_d}")
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _sys.modules[__name__] = _mod
    globals().update(_mod.__dict__)
else:
    import sys
    from pathlib import Path

    _HERE = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _HERE] + [str(_HERE)]

    import argparse
    import json

    import prlib as L

    CFG = L.CFG

    def _num(x):
        return None if x is None else float(x)

    def _le(a, b, extra=0.0):
        """a <= b + extra, with None meaning 'no value' (no episode called / no episode in window)."""
        if a is None and b is None:
            return True
        if a is None or b is None:
            return a is not None      # candidate has a lag but B0 has none: not worse
        return a <= b + extra

    def guards(res: dict, r: str) -> dict:
        pw = CFG["dev"]["primary_window"]
        G = CFG["guards"]
        rr, rb = res["recipes"][r], res["recipes"]["B0"]
        m, b = rr["windows"][pw], rb["windows"][pw]
        v = res["vs_B0"][r][pw]
        out = {}
        fv = m["label"]["fwdvol_failures"]
        out["G1"] = {"gates_passed": rr["gates_passed"], "gates_total": rr["gates_total"],
                     "fwdvol_failures": len(fv), "agree_b0": v["agree_argmax"],
                     "max_share_dev_pp": v["max_abs_share_diff_pp"]}
        out["G1"]["pass"] = (rr["gates_passed"] == rr["gates_total"] and len(fv) <= G["G1"]["max_fwdvol_failures"]
                             and v["agree_argmax"] >= G["G1"]["min_agree_b0"]
                             and v["max_abs_share_diff_pp"] <= G["G1"]["max_share_dev_pp"])
        db = v["dip_brier_base_minus_cand"]
        out["G2"] = {"qlike21": m["qlike21"], "qlike21_b0": b["qlike21"], "spearman21": m["spearman21"],
                     "spearman21_b0": b["spearman21"], "dip_brier_b0_minus_cand_mean": db["mean"],
                     "dip_brier_b0_minus_cand_p90": db["ub90"]}
        out["G2"]["pass"] = (m["qlike21"] <= G["G2"]["qlike_ratio_max"] * b["qlike21"]
                             and m["spearman21"] >= b["spearman21"] - G["G2"]["spearman_drop_max"]
                             and db["ub90"] >= 0)
        tb = v["truth_brier6_base_minus_cand"]
        out["G3"] = {"brier6": m["honesty"]["brier6"], "brier6_b0": b["honesty"]["brier6"],
                     "b0_minus_cand_mean": tb["mean"], "b0_minus_cand_p90": tb["ub90"], "pass": tb["ub90"] >= 0}
        out["G4"] = {k: [m["stability"][k], b["stability"][k]] for k in G["G4"]["le_b0"]}
        out["G4"]["pass"] = all(m["stability"][k] <= b["stability"][k] for k in G["G4"]["le_b0"])
        tm, tb0 = m["timeliness"], b["timeliness"]
        lag, lag0 = _num(tm.get("any_bear_median_days_from_peak")), _num(tb0.get("any_bear_median_days_from_peak"))
        mis, mis0 = tm.get("any_bear_missed", 0), tb0.get("any_bear_missed", 0)
        out["G5"] = {"any_bear_median_lag": lag, "any_bear_median_lag_b0": lag0, "any_bear_missed": mis,
                     "any_bear_missed_b0": mis0, "n_episodes": tm.get("n_episodes", 0)}
        out["G5"]["pass"] = _le(lag, lag0, G["G5"]["anybear_lag_extra_max"]) and mis <= mis0
        d = v["dls1"]
        out["G6"] = {"dls1_mean": d["mean"], "ci95": d["ci95"], "lb90": d["lb90"], "lb95": d["lb95"],
                     "pass": d["mean"] > G["G6"]["dls1_mean_gt"]}
        out["eligible"] = all(out[g]["pass"] for g in ("G1", "G2", "G3", "G4", "G5", "G6"))
        return out

    def walk_down(c: str, E: list, ls1: dict, parents: dict) -> tuple[str, list]:
        """PREREG 8.3: while c* has an eligible parent p with LS1(c*) - LS1(p) below the margin for the
        component p removes, c* <- p (parents checked in the listed order)."""
        trace = []
        while True:
            moved = False
            for p, margin, need_lb in CFG["parents"].get(c, []):
                if p not in E:
                    trace.append({"from": c, "parent": p, "parent_eligible": False, "step_down": False})
                    continue
                d = ls1[c] - ls1[p]
                lb = parents.get(f"{c}-{p}", {}).get("lb90")
                keep = d >= margin and (not need_lb or (lb is not None and lb > 0))
                trace.append({"from": c, "parent": p, "parent_eligible": True, "dls1": d, "margin": margin,
                              "lb90_required": bool(need_lb), "lb90": lb, "step_down": not keep})
                if not keep:
                    c, moved = p, True
                    break
            if not moved:
                return c, trace

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--dev", default=".", help="folder holding dev_results.json (relative to prereg/)")
        a = ap.parse_args()
        ddir = Path(a.dev) if Path(a.dev).is_absolute() else _HERE / a.dev
        res = json.loads((ddir / "dev_results.json").read_text())
        meta = res["meta"]
        if meta["prereg_sha256"] != L.PREREG_SHA or L.sha256(L.PREREG) != L.PREREG_SHA:
            raise SystemExit("PREREG.md hash mismatch")
        if meta["recipes_sha256"] != L.sha256(L.RECIPES_PATH):
            raise SystemExit("recipes.json changed since run_dev.py: re-run the development grid")
        changed = [p for p, h in meta.get("script_hashes", {}).items() if L.sha256(L.WF1 / p) != h]
        if changed:
            raise SystemExit(f"evaluation code changed since run_dev.py: {changed}")
        smoke = ddir.resolve() != _HERE or not meta["complete"]
        rids = meta["recipes"]
        pw = CFG["dev"]["primary_window"]
        if "B0" not in rids:
            raise SystemExit("B0 missing from dev_results.json")
        ls1 = {r: res["recipes"][r]["windows"][pw]["ls1"] for r in rids}
        G = {r: guards(res, r) for r in rids if r != "B0"}
        E = [r for r in rids if r != "B0" and G[r]["eligible"]]
        c0 = max(E, key=lambda r: ls1[r]) if E else None
        c_star, trace = walk_down(c0, E, ls1, res["parents"]) if c0 else (None, [])

        hashes = {L.rel(p): L.sha256(p) for p in [L.PREREG, L.RECIPES_PATH, *L.EVAL_SCRIPTS] if p.exists()}
        missing = [L.rel(p) for p in L.EVAL_SCRIPTS if not p.exists()]
        dev_files = [ddir / "dev_results.json", *sorted((ddir / "dev_daily").glob("*.csv.gz")),
                     *sorted((ddir / "dev_fits").glob("*/*.json"))]
        hashes.update({L.rel(p): L.sha256(p) for p in dev_files})
        ho = CFG["holdout"]
        dec = {"written_utc": L.now_utc(), "smoke": bool(smoke), "complete": bool(meta["complete"] and not smoke
                                                                                  and not missing),
               "c_star": c_star, "argmax_ls1_eligible": c0, "eligible": E, "walk_down": trace,
               "ls1_2001_2018": ls1, "guards": G,
               "holdout": {"start": ho["start"], "end": L.cache_last_complete_day(),
                           "fit_years": ho["fit_years"], "recipes": ([c_star, "B0"] if c_star else []),
                           "deploy_year": ho["deploy_year"]},
               "dev_dir": L.rel(ddir) if ddir.resolve() != L.WF1 else ".",
               "dev_data_fingerprint": meta["dev_data_fingerprint"], "missing_scripts": missing,
               "hashes": hashes, "dev_results": res}
        out = ddir / "decision.json"
        if out.exists() and not smoke:
            raise SystemExit(f"{out} already exists: the selection is written once")
        L.dump(dec, out)

        print(f"{'rid':4} {'LS1':>8} {'dLS1':>8}  G1 G2 G3 G4 G5 G6  eligible")
        print(f"B0   {ls1['B0']:8.4f}")
        for r in rids:
            if r == "B0":
                continue
            g = G[r]
            flags = "  ".join("y" if g[k]["pass"] else "n" for k in ("G1", "G2", "G3", "G4", "G5", "G6"))
            print(f"{r:4} {ls1[r]:8.4f} {g['G6']['dls1_mean']:8.4f}  {flags}   {g['eligible']}")
        for t in trace:
            print("walk-down:", t)
        print(f"c* = {c_star}  (argmax {c0}; eligible {E}); holdout end {dec['holdout']['end']}; "
              f"smoke={smoke} complete={dec['complete']} missing_scripts={missing}")
        print(f"wrote {out}")

    main()
