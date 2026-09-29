"""Bundle the HMM outputs in ./out into the self-contained regime_dashboard.html."""
import json
from pathlib import Path

import pandas as pd

OUT = Path("out")
REGS = ["Bull Quiet", "Bull Volatile", "Sideways Quiet", "Sideways Volatile", "Bear Quiet", "Bear Volatile"]

h = pd.read_csv(OUT / "regime_history.csv", index_col=0, parse_dates=True)
summary = json.loads((OUT / "summary.json").read_text())
trans = pd.read_csv(OUT / "transition_matrix.csv", index_col=0).loc[REGS, REGS]
stats = pd.read_csv(OUT / "regime_stats.csv", index_col=0).loc[REGS]
means = pd.read_csv(OUT / "state_means.csv", index_col=0).loc[REGS]

oos_start = int(h["out_of_sample"].values.argmax())
data = {
    "regimes": REGS,
    "dates": [d.strftime("%Y-%m-%d") for d in h.index],
    "close": h["close"].round(2).tolist(),
    "vix": h["vix"].round(2).tolist(),
    "entropy": h["entropy_filtered"].round(3).tolist(),
    "regime": [REGS.index(r) for r in h["regime_filtered"]],
    "probs": [[round(float(v), 3) for v in h[f"pf_{r}"]] for r in REGS],
    "oosStart": oos_start,
    "summary": summary,
    "transition": [[round(float(v), 4) for v in row] for row in trans.values],
    "stats": {c: [round(float(v), 4) for v in stats[c]] for c in stats.columns},
    "means": {c: [round(float(v), 5) for v in means[c]] for c in means.columns},
    "features": list(means.columns),
    # ship the code with the page so anyone with the link can reproduce it
    "sources": {n: Path(n).read_text(encoding="utf-8")
                for n in ("hmm_regime.py", "build_dashboard.py", "requirements.txt")},
}

template = Path("regime_dashboard.template.html").read_text(encoding="utf-8")
blob = json.dumps(data, separators=(",", ":"))
html = template.replace("/*__DATA__*/null", blob)
# entities instead of raw non-ASCII so the page survives a server that omits the charset
html = html.replace("·", "&middot;").replace("×", "&times;").replace("−", "&minus;")
Path("regime_dashboard.html").write_text(html, encoding="utf-8")
print(f"wrote regime_dashboard.html  ({len(html)/1e6:.2f} MB, {len(data['dates'])} days)")
