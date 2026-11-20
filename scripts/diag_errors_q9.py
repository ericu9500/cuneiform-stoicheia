"""What-if decoders and the train/dev gap (GPU jobs 2 and 3; DEV).
  python scripts/diag_errors_q9.py -> results/diag/errors/sec11.md"""
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_errors_lib import OUT, evalset, load, md, pk  # noqa: E402

ev, cache, P = load()
I = pd.read_pickle(OUT + "/I2.pkl")
inst = ev["instances"]
golds = [evalset.gold(ev, x) for x in inst]
D = pk(os.path.join(OUT, "gpu_decoders.pkl.gz"))
G = pk(os.path.join(OUT, "gpu_probs.pkl.gz"))
o = ["## 11. What a different decoder realises with the same checkpoint (dev)\n"]


def sc(signs):
    ser = np.array([Levenshtein.distance([int(v) for v in p], g[0].tolist()) / len(g[0]) for p, g in zip(signs, golds)])
    ex = np.array([float(len(p) == len(g[0]) and np.array_equal(p, g[0])) for p, g in zip(signs, golds)])
    return ser, ex


def cer(pred):
    out = []
    for p, (gs, gr, gj, gd) in zip(pred, golds):
        g_str = evalset.render(ev, gs, gr, gj, gd)
        out.append(Levenshtein.distance(evalset.render(ev, p["sign"], p["read"], p["join"], p["det"]), g_str) / max(1, len(g_str)))
    return np.array(out)


Lb = I.Lbin.values
rows = []


def add(name, ser, ex, c=None):
    r = {"decoder (known width)": name, "SER": 100 * ser.mean(), "exact": 100 * ex.mean(), "CER": np.nan if c is None else 100 * c.mean()}
    for lb in ("1", "2-5", "6-10", "11-15", "16-20"):
        r["SER L " + lb] = 100 * ser[Lb == lb].mean()
    r["SER novel"] = 100 * ser[I.novel.values].mean()
    r["SER non-novel"] = 100 * ser[~I.novel.values].mean()
    r["exact non-novel"] = 100 * ex[~I.novel.values].mean()
    rows.append(r)


add("one pass (indep)", I.ser_indep.values, I.exact_indep.values, I.cer_indep.values)
add("iterative, most confident first (evaluated)", I.ser_iter.values, I.exact_iter.values, I.cer_iter.values)
s1, e1 = sc(D["known/l2r-beam1"]["sign"])
add("left-to-right greedy", s1, e1)
s8, e8 = sc(D["known/l2r-beam8"]["sign"])
add("left-to-right beam 8", s8, e8)
# rerank iter vs beam by the l2r chain score
B = pd.DataFrame(G["B_tasks"], columns=["i", "kind", "mode", "k"])
B["lp"] = G["B_res"][:, 0]
sums = B[B["mode"] == "l2r"].groupby(["i", "kind"]).lp.sum().unstack("kind")
it_score = sums["pred"].fillna(sums["gold"]).reindex(range(len(inst))).values       # exact predictions: pred = gold
bs = D["known/l2r-beam8"]["score"]
use_beam = bs > it_score + 1e-6
add("iterative or beam 8, whichever has the higher left-to-right score", np.where(use_beam, s8, I.ser_iter.values), np.where(use_beam, e8, I.exact_iter.values))
fb = D["known/iter-fb"]
sf, ef = sc([p["sign"] for p in fb])
add("iterative, predicted reading / joiner / determinative fed back", sf, ef, cer(fb))
o.append("### 11.1 Known width\n\n" + md(pd.DataFrame(rows), index=False))
o.append(f"\nBeam 8 finds a sequence with a higher left-to-right score than the iterative decoder in {100 * use_beam.mean():.1f}% of gaps; the gold scores higher than the beam's best in "
         f"{100 * np.mean(sums['gold'].reindex(range(len(inst))).values > bs + 1e-6):.1f}% of gaps (remaining search error of beam 8).")
# width
rows = []
for lam, ks in D["width_k"].items():
    pred = [D["width_pred"][(i, k)] for i, k in enumerate(ks)]
    ser, ex = sc([p["sign"] for p in pred])
    c = cer(pred)
    d = np.array(ks) - I.L.values
    r = {"penalty per slot (nats)": lam, "CER": 100 * c.mean(), "SER": 100 * ser.mean(), "exact": 100 * ex.mean(), "width right %": 100 * (d == 0).mean(), "mean d": d.mean(),
         "over-filled %": 100 * (d > 0).mean(), "under-filled %": 100 * (d < 0).mean()}
    for lb in ("1", "2-5", "6-10", "11-15", "16-20"):
        r["CER L " + lb] = 100 * c[Lb == lb].mean()
    r["CER novel"] = 100 * c[I.novel.values].mean()
    rows.append(r)
o.append("\n### 11.2 Unknown width: two-stage decoder with a penalty on the width (0 = the evaluated iter-w)\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
o.append(f"\nKnown-width iterative for reference: CER {100 * I.cer_iter.mean():.1f}, SER {100 * I.ser_iter.mean():.1f}, exact {100 * I.exact_iter.mean():.1f}.")
tp = os.path.join(OUT, "gpu_traingap.pkl.gz")
if os.path.exists(tp):
    T = pk(tp)
    o.append("\n## 12. Generalisation gap: the same protocol on 2,000 random train texts\n\nGap instances drawn as for dev (300 per L, same seed) from 2,000 random train texts; fixed noisings as in the training-time dev evaluation.\n")
    rows = []
    for name in ("train-sample", "dev"):
        t = T[name]
        L = t["L"]
        r = {"texts": name, "n gaps": len(L), "SER iter": 100 * t["iter_ser"].mean(), "exact iter": 100 * t["iter_exact"].mean(), "SER one pass": 100 * t["indep_ser"].mean(),
             "L=1 top1": 100 * t["iter_exact"][L == 1].mean(), "SER L 2-5": 100 * t["iter_ser"][(L >= 2) & (L <= 5)].mean(), "SER L 6-10": 100 * t["iter_ser"][(L >= 6) & (L <= 10)].mean(),
             "SER L 11-20": 100 * t["iter_ser"][L >= 11].mean(), "SER sux": 100 * t["iter_ser"][np.array(t["lang"]) == "sux"].mean(), "SER akk": 100 * t["iter_ser"][np.array(t["lang"]) == "akk"].mean(),
             "span t=0.15: sign NLL": t["fixed"][0]["sign"]["nll"], "span: sign acc %": 100 * t["fixed"][0]["sign"]["acc"], "translit: reading NLL": t["fixed"][4]["read"]["nll"],
             "translit: reading acc %": 100 * t["fixed"][4]["read"]["acc"]}
        rows.append(r)
    rows.append({"texts": "dev, non-novel gaps only", "n gaps": int((~I.novel).sum()), "SER iter": 100 * I[~I.novel].ser_iter.mean(), "exact iter": 100 * I[~I.novel].exact_iter.mean(), "SER one pass": 100 * I[~I.novel].ser_indep.mean()})
    rows.append({"texts": "dev, novel gaps only", "n gaps": int(I.novel.sum()), "SER iter": 100 * I[I.novel].ser_iter.mean(), "exact iter": 100 * I[I.novel].exact_iter.mean(), "SER one pass": 100 * I[I.novel].ser_indep.mean()})
    o.append(md(pd.DataFrame(rows), index=False, nd=2))
open(os.path.join(OUT, "sec11.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
