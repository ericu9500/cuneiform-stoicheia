"""Model probabilities (GPU job output), search vs model error, context ablations, headroom oracles (DEV only).
  python scripts/diag_errors_q8.py -> results/diag/errors/sec8.md sec9.md"""
import collections
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_errors_lib import NUM, OUT, evalset, load, md, pk  # noqa: E402

ev, cache, P = load()
I = pd.read_pickle(OUT + "/I2.pkl")
S = pd.read_pickle(OUT + "/S.pkl").sort_values(["i", "k"]).reset_index(drop=True)
G = pk(os.path.join(OUT, "gpu_probs.pkl.gz"))
A = G["A"]
sn, rv = ev["sign_names"], ev["read_vocab"]
inst = ev["instances"]
XID = sn.index("X")
S["lp"] = np.concatenate([a["lp_gold"] for a in A])
S["rank"] = np.concatenate([a["rank"] for a in A])
S["conf"] = np.exp(np.concatenate([a["top_lp"][:, 0] for a in A]))
S["ent"] = np.concatenate([a["ent"] for a in A])
S["edge"] = np.minimum(S.k, S.L - 1 - S.k)
S["edgeb"] = pd.cut(S.edge, [-1, 0, 1, 2, 4, 9], labels=["0 (touches context)", "1", "2", "3-4", "5-9"])
S["Lbin"] = pd.cut(S.L, [0, 1, 5, 10, 15, 20], labels=["1", "2-5", "6-10", "11-15", "16-20"])
o = ["## 8. Model probabilities, search error, context use (GPU job scripts/diag_errors_gpu.py, dev)\n"]


def R(d, by):
    g = d.groupby(by, observed=True)
    return md(pd.DataFrame({"slots": g.size(), "top1": 100 * g["rank"].apply(lambda s: (s == 0).mean()), "top3": 100 * g["rank"].apply(lambda s: (s < 3).mean()),
                            "top10": 100 * g["rank"].apply(lambda s: (s < 10).mean()), "median rank of gold": g["rank"].median() + 1,
                            "NLL of gold (nats)": -g.lp.mean(), "perplexity": np.exp(-g.lp.mean()), "mean entropy": g.ent.mean(), "iter recall": 100 * g.ok.mean()}), nd=2)


o.append("### 8.1 One-pass distribution at every gap slot: rank of the gold sign\n\n" + R(S, "Lbin"))
o.append("\n### 8.1b ... by distance to the gap edge\n\n" + R(S, "edgeb"))
o.append("\n### 8.1c ... by sign type\n\n" + R(S, "type"))
o.append("\n### 8.1d ... novel / non-novel\n\n" + R(S, "novel"))
o.append("\n### 8.1e ... by language (sux, akk)\n\n" + R(S[S.lang.isin(["sux", "akk"])], "lang"))
g = S.groupby("i")
I["all10"] = (g["rank"].max() < 10).reindex(I.i).values
I["all3"] = (g["rank"].max() < 3).reindex(I.i).values
t = I.groupby("Lbin", observed=True)
o.append("\n### 8.1f Gold sign within the one-pass top-k at EVERY slot of the gap (% of gaps)\n\n" +
         md(pd.DataFrame({"n": t.size(), "all slots in top-3": 100 * t.all3.mean(), "all slots in top-10": 100 * t.all10.mean(), "exact (iter)": 100 * t.exact_iter.mean()})))
# calibration
S["cb"] = pd.cut(S.conf, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0001])
S["ok1"] = S["rank"] == 0
g = S.groupby("cb", observed=True)
cal = pd.DataFrame({"slots": g.size(), "mean confidence": 100 * g.conf.mean(), "one-pass accuracy": 100 * g.ok1.mean(), "iter accuracy": 100 * g.ok.mean()})
ece = float((cal.slots * (cal["mean confidence"] - cal["one-pass accuracy"]).abs()).sum() / cal.slots.sum())
o.append(f"\n### 8.2 Calibration of the one-pass top-1 probability (ECE {ece:.1f} points)\n\n" + md(cal))
hi = S[S.conf >= 0.9]
o.append(f"\nSlots with one-pass confidence >= 0.9: {len(hi)} ({100 * len(hi) / len(S):.1f}% of slots), accuracy {100 * hi.ok1.mean():.1f}%. "
         f"Slots with confidence < 0.3: {int((S.conf < .3).sum())} ({100 * (S.conf < .3).mean():.1f}%), one-pass accuracy {100 * S[S.conf < .3].ok1.mean():.1f}%, iter accuracy {100 * S[S.conf < .3].ok.mean():.1f}%.")
# B
tasks, res = G["B_tasks"], G["B_res"]
B = pd.DataFrame(tasks, columns=["i", "kind", "mode", "k"])
B["lp"], B["rank"], B["arg"] = res[:, 0], res[:, 1], res[:, 2]
B = B.merge(I[["i", "L", "novel", "lang", "exact_iter", "ser_iter"]], on="i")
sums = B.groupby(["i", "kind", "mode"]).lp.sum().unstack(["kind", "mode"])
sums.columns = [f"{a}_{b}" for a, b in sums.columns]
o.append("\n### 8.3 Search error or model error (known width)\n\nFor every gap whose iterative prediction is wrong, the model scores the gold gap and its own prediction: 'l2r' = chain rule left to right "
         "(slot k predicted with slots < k filled in, later slots masked); 'pll' = pseudo-log-likelihood (each slot masked in turn, the other gap slots filled in). "
         "If the model scores the gold higher than what the decoder returned, a better search could have found it (search error); otherwise the model prefers its wrong answer (model error).\n")
rows = []
W = sums.dropna(subset=["pred_l2r"])
W = W.join(I.set_index("i")[["L", "novel", "lang", "Lbin"]])
for name, sub in [("all wrong gaps", W)] + [(f"L {lb}", W[W.Lbin == lb]) for lb in ["1", "2-5", "6-10", "11-15", "16-20"]] + [("novel", W[W.novel]), ("non-novel", W[~W.novel])]:
    d_l = sub["gold_l2r"] - sub["pred_l2r"]
    row = {"subset": name, "wrong gaps": len(sub), "gold scores higher, l2r % (search error)": 100 * (d_l > 0).mean(), "mean l2r logp gold - pred (nats)": d_l.mean(),
           "mean l2r logp gold per sign": (sub["gold_l2r"] / sub.L).mean(), "mean l2r logp pred per sign": (sub["pred_l2r"] / sub.L).mean()}
    if "gold_pll" in sub:
        s2 = sub[sub.L > 1]
        d_p = s2["gold_pll"] - s2["pred_pll"]
        row["gold scores higher, pll % (L>=2)"] = 100 * (d_p > 0).mean() if len(s2) else np.nan
        row["both l2r and pll %"] = 100 * ((d_p > 0) & (d_l[s2.index] > 0)).mean() if len(s2) else np.nan
    rows.append(row)
o.append(md(pd.DataFrame(rows), index=False, nd=2))
se = set(W.index[(W["gold_l2r"] - W["pred_l2r"]) > 0])
I["search_err"] = I.i.isin(se)
ser_fix = np.where(I.search_err, 0.0, I.ser_iter)
o.append(f"\nOracle search (every gap where the gold has the higher l2r score is returned right): SER {100 * ser_fix.mean():.1f} vs {100 * I.ser_iter.mean():.1f}; exact {100 * ((I.exact_iter == 1) | I.search_err).mean():.1f} vs {100 * I.exact_iter.mean():.1f}. "
         "This is an upper bound for better decoding with this model; the rest needs a better model.")
t = I.assign(fix=ser_fix).groupby("Lbin", observed=True)
o.append("\n" + md(pd.DataFrame({"n": t.size(), "SER iter": 100 * t.ser_iter.mean(), "SER oracle search": 100 * t.fix.mean(), "search-error gaps %": 100 * t.search_err.mean(), "exact iter": 100 * t.exact_iter.mean()})))
# teacher forcing
Bg = B[B.kind == "gold"]
tf = Bg.groupby(["mode"]).apply(lambda d: pd.Series({"slots": len(d), "top1 %": 100 * (d["rank"] == 0).mean(), "top10 %": 100 * (d["rank"] < 10).mean(), "NLL": -d.lp.mean()}))
o.append("\n### 8.4 Teacher forcing: accuracy of a slot when the other gap signs are the gold ones\n\n'l2r' = signs to the left in the gap are gold, the rest masked; 'pll' = all other gap signs gold (the single-sign task at that position; L >= 2). "
         "Compare the free-running iterative recall and the one-pass top-1.\n")
rows = []
Bg = Bg.merge(S[["i", "k", "ok", "ok1", "edgeb", "type"]], on=["i", "k"])
Bg["Lbin"] = pd.cut(Bg.L, [0, 1, 5, 10, 15, 20], labels=["1", "2-5", "6-10", "11-15", "16-20"])
for lb, d in Bg.groupby("Lbin", observed=True):
    l, p_ = d[d["mode"] == "l2r"], d[d["mode"] == "pll"]
    rows.append({"L": lb, "slots": len(l), "one-pass top1": 100 * l.ok1.mean(), "iterative (free)": 100 * l.ok.mean(), "l2r teacher-forced top1": 100 * (l["rank"] == 0).mean(),
                 "all-others-given top1": 100 * (p_["rank"] == 0).mean() if len(p_) else np.nan, "all-others-given top10": 100 * (p_["rank"] < 10).mean() if len(p_) else np.nan,
                 "NLL one-pass": np.nan, "NLL l2r": -l.lp.mean(), "NLL all-others-given": -p_.lp.mean() if len(p_) else np.nan})
t8 = pd.DataFrame(rows)
t8["NLL one-pass"] = (-S.groupby("Lbin", observed=True).lp.mean()).values
o.append(md(t8, index=False, nd=2))
rows = []
for nv, d in Bg.groupby("novel"):
    l, p_ = d[d["mode"] == "l2r"], d[d["mode"] == "pll"]
    rows.append({"novel": nv, "slots": len(l), "one-pass top1": 100 * l.ok1.mean(), "iterative (free)": 100 * l.ok.mean(), "l2r teacher-forced top1": 100 * (l["rank"] == 0).mean(),
                 "all-others-given top1": 100 * (p_["rank"] == 0).mean(), "all-others-given top10": 100 * (p_["rank"] < 10).mean()})
o.append("\n" + md(pd.DataFrame(rows), index=False, nd=2))
rows = []
for ty, d in Bg.groupby("type"):
    l, p_ = d[d["mode"] == "l2r"], d[d["mode"] == "pll"]
    rows.append({"type": ty, "slots": len(l), "one-pass top1": 100 * l.ok1.mean(), "iterative (free)": 100 * l.ok.mean(), "l2r teacher-forced top1": 100 * (l["rank"] == 0).mean(),
                 "all-others-given top1": 100 * (p_["rank"] == 0).mean(), "all-others-given top10": 100 * (p_["rank"] < 10).mean()})
o.append("\n" + md(pd.DataFrame(rows), index=False, nd=2))
# C
C = G["C"]
golds = [evalset.gold(ev, x)[0] for x in inst]
names = {"base": "full context (as evaluated)", "noread": "context readings hidden", "bare": "context readings, joiners, determinatives hidden (bare signs)", "nolang": "language unknown",
         "left": "left context only", "right": "right context only", "win5": "context +-5 signs", "win10": "context +-10 signs", "win20": "context +-20 signs", "win50": "context +-50 signs"}
rows = []
for k, ps in C.items():
    ham = np.array([float((p_ != g_).mean()) for p_, g_ in zip(ps, golds)])
    ser = np.array([Levenshtein.distance(p_.tolist(), g_.tolist()) / len(g_) for p_, g_ in zip(ps, golds)])
    ex = np.array([float((p_ == g_).all()) for p_, g_ in zip(ps, golds)])
    row = {"context": names[k], "SER (one pass)": 100 * ser.mean(), "exact": 100 * ex.mean(), "L=1 top1": 100 * ex[(I.L == 1).values].mean(), "SER L 2-5": 100 * ser[((I.L >= 2) & (I.L <= 5)).values].mean(),
           "SER L 6-20": 100 * ser[(I.L >= 6).values].mean(), "SER sux": 100 * ser[(I.lang == "sux").values].mean(), "SER akk": 100 * ser[(I.lang == "akk").values].mean(),
           "SER novel": 100 * ser[I.novel.values].mean(), "SER non-novel": 100 * ser[~I.novel.values].mean()}
    if k == "base":
        base_ser = ser
    row["SER, gaps with >50 signs on both sides"] = 100 * ser[((I.d_start > 50) & (I.d_end > 50)).values].mean()
    rows.append(row)
o.append(f"\n### 8.5 Context ablations (one-pass argmax, known width; n = 6000; gaps with > 50 signs on both sides: n = {int(((I.d_start > 50) & (I.d_end > 50)).sum())})\n\n" + md(pd.DataFrame(rows), index=False))
open(os.path.join(OUT, "sec8.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))

# ---------------- headroom
o = ["## 9. Headroom: oracles that fix one weakness at a time\n\nKnown width, iterative decoder unless noted. Baseline: "
     f"SER {100 * I.ser_iter.mean():.1f}, CER {100 * I.cer_iter.mean():.1f}, exact {100 * I.exact_iter.mean():.1f} (mean over 6,000 gaps). Each oracle repairs only the named weakness and "
     "leaves every other slot as predicted, so it is an upper bound on what fixing that weakness alone can buy (fixing one sign would in practice also help its neighbours, which the oracle does not credit).\n"]
Sg = {i: d for i, d in S.groupby("i")}


def rescore(fix_sign=None, fix_read=False, fix_join=False, fix_det=False, pred_key="known/iter"):
    ser, cer, ex = [], [], []
    for i, x in enumerate(inst):
        gs, gr, gj, gd = evalset.gold(ev, x)
        p = P[pred_key][i]
        ps, pr, pj, pd_ = (np.array(p[k], np.int64).copy() for k in ("sign", "read", "join", "det"))
        if fix_sign is not None:
            m = fix_sign(i, gs, gr, gj, gd, ps)
            ps[m], pr[m], pj[m], pd_[m] = gs[m], gr[m], gj[m], gd[m]
        ok = ps == gs
        if fix_read:
            pr[ok] = gr[ok]
        if fix_join:
            pj = gj.copy() if fix_join == "all" else np.where(ok, gj, pj)
        if fix_det:
            pd_ = gd.copy() if fix_det == "all" else np.where(ok, gd, pd_)
        g_str = evalset.render(ev, gs, gr, gj, gd)
        ser.append(Levenshtein.distance(ps.tolist(), gs.tolist()) / x["L"])
        cer.append(Levenshtein.distance(evalset.render(ev, ps, pr, pj, pd_), g_str) / max(1, len(g_str)))
        ex.append(float((ps == gs).all()))
    return np.array(ser), np.array(cer), np.array(ex)


b_ser, b_cer, b_ex = rescore()
assert abs(b_ser.mean() - I.ser_iter.mean()) < 1e-9
types = {i: d.type.values for i, d in Sg.items()}
rows = []


def add(name, ser, cer, ex, note=""):
    rows.append({"oracle": name, "SER": 100 * ser.mean(), "d SER": 100 * (ser.mean() - b_ser.mean()), "CER": 100 * cer.mean(), "d CER": 100 * (cer.mean() - b_cer.mean()),
                 "exact": 100 * ex.mean(), "d exact": 100 * (ex.mean() - b_ex.mean()), "note": note})


add("none (baseline)", b_ser, b_cer, b_ex)
n_num = int((S.type == "numeral").sum())
add("numerals: every numeral sign right", *rescore(lambda i, gs, gr, gj, gd, ps: types[i] == "numeral"), f"{n_num} gold slots ({100 * n_num / len(S):.1f}%), recall now {100 * S[S.type == 'numeral'].ok.mean():.1f}%")
isnum_pred = {}
add("numerals: right value whenever a numeral was predicted for a numeral (unit/value errors only)",
    *rescore(lambda i, gs, gr, gj, gd, ps: (types[i] == "numeral") & np.array([bool(NUM.match(rv[r] if r > 1 else sn[s].lower())) for s, r in zip(ps, P["known/iter"][i]["read"])])),
    "only numeral -> other numeral confusions repaired")
add("no placeholder 'X' predicted (slots where X is predicted and wrong become right)", *rescore(lambda i, gs, gr, gj, gd, ps: (ps == XID) & (gs != XID)),
    f"{int(((S.p == XID) & (S.g != XID)).sum())} slots")
for ty in ("determinative", "akk:syllabic", "akk:logogram", "sux:in multi-sign word", "sux:1-sign word"):
    n = int((S.type == ty).sum())
    add(f"type '{ty}': every such sign right", *rescore(lambda i, gs, gr, gj, gd, ps, ty=ty: types[i] == ty), f"{n} slots ({100 * n / len(S):.1f}%), recall now {100 * S[S.type == ty].ok.mean():.1f}% (size of the class, not a realistic target)")
fall = cache["f_sign"].sum(0)
add("rare signs (train frequency < 10k) right", *rescore(lambda i, gs, gr, gj, gd, ps: fall[gs] < 10000), f"{int((S.f_all < 10000).sum())} slots, recall now {100 * S[S.f_all < 10000].ok.mean():.1f}%")
add("readings: right reading whenever the sign is right", *rescore(fix_read=True))
add("joiners: right joiner whenever the sign is right", *rescore(fix_join=True))
add("joiners: gold joiners at every slot (word and line structure of the gap known)", *rescore(fix_join="all"))
add("determinative plane: gold at every slot", *rescore(fix_det="all"))
add("readings + joiners + determinatives right whenever the sign is right", *rescore(fix_read=True, fix_join=True, fix_det=True))
# shifts: Hamming -> already Levenshtein; oracle: displaced-right slots fixed
add("interior slots (not touching context) as good as... all right", *rescore(lambda i, gs, gr, gj, gd, ps: (np.arange(len(gs)) > 0) & (np.arange(len(gs)) < len(gs) - 1)), "size of the interior problem (51,300 slots)")
add("edge slots (first and last of the gap) all right", *rescore(lambda i, gs, gr, gj, gd, ps: (np.arange(len(gs)) == 0) | (np.arange(len(gs)) == len(gs) - 1)), "11,700 slots")
# top-10 oracle reranker (one pass)
rk = {i: d["rank"].values for i, d in Sg.items()}
add("oracle choice among the one-pass top-10 of each slot", *rescore(lambda i, gs, gr, gj, gd, ps: rk[i] < 10), "upper bound for reranking / a human choosing from 10 candidates")
add("oracle choice among the one-pass top-3 of each slot", *rescore(lambda i, gs, gr, gj, gd, ps: rk[i] < 3))
se_ser = np.where(I.search_err, 0, b_ser)
se_cer = np.where(I.search_err, 0, b_cer)
add("perfect search under the model's own left-to-right score", se_ser, se_cer, np.where(I.search_err, 1, b_ex), f"{int(I.search_err.sum())} gaps where the model scores the gold above the decoder's output")
m4 = (I.ser_4gram < I.ser_iter).values
add("per-gap choice between model and 4-gram", np.where(m4, I.ser_4gram, b_ser), np.where(m4, I.cer_4gram, b_cer), np.maximum(b_ex, I.exact_4gram.values), "what the 4-gram knows that the model does not use")
# context-related, by reweighting (not per-slot oracles)
I["ser_adj_"] = I.ser_iter - I.L.map(I.groupby("L").ser_iter.mean()) + I.ser_iter.mean()


def gap_to(mask, ref):
    """points of overall SER if the masked gaps had the L-adjusted SER of the reference gaps"""
    return 100 * (I.ser_adj_[mask].mean() - I.ser_adj_[ref].mean()) * mask.mean()


extra = []
m = I.unk10 > 0
extra.append(("gaps with unknown tokens (x, gap, X) within 10 signs behave like gaps without", int(m.sum()), gap_to(m, ~m)))
m = I.unk3 > 0
extra.append(("... within 3 signs", int(m.sum()), gap_to(m, ~m)))
m = (I.d_start <= 10) & I.cut_left
extra.append(("gaps <= 10 signs after a chunk cut behave like gaps with > 10 signs of left context", int(m.sum()), gap_to(m, I.d_start > 10)))
m = (I.d_end <= 10) & I.cut_right
extra.append(("gaps <= 10 signs before a chunk cut behave like gaps with > 10 signs of right context", int(m.sum()), gap_to(m, I.d_end > 10)))
m = (I.d_start <= 2)
extra.append(("gaps with <= 2 signs of left context behave like the rest", int(m.sum()), gap_to(m, ~m)))
m = I.lang == "akk"
extra.append(("Akkadian gaps as good as Sumerian gaps", int(m.sum()), gap_to(m, I.lang == "sux")))
m = (I.lang == "akk") & I.novel
extra.append(("novel Akkadian gaps as good as novel Sumerian gaps", int(m.sum()), gap_to(m, (I.lang == "sux") & I.novel)))
for per in ("Old Akkadian", "Old Assyrian", "Middle Babylonian"):
    m = I.period == per
    extra.append((f"{per} gaps as good as the rest", int(m.sum()), gap_to(m, ~m)))
m = I.genre == "letter"
extra.append(("letters as good as the rest", int(m.sum()), gap_to(m, ~m)))
m = (I.lang == "akk") & (I.src == "cdli")
extra.append(("Akkadian from CDLI as good as Akkadian from Oracc", int(m.sum()), gap_to(m, (I.lang == "akk") & (I.src == "oracc"))))
m = I.novel
extra.append(("novel gaps as good as non-novel gaps (memorisation gap; not reachable)", int(m.sum()), gap_to(m, ~I.novel)))
o.append("### 9.1 Slot / plane oracles\n\n" + md(pd.DataFrame(rows), index=False))
o.append("\n### 9.2 Group oracles (the affected gaps reach the L-adjusted SER of the reference gaps; confounded with language, period and novelty, so these are generous)\n\n" +
         md(pd.DataFrame(extra, columns=["oracle", "gaps affected", "points of overall SER"]), index=False, nd=2))
# unknown width
o.append("\n### 9.3 Unknown width\n")
for tag, k in (("iter-w", "iter"), ("indep", "indep"), ("iter", "iter")):
    okw = I["d_" + tag] == 0
    o.append(f"- {tag}: CER {100 * I['ucer_' + tag].mean():.1f}; oracle width (= known-width {k}) {100 * I['cer_' + k].mean():.1f}: headroom {100 * (I['ucer_' + tag].mean() - I['cer_' + k].mean()):.1f} CER points, "
             f"SER {100 * I['user_' + tag].mean():.1f} -> {100 * I['ser_' + k].mean():.1f}; width right in {100 * okw.mean():.1f}% of gaps; "
             f"over-filled gaps cost {100 * ((I['ucer_' + tag] - I['cer_' + k])[I['d_' + tag] > 0]).sum() / len(I):.1f} points, under-filled {100 * ((I['ucer_' + tag] - I['cer_' + k])[I['d_' + tag] < 0]).sum() / len(I):.1f}.")
open(os.path.join(OUT, "sec9.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
