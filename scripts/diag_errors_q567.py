"""Q5 unknown width, Q6 transliteration of bare signs, Q7 4-gram wins (DEV only).
  python scripts/diag_errors_q567.py -> results/diag/errors/sec5.md sec6.md sec7.md"""
import collections
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_errors_lib import NUM, OUT, FIRST_REAL_SIGN, evalset, genre_norm, load, md, period_norm, read_diff, src_norm  # noqa: E402

ev, cache, P = load()
I = pd.read_pickle(OUT + "/I.pkl")
S = pd.read_pickle(OUT + "/S.pkl")
sn, rv = ev["sign_names"], ev["read_vocab"]
inst = ev["instances"]
I["Lbin"] = pd.cut(I.L, [0, 1, 5, 10, 15, 20], labels=["1", "2-5", "6-10", "11-15", "16-20"])


def score(p, x):
    gs, gr, gj, gd = evalset.gold(ev, x)
    g_str = evalset.render(ev, gs, gr, gj, gd)
    ps = [int(v) for v in p["sign"]]
    return (Levenshtein.distance(ps, gs.tolist()) / x["L"],
            Levenshtein.distance(evalset.render(ev, p["sign"], p["read"], p["join"], p["det"]), g_str) / max(1, len(g_str)),
            float(ps == gs.tolist()))


# ---------------- Q5
o = ["## 5. Unknown width\n\nThe gap is shown as L + extra slots, extra ~ U{0..ceil(0.3 L)+2}; d = predicted width - true width L.\n"]
decs = ("unknown/indep", "unknown/iter", "unknown/iter-w")
for d in decs:
    tag = d.split("/")[1]
    I["w_" + tag] = [len(p["sign"]) for p in P[d]]
    I["d_" + tag] = I["w_" + tag] - I.L
    sc = np.array([score(p, x) for p, x in zip(P[d], inst)])
    I["user_" + tag], I["ucer_" + tag], I["uex_" + tag] = sc[:, 0], sc[:, 1], sc[:, 2]
rows = []
for d in decs:
    tag = d.split("/")[1]
    for lb, sub in list(I.groupby("Lbin", observed=True)) + [("1-20", I)]:
        dd = sub["d_" + tag]
        rows.append({"decoder": tag, "L": lb, "n": len(sub), "mean d": dd.mean(), "mean |d|": dd.abs().mean(), "d<=-3 %": 100 * (dd <= -3).mean(),
                     "d=-2 %": 100 * (dd == -2).mean(), "d=-1 %": 100 * (dd == -1).mean(), "d=0 %": 100 * (dd == 0).mean(),
                     "d=+1 %": 100 * (dd == 1).mean(), "d=+2 %": 100 * (dd == 2).mean(), "d>=+3 %": 100 * (dd >= 3).mean(),
                     "mean extra offered": sub.extra.mean()})
o.append("### 5.1 Predicted minus true width, by true L\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
rows = []
for d in decs:
    tag = d.split("/")[1]
    for L in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 20):
        sub = I[I.L == L]
        dd = sub["d_" + tag]
        rows.append({"decoder": tag, "L": L, "n": len(sub), "mean d": dd.mean(), "d<0 %": 100 * (dd < 0).mean(), "d=0 %": 100 * (dd == 0).mean(), "d>0 %": 100 * (dd > 0).mean()})
o.append("\n### 5.1b ... selected single L (n = 300 each)\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
rows = []
for d in decs:
    tag = d.split("/")[1]
    for e, sub in I[I.L <= 5].groupby("extra"):
        dd = sub["d_" + tag]
        rows.append({"decoder": tag, "extra slots offered": e, "n": len(sub), "mean d": dd.mean(), "d=0 %": 100 * (dd == 0).mean(), "d>0 %": 100 * (dd > 0).mean(),
                     "filled every slot %": 100 * (sub["w_" + tag] == sub.L + sub.extra).mean(), "CER": 100 * sub["ucer_" + tag].mean()})
o.append("\n### 5.2 Short gaps (L 1-5): width error by number of extra slots offered\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
rows = []
for d in decs:
    tag = d.split("/")[1]
    I["fill_frac"] = (I["w_" + tag] - I.L) / I.extra.replace(0, np.nan)
    for lb, sub in list(I.groupby("Lbin", observed=True)):
        s2 = sub[sub.extra > 0]
        rows.append({"decoder": tag, "L": lb, "n (extra>0)": len(s2), "mean (width - L)/extra": s2.fill_frac.mean(), "corr(width - L, extra)": np.corrcoef(s2["d_" + tag], s2.extra)[0, 1]})
o.append("\n### 5.2b Does the predicted width follow the number of slots offered rather than the content? (extra > 0)\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
o.append("\n(mean (width - L)/extra = 0 would be a perfect width model, 1 = every offered slot filled; a correlation near 1 means the width estimate is driven by the slots offered.)")
# decomposition
rows = []
kn = {"indep": "indep", "iter": "iter", "iter-w": "iter"}
for d in decs:
    tag = d.split("/")[1]
    k = kn[tag]
    okw = I["d_" + tag] == 0
    for name, sub in (("all", I), ("L 1", I[I.L == 1]), ("L 2-5", I[(I.L >= 2) & (I.L <= 5)]), ("L 6-10", I[(I.L >= 6) & (I.L <= 10)]), ("L 11-20", I[I.L >= 11]),
                      ("novel", I[I.novel]), ("sux", I[I.lang == "sux"]), ("akk", I[I.lang == "akk"])):
        m = okw[sub.index]
        mixed_c = np.where(m, sub["ucer_" + tag], sub["cer_" + k])
        mixed_s = np.where(m, sub["user_" + tag], sub["ser_" + k])
        rows.append({"decoder": tag, "subset": name, "n": len(sub), "width right %": 100 * m.mean(), "CER unknown": 100 * sub["ucer_" + tag].mean(),
                     "CER known (same decoder)": 100 * sub["cer_" + k].mean(), "loss": 100 * (sub["ucer_" + tag].mean() - sub["cer_" + k].mean()),
                     "CER unknown | width right": 100 * sub["ucer_" + tag][m].mean(), "CER known | width right": 100 * sub["cer_" + k][m].mean(),
                     "CER unknown | width wrong": 100 * sub["ucer_" + tag][~m].mean(), "CER known | width wrong": 100 * sub["cer_" + k][~m].mean(),
                     "CER, wrong-width gaps replaced by known-width prediction": 100 * mixed_c.mean(),
                     "SER unknown": 100 * sub["user_" + tag].mean(), "SER known": 100 * sub["ser_" + k].mean()})
t = pd.DataFrame(rows)
o.append("\n### 5.3 How much of the CER loss against known width is width error\n\n'CER known' is the known-width prediction of the same decoder family (iter for iter-w): the upper bound an oracle width would give. "
         "'| width right' columns compare the two on the gaps where the predicted width equals L: any residual there is not a width error.\n\n" + md(t, index=False))
# cost by size of width error
rows = []
for d in decs:
    tag = d.split("/")[1]
    k = kn[tag]
    I["dc"] = pd.cut(I["d_" + tag], [-99, -3, -2, -1, 0, 1, 2, 99], labels=["<=-3", "-2", "-1", "0", "+1", "+2", ">=+3"])
    g = I.groupby("dc", observed=True)
    for dc, sub in g:
        rows.append({"decoder": tag, "d": dc, "n": len(sub), "CER unknown": 100 * sub["ucer_" + tag].mean(), "CER known": 100 * sub["cer_" + k].mean(),
                     "loss x share (points of overall CER)": 100 * (sub["ucer_" + tag] - sub["cer_" + k]).sum() / len(I), "mean L": sub.L.mean()})
o.append("\n### 5.4 CER by size of the width error; contribution to the overall loss\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
gp = os.path.join(OUT, "gpu_probs.pkl.gz")
if os.path.exists(gp):
    from diag_errors_lib import pk
    A = pk(gp)["A"]
    # the width model itself: p(BLANK) at slots that should be real / should be empty (one pass)
    real, blank = [], []
    rows = []
    for a, x in zip(A, inst):
        L = x["L"]
        real.append(a["pB"][:L]); blank.append(a["pB"][L:])
    r, b = np.concatenate(real), np.concatenate(blank)
    o.append(f"\n### 5.5 The blank probability itself (one pass, unknown width)\n\nSlots that hold a sign (n = {len(r)}): mean p(BLANK) {r.mean():.3f}, p(BLANK) > 0.5 in {100 * (r > 0.5).mean():.1f}%. "
             f"Slots that should be empty (n = {len(b)}): mean p(BLANK) {b.mean():.3f}, p(BLANK) > 0.5 in {100 * (b > 0.5).mean():.1f}%.")
    rows = []
    for lb, sub in I.groupby("Lbin", observed=True):
        rr = np.concatenate([A[i]["pB"][:inst[i]["L"]] for i in sub.i])
        bb = np.concatenate([A[i]["pB"][inst[i]["L"]:] for i in sub.i] + [np.zeros(0)])
        last = np.array([A[i]["pB"][inst[i]["L"] - 1] for i in sub.i])
        firstb = np.array([A[i]["pB"][inst[i]["L"]] for i in sub.i if inst[i]["extra"] > 0])
        rows.append({"L": lb, "real slots": len(rr), "mean p(BLANK) real": rr.mean(), "p(BLANK) of the last real slot": last.mean(), "empty slots": len(bb), "mean p(BLANK) empty": bb.mean(),
                     "p(BLANK) of the first empty slot": firstb.mean()})
    o.append("\n" + md(pd.DataFrame(rows), index=False, nd=3))
open(os.path.join(OUT, "sec5.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))

# ---------------- Q6
o = ["## 6. Transliteration of bare signs (readings, joiners, determinatives hidden; every dev chunk)\n"]
rows = []
for ci, ((ti, lg, a), d, dn) in enumerate(zip(ev["chunks"], P["translit"], P["translit_ngram"])):
    m = (a[1] > 1) & (a[0] >= FIRST_REAL_SIGN)
    idx = np.flatnonzero(m)
    meta = cache["dev_meta"][ti]
    df = pd.DataFrame({"sign": a[0][idx], "g": a[1][idx], "p": np.asarray(d["con"])[idx], "pf": np.asarray(d["free"])[idx], "hmm": np.asarray(dn["hmm"])[idx],
                       "det": a[3][idx], "brk": a[4][idx], "join": a[2][idx]})
    df["lang"] = ev["langs"][lg]
    df["period"] = period_norm(meta["period"])
    df["genre"] = genre_norm(meta["genre"])
    df["src"] = src_norm(meta["src"])
    df["text"] = ti
    rows.append(df)
T = pd.concat(rows, ignore_index=True)
T["ok"] = T.g == T.p
T["ok_f"] = T.g == T.pf
T["ok_h"] = T.g == T.hmm
E = T[~T.ok].copy()
E["cls"] = [read_diff(rv[a], rv[b]) for a, b in zip(E.g, E.p)]
o.append(f"Scored positions: {len(T)}; errors of the constrained prediction: {len(E)} ({100 * len(E) / len(T):.2f}%); unconstrained: {int((~T.ok_f).sum())} ({100 * (~T.ok_f).mean():.2f}%); HMM baseline: {100 * (~T.ok_h).mean():.2f}%.\n")
trc = collections.Counter()
for (lg_, a_, r_), n in cache["sign_read_lang"].items():
    trc[(a_, r_)] += n
adm = ev["sign_reads"]
T["adm"] = [(r in adm.get(a, [r])) for a, r in zip(T.sign, T.g)]
E["adm"] = T.adm[E.index]
E["train_pair"] = [trc.get((a, r), 0) for a, r in zip(E.sign, E.g)]
o.append(f"Gold reading excluded by the 'attested readings' constraint: {int((~T.adm).sum())} positions ({100 * (~T.adm).mean():.3f}% of all; {100 * (~T.adm).sum() / len(E):.1f}% of the errors); "
         f"of these the unconstrained head gets {int((T.ok_f & ~T.adm).sum())} right. Gold (sign, reading) pair never seen in train: {int((E.train_pair == 0).sum())}; seen once: {int((E.train_pair == 1).sum())}.")
o.append(f"Constrained right but unconstrained wrong: {int((T.ok & ~T.ok_f).sum())}; unconstrained right but constrained wrong: {int((~T.ok & T.ok_f).sum())}.\n")
cl = E.cls.value_counts()
desc = {"index": "convention: same letters, other index", "auslaut": "convention: long vs short form (du₁₁/dug₄, tu₄/tum)", "numeral": "numeral notation (unit / system / fraction)",
        "voicing": "same syllable family (b/p, d/t/ṭ, g/k/q, z/s/ṣ, e/i): lexical choice", "different": "a different value of the sign"}
o.append("### 6.1 Class of the error (heuristic, diag_errors_lib.read_diff)\n\n" + md(pd.DataFrame({"class": [desc[k] for k in cl.index], "errors": cl.values, "% of errors": 100 * cl.values / len(E),
                                                                                                "points of error rate": 100 * cl.values / len(T)}), index=False, nd=2))
o.append("\n### 6.1b ... by language\n\n" + md((pd.crosstab(E.lang, E.cls)).assign(total=lambda d: d.sum(axis=1))))
c = collections.Counter(zip(E.sign, E.g, E.p))
gcount = collections.Counter(zip(T.sign, T.g))
rows = [{"sign": sn[a], "gold": rv[g], "predicted": rv[p_], "n": n, "% of errors": 100 * n / len(E), "% of this gold reading": 100 * n / gcount[(a, g)],
         "reverse n": c.get((a, p_, g), 0), "class": read_diff(rv[g], rv[p_])} for (a, g, p_), n in c.most_common(60)]
o.append("\n### 6.2 Top 60 reading confusions (sign, gold -> predicted)\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
g = T.groupby("sign")
t = pd.DataFrame({"n": g.size(), "errors": g.size() - g.ok.sum(), "error %": 100 * (1 - g.ok.mean())})
t["% of all errors"] = 100 * t.errors / len(E)
t["top confusions"] = [", ".join(f"{rv[a]}->{rv[b]} {n}" for (a, b), n in collections.Counter(zip(E[E.sign == s].g, E[E.sign == s].p)).most_common(3)) for s in t.index]
t.index = [sn[s] for s in t.index]
t = t.sort_values("errors", ascending=False)
t["cum % of errors"] = t["% of all errors"].cumsum()
o.append("\n### 6.3 Signs with most errors\n\n" + md(t.head(40), nd=2))
o.append(f"\n{int((t.errors > 0).sum())} signs have at least one error; the top 10 signs carry {t['% of all errors'].head(10).sum():.1f}% of the errors, the top 40 {t['% of all errors'].head(40).sum():.1f}%.")
# AN / numerals detail
for nm, sub in (("AN (d / an / dingir)", E[E.sign == sn.index("AN")]), ("numerals (class 'numeral')", E[E.cls == "numeral"])):
    o.append(f"- {nm}: {len(sub)} errors = {100 * len(sub) / len(E):.1f}% of all")


T["cls"] = E.cls.reindex(T.index)


def by(col, min_n=1500):
    g = T.groupby(col, observed=True)
    t = pd.DataFrame({"n": g.size(), "error %": 100 * (1 - g.ok.mean()), "HMM error %": 100 * (1 - g.ok_h.mean())})
    e = T[~T.ok].groupby(col, observed=True)
    t["convention (index+auslaut) % of errors"] = 100 * e.cls.apply(lambda s: s.isin(["index", "auslaut"]).mean())
    t["numeral % of errors"] = 100 * e.cls.apply(lambda s: (s == "numeral").mean())
    t["voicing % of errors"] = 100 * e.cls.apply(lambda s: (s == "voicing").mean())
    return md(t[t.n >= min_n].sort_values("n", ascending=False), nd=2)


o.append("\n### 6.4 Errors by language\n\n" + by("lang", 300))
o.append("\n### 6.4b ... by period (n >= 1500 positions)\n\n" + by("period"))
o.append("\n### 6.4c ... by language x period\n\n" + by(["lang", "period"], 2000))
o.append("\n### 6.4d ... by genre\n\n" + by("genre"))
o.append("\n### 6.4e ... by source\n\n" + by(["lang", "src"], 1000))
T["dmg"] = np.where(T.brk == 0, "clean", np.where(T.brk == 1, "damaged", "restored by editor"))
o.append("\n### 6.4f ... by state of the sign\n\n" + by("dmg", 100))
T["isdet"] = T.det > 0
o.append("\n### 6.4g ... determinatives\n\n" + by("isdet", 100))
# text concentration
te = E.groupby("text").size().sort_values(ascending=False)
o.append(f"\nErrors are spread over {len(te)} of {T.text.nunique()} texts; the 20 texts with most errors hold {100 * te.head(20).sum() / len(E):.1f}% of them.")
# 'real' estimate
conv = E.cls.isin(["index", "auslaut"]).sum()
o.append(f"\n### 6.5 Convention noise vs real\n\nStrict convention noise (index or long/short form of the same value): {conv} errors = {100 * conv / len(E):.1f}% of errors = {100 * conv / len(T):.2f} points. "
         f"Numeral notation: {int((E.cls == 'numeral').sum())} ({100 * (E.cls == 'numeral').mean():.1f}%). Same syllable family (voicing / emphatic / e-i): {int((E.cls == 'voicing').sum())} ({100 * (E.cls == 'voicing').mean():.1f}%). "
         f"Different value: {int((E.cls == 'different').sum())} ({100 * (E.cls == 'different').mean():.1f}%).")
# symmetric pairs as a proxy of editor variation: confusions a->b where b->a is also frequent and both readings are used for the same sign in the same language
sym = 0
for (a, g_, p_), n in c.items():
    r = c.get((a, p_, g_), 0)
    if r >= 0.33 * n and n >= 0.33 * r and r > 0:
        sym += n
o.append(f"Errors in confusion pairs that go both ways at comparable rates (each direction >= 1/3 of the other; a sign of free variation between editors or of a distinction the context does not decide): {sym} = {100 * sym / len(E):.1f}% of errors.")
open(os.path.join(OUT, "sec6.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))

# ---------------- Q7
o = ["## 7. Where the 4-gram beats the model (known width; model = iterative)\n"]
I["win4"] = I.ser_4gram < I.ser_iter
I["tie"] = I.ser_4gram == I.ser_iter
I["winm"] = I.ser_4gram > I.ser_iter
o.append(f"Instance level (SER): 4-gram strictly better in {int(I.win4.sum())} gaps ({100 * I.win4.mean():.1f}%), tie in {int(I.tie.sum())} ({100 * I.tie.mean():.1f}%; "
         f"both exact in {int(((I.exact_iter == 1) & (I.exact_4gram == 1)).sum())}), model better in {int(I.winm.sum())} ({100 * I.winm.mean():.1f}%). "
         f"4-gram exact where the model is not: {int(((I.exact_4gram == 1) & (I.exact_iter == 0)).sum())}; model exact where the 4-gram is not: {int(((I.exact_iter == 1) & (I.exact_4gram == 0)).sum())}.")
I["gain4"] = I.ser_iter - I.ser_4gram
o.append(f"Mean SER margin when the 4-gram wins: {100 * I[I.win4].gain4.mean():.1f} points (median {100 * I[I.win4].gain4.median():.1f}); an oracle choosing the better of the two per gap: SER {100 * np.minimum(I.ser_iter, I.ser_4gram).mean():.1f} vs model {100 * I.ser_iter.mean():.1f}.\n")


def W(col, title, min_n=1):
    g = I.groupby(col, observed=True)
    t = pd.DataFrame({"n": g.size(), "4-gram wins %": 100 * g.win4.mean(), "tie %": 100 * g.tie.mean(), "model wins %": 100 * g.winm.mean(), "model SER": 100 * g.ser_iter.mean(),
                      "4-gram SER": 100 * g.ser_4gram.mean(), "oracle-min SER": 100 * g.apply(lambda d: np.minimum(d.ser_iter, d.ser_4gram).mean())})
    o.append(f"### {title}\n\n" + md(t[t.n >= min_n]) + "\n")


W("Lbin", "7.1 By gap length")
W("novel", "7.2 By novelty")
W(["Lbin", "novel"], "7.2b By gap length x novelty")
W("lang", "7.3 By language", 40)
W("genre", "7.4 By genre", 60)
W("period", "7.5 By period", 100)
I["dS"] = pd.cut(I.d_start, [-1, 0, 2, 5, 20, 1e9], labels=["0", "1-2", "3-5", "6-20", "21+"])
W("dS", "7.6 By left context length")
num_share = S.assign(v=S.type == "numeral").groupby("i").v.mean()
I["num_share"] = num_share.reindex(I.i).values
I["numb"] = pd.cut(I.num_share, [-1, 0, 0.25, 0.5, 1], labels=["no numeral", "<=25% numerals", "25-50%", ">50%"])
W("numb", "7.7 By share of numerals in the gold gap")
I["mser"] = pd.cut(I.ser_iter, [-0.01, 0, 0.25, 0.5, 0.75, 0.999, 1], labels=["0", "0-25%", "25-50%", "50-75%", "75-<100%", "100%"])
W("mser", "7.8 By the model's own SER on the gap (the 4-gram wins where the model fails completely?)")
w = I[I.win4]
o.append(f"Profile of the {len(w)} gaps the 4-gram wins vs all gaps: novel {100 * w.novel.mean():.1f}% vs {100 * I.novel.mean():.1f}%; mean L {w.L.mean():.1f} vs {I.L.mean():.1f}; "
         f"sux {100 * (w.lang == 'sux').mean():.1f}% vs {100 * (I.lang == 'sux').mean():.1f}%; model SER on them {100 * w.ser_iter.mean():.1f}%, 4-gram SER {100 * w.ser_4gram.mean():.1f}%; "
         f"model had every slot wrong in {100 * (w.ham_iter == 1).mean():.1f}%; 4-gram exact in {100 * w.exact_4gram.mean():.1f}%; numerals share of gold {100 * w.num_share.mean():.1f}% vs {100 * I.num_share.mean():.1f}%.")
big = I[(I.gain4 >= 0.5)]
o.append(f"Large wins (4-gram better by >= 50 points of SER): {len(big)} gaps ({100 * len(big) / len(I):.1f}%), novel {100 * big.novel.mean():.1f}%, mean L {big.L.mean():.1f}.")
# slot-level: where is the 4-gram right and the model wrong
sw = S[S.ok_4 & ~S.ok]
o.append(f"\nSlot level: 4-gram right and model wrong at {len(sw)} slots ({100 * len(sw) / len(S):.1f}%); model right and 4-gram wrong at {int((S.ok & ~S.ok_4).sum())} ({100 * (S.ok & ~S.ok_4).mean():.1f}%). "
         f"Of the 4-gram-only slots: novel {100 * sw.novel.mean():.1f}%; by type: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sw.type.value_counts(normalize=True).items()) +
         "; (all slots by type: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in S.type.value_counts(normalize=True).items()) + ").")
ex = I[I.win4].sort_values("gain4", ascending=False)
JS = {0: " ", 1: "-", 2: ".", 3: ":", 4: "+", 5: "", 6: " / "}


def show(a, lo, hi):
    out = []
    for k in range(lo, hi):
        r = int(a[1, k])
        t = rv[r] if r > 1 else sn[int(a[0, k])]
        if a[3, k]:
            t = "{" + t + "}"
        out.append(t + JS[int(a[2, k])])
    return "".join(out).strip()


o.append("\n### 7.9 Examples (largest margins, then a random sample of wins)\n")
rng = np.random.default_rng(0)
pick = list(ex.i.head(8)) + list(rng.choice(ex.i.values, 12, replace=False))
for i in pick:
    x = inst[i]
    a = ev["chunks"][x["chunk"]][2]
    s, L = x["start"], x["L"]
    pm, p4 = P["known/iter"][i], P["known/4gram"][i]
    r = I.loc[i]
    o.append(f"- [{r.lang}, {r.period}, {r.genre}, L={L}, novel={r.novel}, model SER {100 * r.ser_iter:.0f}, 4-gram SER {100 * r.ser_4gram:.0f}]  \n  left: `{show(a, max(0, s - 10), s)}`  \n  gold: `{show(a, s, s + L)}`  \n"
             f"  model: `{evalset.render(ev, pm['sign'], pm['read'], pm['join'], pm['det'])}`  \n  4-gram: `{evalset.render(ev, p4['sign'], p4['read'], p4['join'], p4['det'])}`  \n  right: `{show(a, s + L, min(a.shape[1], s + L + 8))}`")
open(os.path.join(OUT, "sec7.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
I.to_pickle(OUT + "/I2.pkl")
