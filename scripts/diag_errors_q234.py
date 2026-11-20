"""Q2-Q4: per gold sign, within-gap structure, readings (DEV, known width, iterative decoding).
  python scripts/diag_errors_q234.py -> results/diag/errors/sec2.md sec3.md sec4.md"""
import collections
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_errors_lib import NUM, OUT, evalset, load, md, read_diff  # noqa: E402

ev, cache, P = load()
I = pd.read_pickle(OUT + "/I.pkl")
S = pd.read_pickle(OUT + "/S.pkl")
sn, rv = ev["sign_names"], ev["read_vocab"]
lgi = {l: j for j, l in enumerate(cache["langs"])}
fall = cache["f_sign"].sum(0)
o = []


def G(d, by, cols=("ok", "ok_n", "ok_4"), names=("recall iter", "recall 1-pass", "recall 4-gram"), min_n=1, sort=None):
    g = d.groupby(by, observed=True, dropna=False)
    t = pd.DataFrame({"n": g.size()})
    for c, nm in zip(cols, names):
        t[nm] = 100 * g[c].mean()
    t = t[t.n >= min_n]
    if sort:
        t = t.sort_values(sort, ascending=sort != "n")
    return md(t)


# ---------------- Q2
o.append("## 2. Per gold sign\n\nSlot level: a gold sign counts as recalled if the prediction has it at the same slot (positional; 63,000 slots, "
         f"positional error {100 * (1 - S.ok.mean()):.1f}%; the scorer's Levenshtein SER pooled over slots is "
         f"{100 * (I.ser_iter * I.L).sum() / I.L.sum():.1f}%, mean over instances {100 * I.ser_iter.mean():.1f}%).\n")
S["fbin"] = pd.cut(S.f_all, [-1, 99, 999, 9999, 49999, 99999, 299999, 1e9],
                   labels=["<100", "100-999", "1k-10k", "10k-50k", "50k-100k", "100k-300k", ">300k"])
S["fbin_lang"] = pd.cut(S.f_lang, [-1, 99, 999, 9999, 49999, 99999, 299999, 1e9],
                        labels=["<100", "100-999", "1k-10k", "10k-50k", "50k-100k", "100k-300k", ">300k"])
o.append("### 2.1 Recall by training frequency of the gold sign (all train texts, 15.6 M signs)\n\n" + G(S, "fbin"))
nv = np.zeros(len(fall), int)
for b, lo, hi in (("<100", 0, 99), ("100-999", 100, 999), ("1k-10k", 1000, 9999), ("10k-50k", 10000, 49999),
                  ("50k-100k", 50000, 99999), ("100k-300k", 100000, 299999), (">300k", 300000, 10 ** 9)):
    m = (fall >= lo) & (fall <= hi)
    o.append(f"- bin {b}: {int(m[6:].sum())} sign types, {100 * fall[m].sum() / fall.sum():.1f}% of train tokens")
o.append("\n### 2.1b ... by frequency within the text's language\n\n" + G(S, "fbin_lang"))
o.append("\n### 2.1c ... novel gaps only\n\n" + G(S[S.novel], "fbin"))
o.append("\n### 2.1d ... by language\n\n" + G(S[S.lang.isin(["sux", "akk"])], ["lang", "fbin"]))
o.append("\n### 2.2 Recall by sign type (proxy classes, see scripts/diag_errors_lib.sign_type)\n\n" + G(S, "type", sort="n"))
o.append("\n### 2.2b ... novel / non-novel\n\n" + G(S, ["type", "novel"]))
num = S[S.type == "numeral"].copy()
num["same_class"] = [NUM.match(rv[cache_r]) is not None if False else None for cache_r in num.gr]
# is the wrong prediction another numeral sign? (predicted reading is a number notation)
num["pred_is_num"] = [bool(NUM.match(rv[r] if r > 1 else sn[p].lower())) for p, r in zip(num.p, num.pr)]
e = num[~num.ok]
o.append(f"\nNumerals: {len(num)} gold slots, recall {100 * num.ok.mean():.1f}%. Of the {len(e)} errors, "
         f"{100 * e.pred_is_num.mean():.1f}% predict another numeral sign (wrong value or unit), "
         f"{100 * (1 - e.pred_is_num.mean()):.1f}% a non-numeral.")
allp = S[~S.ok]
pin = np.array([bool(NUM.match(rv[r] if r > 1 else sn[p].lower())) for p, r in zip(allp.p, allp.pr)])
o.append(f"Non-numeral gold predicted as a numeral: {int((pin & (allp.type != 'numeral').values).sum())} slots "
         f"({100 * (pin & (allp.type != 'numeral').values).sum() / (S.type != 'numeral').sum():.1f}% of non-numeral gold slots).")
# det
dd = S[S.type == "determinative"]
o.append(f"\nDeterminatives: recall {100 * dd.ok.mean():.1f}% (n={len(dd)}); determinative plane right when the sign is right: "
         f"{100 * (dd[dd.ok].pd == dd[dd.ok].gd).mean():.1f}%.")
# per sign
g = S.groupby("g")
t = pd.DataFrame({"gold n": g.size(), "recall": 100 * g.ok.mean(), "recall novel": 100 * S[S.novel].groupby("g").ok.mean(),
                  "recall 4-gram": 100 * g.ok_4.mean()})
t["predicted n"] = S.groupby("p").size().reindex(t.index).fillna(0).astype(int)
t["precision"] = 100 * S[S.ok].groupby("p").size().reindex(t.index).fillna(0) / t["predicted n"].replace(0, np.nan)
t["train freq"] = fall[t.index.values]
conf = collections.defaultdict(collections.Counter)
for a, b in zip(allp.g, allp.p):
    conf[a][b] += 1
top_read = {}
for a, d in S.groupby("g"):
    top_read[a] = rv[d.gr.value_counts().index[0]]
t["usual reading"] = [top_read[a] for a in t.index]
t["top wrong predictions"] = [", ".join(f"{sn[b]} {n}" for b, n in conf[a].most_common(3)) for a in t.index]
t.index = [sn[a] for a in t.index]
t.index.name = "gold sign"
o.append("\n### 2.3 Frequent signs (>= 150 gold slots) with the lowest recall\n\n" + md(t[t["gold n"] >= 150].sort_values("recall").head(40)))
o.append("\n### 2.3b Frequent signs with the highest recall (for contrast)\n\n" + md(t[t["gold n"] >= 150].sort_values("recall").tail(12)))
o.append("\n### 2.3c Over- and under-predicted signs (predicted n / gold n; gold n >= 150)\n\n")
t["pred/gold"] = t["predicted n"] / t["gold n"]
o.append(md(pd.concat([t[t["gold n"] >= 150].sort_values("pred/gold").head(10), t[t["gold n"] >= 150].sort_values("pred/gold").tail(15)])[
    ["gold n", "predicted n", "pred/gold", "recall", "precision", "usual reading"]], nd=2))


_tr = collections.Counter()
for (_l, _a, _r), _n in cache["sign_read_lang"].items():
    _tr[(_a, _r)] += _n
rshare = {k: v / max(1, fall[k[0]]) for k, v in _tr.items()}


def relation(a, b, lang):
    """Relation of a confusion pair gold a -> predicted b (heuristic)."""
    ra = {rv[r] for r in ev["sign_reads"].get(a, []) if rshare.get((a, r), 0) >= 0.05}      # main values only
    rb = {rv[r] for r in ev["sign_reads"].get(b, []) if rshare.get((b, r), 0) >= 0.05}
    na = any(NUM.match(x) for x in ra) or bool(NUM.match(sn[a].lower()))
    nb = any(NUM.match(x) for x in rb) or bool(NUM.match(sn[b].lower()))
    numa = sum(cache["sign_read_lang"].get((l, a, r), 0) for l in range(len(cache["langs"])) for r in ev["sign_reads"].get(a, []) if NUM.match(rv[r]))
    numb = sum(cache["sign_read_lang"].get((l, b, r), 0) for l in range(len(cache["langs"])) for r in ev["sign_reads"].get(b, []) if NUM.match(rv[r]))
    if numa > 0.5 * fall[a] and numb > 0.5 * fall[b]:
        return "numeral -> numeral"
    import re
    sub = str.maketrans("", "", "₀₁₂₃₄₅₆₇₈₉ₓ")
    ba = {x.translate(sub) for x in ra if not NUM.match(x)}
    bb = {x.translate(sub) for x in rb if not NUM.match(x)}
    comp = lambda s: set(re.split(r"[|.×&%@+()]", s)) - {""}
    ca, cb = comp(sn[a]), comp(sn[b])
    if (len(ca) > 1 or len(cb) > 1) and ca & cb:
        return "graphic (shared component)"
    if ba & bb:
        return "homophone (share a value)"
    return "unrelated"


pairs = collections.Counter(zip(allp.g, allp.p))
gcount = S.g.value_counts()
rel_tot = collections.Counter()
relc = {}
for (a, b), n in pairs.items():
    relc[(a, b)] = relation(a, b, None)
    rel_tot[relc[(a, b)]] += n
rows = [{"gold": sn[a], "predicted": sn[b], "n": n, "% of gold slots": 100 * n / gcount[a], "reverse n": pairs.get((b, a), 0),
         "relation": relc[(a, b)]} for (a, b), n in pairs.most_common(50)]
o.append("\n### 2.4 Top 50 confusion pairs (gold -> predicted, same slot)\n\n" + md(pd.DataFrame(rows), index=False))
o.append(f"\nAll {len(allp)} slot errors by relation of the pair (heuristic: numeral = both signs are mostly number notations in train; "
         "graphic = compound signs sharing a component; homophone = the two signs share a main reading (>= 5% of the sign's train occurrences) up to the index):\n")
o.append(md(pd.DataFrame([{"relation": k, "errors": v, "% of errors": 100 * v / len(allp)} for k, v in rel_tot.most_common()]), index=False))
top50 = sum(n for _, n in pairs.most_common(50))
o.append(f"\nThe top 50 pairs cover {100 * top50 / len(allp):.1f}% of the errors; {len(pairs)} distinct pairs; "
         f"{sum(1 for n in pairs.values() if n == 1)} occur once ({100 * sum(1 for n in pairs.values() if n == 1) / len(allp):.1f}% of errors).")
rows = []
for k in ("numeral -> numeral", "homophone (share a value)", "graphic (shared component)"):
    for (a, b), n in [(p_, n) for p_, n in pairs.most_common() if relc[p_] == k][:15]:
        rows.append({"relation": k, "gold": sn[a], "predicted": sn[b], "n": n, "% of gold slots": 100 * n / gcount[a]})
o.append("\n### 2.4b Top pairs within each related class\n\n" + md(pd.DataFrame(rows), index=False))
# odd targets
odd = S[S.g.map(lambda a: sn[a] in ("X", "P₁", "P₂", "P₃", "P₄"))]
o.append(f"\nGold slots whose sign is the placeholder-like 'X' or a punctuation sign (P1..P4): {len(odd)} "
         f"({100 * len(odd) / len(S):.2f}% of slots), recall {100 * odd.ok.mean():.1f}%; train frequency of 'X' as a real sign id: "
         f"{int(fall[sn.index('X')])}.")
open(os.path.join(OUT, "sec2.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))

# ---------------- Q3
o = ["## 3. Within gaps\n"]
S["pos"] = np.where(S.L == 1, "single", np.where(S.k == 0, "first", np.where(S.k == S.L - 1, "last", "middle")))
o.append("### 3.1 Recall by slot position\n\n" + G(S, "pos"))
S["edge"] = np.minimum(S.k, S.L - 1 - S.k)
S["edgeb"] = pd.cut(S.edge, [-1, 0, 1, 2, 4, 9], labels=["0 (touches context)", "1", "2", "3-4", "5-9"])
o.append("\n### 3.1b Recall by distance to the nearest gap edge (slots between the sign and known context)\n\n" + G(S, "edgeb"))
S["Lbin"] = pd.cut(S.L, [0, 1, 5, 10, 15, 20], labels=["1", "2-5", "6-10", "11-15", "16-20"])
o.append("\n### 3.1c ... by gap length\n\n" + G(S, ["Lbin", "edgeb"]))
S["kb"] = pd.cut(S.k, [-1, 0, 1, 2, 4, 9, 14, 19], labels=["0", "1", "2", "3-4", "5-9", "10-14", "15-19"])
S["kend"] = pd.cut(S.L - 1 - S.k, [-1, 0, 1, 2, 4, 9, 14, 19], labels=["0", "1", "2", "3-4", "5-9", "10-14", "15-19"])
o.append("\n### 3.1d Recall by distance from the gap start (L >= 6)\n\n" + G(S[S.L >= 6], "kb"))
o.append("\n### 3.1e Recall by distance from the gap end (L >= 6)\n\n" + G(S[S.L >= 6], "kend"))
# clustering
ok = {i: d.sort_values("k").ok.values for i, d in S.groupby("i")}
prev_ok = []
for i, v in ok.items():
    prev_ok.extend([np.nan] + list(v[:-1].astype(float)))
S = S.sort_values(["i", "k"])
S["prev_ok"] = prev_ok
d = S[S.k > 0]
o.append("\n### 3.2 Error clustering: P(error at slot k | slot k-1)\n")
rows = []
for name, sub in (("all gaps L>=2", d), ("novel", d[d.novel]), ("non-novel", d[~d.novel]), ("L 2-5", d[d.L <= 5]), ("L 6-10", d[(d.L >= 6) & (d.L <= 10)]), ("L 11-20", d[d.L >= 11])):
    rows.append({"subset": name, "n": len(sub), "P(err)": 100 * (1 - sub.ok.mean()), "P(err | previous wrong)": 100 * (1 - sub[sub.prev_ok == 0].ok.mean()),
                 "P(err | previous right)": 100 * (1 - sub[sub.prev_ok == 1].ok.mean())})
o.append(md(pd.DataFrame(rows), index=False))
runs_e, runs_tot = collections.Counter(), 0
allwrong = allright = 0
for i, v in ok.items():
    if len(v) < 2:
        continue
    allwrong += (~v).all()
    allright += v.all()
    k = 0
    while k < len(v):
        if not v[k]:
            j = k
            while j < len(v) and not v[j]:
                j += 1
            runs_e[min(j - k, 10)] += j - k
            k = j
        else:
            k += 1
tot_e = sum(runs_e.values())
o.append("\nErrors by length of the run of consecutive wrong slots they sit in (L >= 2):\n\n" +
         md(pd.DataFrame([{"run length": (str(k) if k < 10 else "10+"), "wrong slots": n, "% of wrong slots": 100 * n / tot_e} for k, n in sorted(runs_e.items())]), index=False))
o.append(f"\nGaps (L >= 2, n = {sum(1 for v in ok.values() if len(v) > 1)}): every slot wrong in {allwrong}, every slot right in {allright}.")
# within an instance: distribution of the per-gap error share by L bin (bimodality)
I["hb"] = pd.cut(I.ham_iter, [-0.01, 0, 0.25, 0.5, 0.75, 0.999, 1], labels=["0", "0-25%", "25-50%", "50-75%", "75-<100%", "100%"])
I["Lbin"] = pd.cut(I.L, [0, 1, 5, 10, 15, 20], labels=["1", "2-5", "6-10", "11-15", "16-20"])
ct = pd.crosstab(I.Lbin, I.hb, normalize="index") * 100
o.append("\n### 3.2b Share of the gap's slots that are wrong (% of gaps per L bin)\n\n" + md(ct))
# first error position: once the first error happens, error rate after
rows = []
for i, v in ok.items():
    if len(v) >= 6 and not v.all():
        f = int(np.argmin(v))
        rows.append((len(v), f, (~v[f + 1:]).mean() if f + 1 < len(v) else np.nan, (~v[:f]).mean() if f else np.nan))
r = pd.DataFrame(rows, columns=["L", "first", "after", "before"])
o.append(f"\nGaps with L >= 6 and at least one error (n = {len(r)}): error rate of the slots after the first wrong slot "
         f"{100 * r.after.mean():.1f}% (slots before it are right by definition).")
# shifts
sh_slots = tot_wrong = inst_shift = 0
shift_hist = collections.Counter()
bag_rec = []
exact_shift = collections.Counter()
for i, x in enumerate(ev["instances"]):
    gs = evalset.gold(ev, x)[0].tolist()
    ps = [int(v) for v in P["known/iter"][i]["sign"]]
    L = x["L"]
    cg, cp = collections.Counter(gs), collections.Counter(ps)
    bag_rec.append((sum((cg & cp).values()), L, sum(a == b for a, b in zip(gs, ps))))
    if L < 2 or gs == ps:
        continue
    any_s = False
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ps, gs):
        if tag == "equal" and i1 != j1:
            sh_slots += i2 - i1
            shift_hist[max(-4, min(4, i1 - j1))] += i2 - i1
            any_s = True
    tot_wrong += sum(a != b for a, b in zip(gs, ps))
    inst_shift += any_s
    for dlt in (1, 2, 3, -1, -2, -3):                    # prediction = gold moved by dlt slots over >= half of the gap
        if dlt > 0:
            m = sum(a == b for a, b in zip(ps[dlt:], gs[:-dlt]))
        else:
            m = sum(a == b for a, b in zip(ps[:dlt], gs[-dlt:]))
        if m >= max(2, 0.5 * L) and m > sum(a == b for a, b in zip(gs, ps)):
            exact_shift[dlt] += 1
            break
br = np.array(bag_rec)
o.append("\n### 3.3 Alignment (shift) errors\n")
o.append(f"- Positional (Hamming) error over all slots {100 * (1 - S.ok.mean()):.1f}% vs Levenshtein SER pooled over slots "
         f"{100 * (I.ser_iter * I.L).sum() / I.L.sum():.1f}%: the scorer already forgives shifts; the difference is the share of slots that are right but displaced.")
o.append(f"- Wrong slots (positional) that the optimal alignment matches to a gold sign at another position: {sh_slots} of {tot_wrong} "
         f"({100 * sh_slots / tot_wrong:.1f}%); {inst_shift} gaps ({100 * inst_shift / (I.L >= 2).sum():.1f}% of gaps with L >= 2) contain such a displaced stretch.")
o.append("- Displacement (predicted position minus gold position) of those slots: " + ", ".join(f"{k:+d}{'+' if abs(k) == 4 else ''}: {v}" for k, v in sorted(shift_hist.items())))
o.append(f"- Gaps where the prediction matches the gold moved by 1-3 slots over at least half of the gap (and better than unshifted): "
         f"{sum(exact_shift.values())} ({100 * sum(exact_shift.values()) / (I.L >= 2).sum():.1f}% of gaps L >= 2); by shift " + str(dict(sorted(exact_shift.items()))))
o.append(f"- Bag-of-signs recall (gold signs present anywhere in the predicted gap, multiset): {100 * br[:, 0].sum() / br[:, 1].sum():.1f}% vs positional recall "
         f"{100 * br[:, 2].sum() / br[:, 1].sum():.1f}%: the right signs in the wrong place (or order) are {100 * (br[:, 0].sum() - br[:, 2].sum()) / br[:, 1].sum():.1f}% of slots.")
# repetition
rep_g = rep_p = n_b = 0
for i, x in enumerate(ev["instances"]):
    gs = evalset.gold(ev, x)[0]
    ps = np.asarray(P["known/iter"][i]["sign"])
    rep_g += int((gs[1:] == gs[:-1]).sum())
    rep_p += int((ps[1:] == ps[:-1]).sum())
    n_b += len(gs) - 1
o.append(f"- Immediate repetitions (same sign twice in a row) inside gaps: gold {100 * rep_g / n_b:.2f}% of bigrams, iterative prediction {100 * rep_p / n_b:.2f}%.")
# predicted word boundaries / line ends inside the gap
o.append("\n### 3.4 Structure planes inside the gap (all slots)\n")
S["gb"] = S.gj.isin([0, 6])
S["pb"] = S.pj.isin([0, 6])
rows = []
for name, sub in (("all slots", S), ("sign right", S[S.ok]), ("sign wrong", S[~S.ok])):
    tp = (sub.gb & sub.pb).sum(); fp = (~sub.gb & sub.pb).sum(); fn = (sub.gb & ~sub.pb).sum()
    rows.append({"slots": name, "n": len(sub), "joiner exact": 100 * (sub.gj == sub.pj).mean(), "word-end P": 100 * tp / max(1, tp + fp), "word-end R": 100 * tp / max(1, tp + fn),
                 "line-end R": 100 * (sub[sub.gj == 6].pj == 6).mean(), "line-end P": 100 * (sub[sub.pj == 6].gj == 6).mean(), "det exact": 100 * (sub.gd == sub.pd).mean()})
o.append(md(pd.DataFrame(rows), index=False))
I["g_words"] = S.groupby("i").gb.sum().reindex(I.i).values
I["p_words"] = S.groupby("i").pb.sum().reindex(I.i).values
I["g_lines"] = S.assign(v=S.gj == 6).groupby("i").v.sum().reindex(I.i).values
I["p_lines"] = S.assign(v=S.pj == 6).groupby("i").v.sum().reindex(I.i).values
I["lines_ok"] = I.g_lines == I.p_lines
I["words_ok"] = I.g_words == I.p_words
m = I.L >= 6
o.append(f"\nGaps L >= 6 (n = {int(m.sum())}): number of line ends in the gap predicted exactly in {100 * I[m].lines_ok.mean():.1f}%, number of word ends in {100 * I[m].words_ok.mean():.1f}%. "
         f"SER when the number of line ends is right {100 * I[m & I.lines_ok].ser_iter.mean():.1f}% (n={int((m & I.lines_ok).sum())}) vs wrong "
         f"{100 * I[m & ~I.lines_ok].ser_iter.mean():.1f}% (n={int((m & ~I.lines_ok).sum())}); when the number of word ends is right "
         f"{100 * I[m & I.words_ok].ser_iter.mean():.1f}% (n={int((m & I.words_ok).sum())}) vs wrong {100 * I[m & ~I.words_ok].ser_iter.mean():.1f}% (n={int((m & ~I.words_ok).sum())}).")
open(os.path.join(OUT, "sec3.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))

# ---------------- Q4
o = ["## 4. Readings inside gaps\n"]
R = S[(S.gr > 1)].copy()
Rk = R[R.ok]
o.append(f"Slots whose gold sign has a reading: {len(R)}; sign right: {len(Rk)}. Reading right given the sign is right: "
         f"{100 * (Rk.gr == Rk.pr).mean():.2f}% (wrong: {int((Rk.gr != Rk.pr).sum())} slots = {100 * (Rk.gr != Rk.pr).mean():.2f}%). "
         f"Overall reading accuracy {100 * (R.gr == R.pr).mean():.1f}% is therefore almost entirely the sign accuracy.\n")
Rk = Rk.assign(rok=Rk.gr == Rk.pr)
g = Rk.groupby("type")
o.append("### 4.1 Reading error given a right sign, by sign type\n\n" + md(pd.DataFrame({"n": g.size(), "reading wrong %": 100 * (1 - g.rok.mean())})))
g = Rk.groupby("lang")
o.append("\n### 4.1b ... by language\n\n" + md(pd.DataFrame({"n": g.size(), "reading wrong %": 100 * (1 - g.rok.mean())})))
W = Rk[~Rk.rok]
W = W.assign(cls=[read_diff(rv[a], rv[b]) for a, b in zip(W.gr, W.pr)])
o.append("\n### 4.2 Class of the reading error (sign right)\n\n" + md(pd.DataFrame({"n": W.cls.value_counts(), "%": 100 * W.cls.value_counts(normalize=True)})))
c = collections.Counter(zip(W.g, W.gr, W.pr))
gs_n = Rk.groupby("g").size()
rows = [{"sign": sn[a], "gold reading": rv[b], "predicted": rv[p_], "n": n, "class": read_diff(rv[b], rv[p_])} for (a, b, p_), n in c.most_common(40)]
o.append("\n### 4.3 Top reading confusions (sign right)\n\n" + md(pd.DataFrame(rows), index=False))
g = Rk.groupby("g")
t = pd.DataFrame({"n (sign right)": g.size(), "reading wrong": (~g.rok.mean().isna()) * (g.size() - g.rok.sum()), "reading wrong %": 100 * (1 - g.rok.mean())})
t.index = [sn[a] for a in t.index]
o.append("\n### 4.3b Signs with most reading errors\n\n" + md(t.sort_values("reading wrong", ascending=False).head(25)))
# constraint
adm = ev["sign_reads"]
R["gold_adm"] = [(r in adm.get(a, [r])) for a, r in zip(R.g, R.gr)]
ex = R[~R.gold_adm]
o.append(f"\n### 4.4 The 'attested readings' constraint\n\nGold (sign, reading) pairs not admissible under the constraint (reading attested >= 2 times for the sign in train): "
         f"{len(ex)} of {len(R)} slots ({100 * len(ex) / len(R):.2f}%); among slots with the sign right: {int((~R[R.ok].gold_adm).sum())} of {int(R.ok.sum())} "
         f"({100 * (~R[R.ok].gold_adm).mean():.2f}%), i.e. the constraint can cost at most that much reading accuracy.")
trc = collections.Counter()
for (lg, a, r), n in cache["sign_read_lang"].items():
    trc[(a, r)] += n
ex = ex.assign(train_n=[trc.get((a, r), 0) for a, r in zip(ex.g, ex.gr)],
               read_other=[sum(n for (a2, r2), n in ()) for _ in ex.g])
rtot = collections.Counter()
for (a, r), n in trc.items():
    rtot[r] += n
ex["read_train_any_sign"] = [rtot.get(r, 0) for r in ex.gr]
o.append(f"- pair never seen in train: {int((ex.train_n == 0).sum())}; seen once: {int((ex.train_n == 1).sum())}; seen >= 2 (should not happen): {int((ex.train_n >= 2).sum())}")
o.append(f"- of the never-seen pairs, the reading exists in train with another sign: {int(((ex.train_n == 0) & (ex.read_train_any_sign > 0)).sum())}; reading never in train at all: {int(((ex.train_n == 0) & (ex.read_train_any_sign == 0)).sum())}")
o.append("- by language: " + str(ex.lang.value_counts().to_dict()) + "; by type: " + str(ex.type.value_counts().to_dict()))
cc = collections.Counter((sn[a], rv[r]) for a, r in zip(ex.g, ex.gr))
o.append("- most frequent excluded gold pairs: " + ", ".join(f"{a}={r} ({n})" for (a, r), n in cc.most_common(25)))
# sign wrong, same rendered reading
E = R[~R.ok]
same = E[[rv[a] == rv[b] for a, b in zip(E.gr, E.pr)]]
base = E[[read_diff(rv[a], rv[b]) in ("index",) for a, b in zip(E.gr, E.pr)]]
o.append(f"\n### 4.5 Sign wrong but reading the same\n\nWrong-sign slots whose predicted reading string equals the gold reading: {len(same)} of {len(E)}; "
         f"same letters with another index (a homophone sign: u / u₂ / u₃): {len(base)} ({100 * len(base) / len(E):.1f}% of sign errors).")
cc = collections.Counter((rv[a], rv[b]) for a, b in zip(base.gr, base.pr))
o.append("Most frequent homophone substitutions (gold -> predicted): " + ", ".join(f"{a}->{b} ({n})" for (a, b), n in cc.most_common(30)))
open(os.path.join(OUT, "sec4.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
