"""Extra checks (DEV): in-document copying, blank probability when the slots are exact, placeholder signs.
  python scripts/diag_errors_extra.py -> results/diag/errors/sec10.md"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_errors_lib import OUT, evalset, load, md, pk, src_norm  # noqa: E402

ev, cache, P = load()
I = pd.read_pickle(OUT + "/I2.pkl")
S = pd.read_pickle(OUT + "/S.pkl").sort_values(["i", "k"]).reset_index(drop=True)
A = pk(os.path.join(OUT, "gpu_probs.pkl.gz"))["A"]
inst = ev["instances"]
sn = ev["sign_names"]
o = ["## 10. Extra checks\n"]
# (a) copy from the same chunk
rep, rep_slots = [], []
for i, x in enumerate(inst):
    a = ev["chunks"][x["chunk"]][2][0].astype(np.int64)
    s, L = x["start"], x["L"]
    g = a[s:s + L]
    ctx = np.r_[a[:s], [-1], a[s + L:]]
    found = False
    if L >= 2:
        for j in range(len(ctx) - L + 1):
            if np.array_equal(ctx[j:j + L], g):
                found = True
                break
    rep.append(found)
    # slot level: the trigram (previous sign, sign, next sign) occurs elsewhere in the chunk
    tri = set()
    for j in range(1, len(ctx) - 1):
        tri.add((ctx[j - 1], ctx[j], ctx[j + 1]))
    for k in range(L):
        p = s + k
        t = (a[p - 1] if p > 0 else -2, a[p], a[p + 1] if p + 1 < len(a) else -2)
        rep_slots.append(t in tri)
I["rep"] = rep
S["tri_in_chunk"] = rep_slots
I["Lb"] = pd.cut(I.L, [1, 3, 6, 10, 20], labels=["2-3", "4-6", "7-10", "11-20"])
g = I[I.L >= 2].groupby(["Lb", "novel", "rep"], observed=True)
o.append("### 10.1 Copying from the same chunk\n\nThe gold gap (L >= 2) occurs verbatim elsewhere in the visible chunk ('rep'): does the model use it?\n\n" +
         md(pd.DataFrame({"n": g.size(), "SER": 100 * g.ser_iter.mean(), "exact": 100 * g.exact_iter.mean(), "4-gram SER": 100 * g.ser_4gram.mean()})))
g = S.groupby(["novel", "tri_in_chunk"])
o.append("\nSlot level: the trigram (previous sign, gold sign, next sign) occurs elsewhere in the same chunk:\n\n" +
         md(pd.DataFrame({"slots": g.size(), "recall iter": 100 * g.ok.mean(), "recall 1-pass": 100 * g.ok_n.mean(), "recall 4-gram": 100 * g.ok_4.mean()})))
# (b) blank probability when the offered slots are exactly the true width
rows = []
for lb, sub in I.groupby("Lbin", observed=True):
    for e in (0, 1, 2, 3):
        s2 = sub[sub.extra == e]
        if len(s2) < 20:
            continue
        real = np.concatenate([A[i]["pB"][:inst[i]["L"]] for i in s2.i])
        last = np.array([A[i]["pB"][inst[i]["L"] - 1] for i in s2.i])
        fin = np.array([A[i]["pB"][-1] for i in s2.i])
        rows.append({"L": lb, "extra slots offered": e, "n gaps": len(s2), "mean p(BLANK), real slots": real.mean(), "p(BLANK) last real slot": last.mean(),
                     "p(BLANK) final offered slot": fin.mean(), "any real slot with p(BLANK) > 0.5 (% gaps)": 100 * np.mean([(A[i]["pB"][:inst[i]["L"]] > 0.5).any() for i in s2.i])})
o.append("\n### 10.2 The model cannot tell exact slots from elastic slots\n\nA known-width gap and an unknown-width gap with 0 extra slots are the same input (a run of MASK). "
         "One-pass p(BLANK) at slots that hold a sign:\n\n" + md(pd.DataFrame(rows), index=False, nd=3))
k0 = I[I.extra == 0]
o.append(f"\nGaps offered exactly L slots under the unknown-width decoders (n = {len(k0)}): width right in {100 * (k0.d_indep == 0).mean():.1f}% (indep), {100 * (k0['d_iter-w'] == 0).mean():.1f}% (iter-w); "
         f"CER {100 * k0.ucer_indep.mean():.1f} (indep), {100 * k0['ucer_iter-w'].mean():.1f} (iter-w) vs known-width {100 * k0.cer_indep.mean():.1f} / {100 * k0.cer_iter.mean():.1f}.")
# (c) placeholder signs
X = sn.index("X")
fall = cache["f_sign"].sum(0)
rows = []
for name in ("X", "N", "P₁", "P₂"):
    if name in sn:
        j = sn.index(name)
        rows.append({"sign id": name, "train tokens": int(fall[j]), "% of train": 100 * fall[j] / fall.sum(), "akk": int(cache["f_sign"][cache["langs"].index("akk"), j]),
                     "sux": int(cache["f_sign"][cache["langs"].index("sux"), j]), "gold slots (dev gaps)": int((S.g == j).sum()), "predicted slots (iter)": int((S.p == j).sum()),
                     "predicted and wrong": int(((S.p == j) & (S.g != j)).sum()), "predicted slots (4-gram)": int((S.p4 == j).sum())})
o.append("\n### 10.3 Placeholders that are real sign ids\n\n'X' is Oracc's illegible sign (reading 'x'); CDLI's 'x' is the special <x> (never a target). 'N' is the unspecified numeral n.\n\n" + md(pd.DataFrame(rows), index=False, nd=2))
c = {}
for ti, lg, a in ev["chunks"]:
    s = src_norm(cache["dev_meta"][ti]["src"])
    d = c.setdefault(s, np.zeros(5, int))
    d += [a.shape[1], int((a[0] == X).sum()), int(((a[0] == X) & (a[4] == 0)).sum()), int((a[0] == 4).sum()), int((a[0] == 5).sum())]
o.append("\nDev chunks by source:\n\n" + md(pd.DataFrame([{"source": k, "signs": v[0], "'X' sign": v[1], "'X' not flagged damaged (eligible as gold)": v[2], "<x>": v[3], "<gap>": v[4]} for k, v in c.items()]), index=False))
ix = S[(S.p == X) & (S.g != X)].i.unique()
d = I.set_index("i").loc[ix]
o.append(f"\nGaps in which the model predicts 'X' wrongly: {len(d)} ({100 * len(d) / len(I):.1f}%), by source {d.src.value_counts().to_dict()}, SER {100 * d.ser_iter.mean():.1f}; "
         f"gaps whose gold contains 'X': {S[S.g == X].i.nunique()} (unanswerable slots: {int((S.g == X).sum())}).")
orc = I[I.src == "oracc"]
hasx = orc.i.isin(ix)
o.append(f"Oracc gaps: {len(orc)}; with a wrong X prediction {int(hasx.sum())} ({100 * hasx.mean():.1f}%).")
open(os.path.join(OUT, "sec10.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
