"""Noise audit, step 2: candidate convention pairs (sign, reading a, reading b) with their evidence.

Candidates: (i) every reading pair observed on sign-matched positions of both-source texts (Oracc curated
projects vs CDLI; Oracc's own copy of CDLI, project 'cdli', is left out); (ii) every pair of train readings of
one sign that differ formally only by index, by the ŋ / geminate convention, or as long vs short form.
Evidence per pair and language (text language sux / akk), TRAIN split only:
  n_a, n_b          tokens of each reading on the sign, by source group (cdli, oracc:cdli, oracc curated)
  cooc              share of the rarer reading's tokens that stand in a text which also has the other reading
                    (editors are consistent within a text: low = the two never contrast = convention;
                     high = one editor uses both = a real distinction)
  pair_ab / pair_ba both-source positions with Oracc a | CDLI b and Oracc b | CDLI a
  agree_a, agree_b  both-source: of the positions where one side reads a (b), the share where the other side
                    reads a or b (how often swapping a for b reproduces the other edition)
  osl               both values listed for the sign in OSL; deprecated (@v-); Attinger @sys mapping
Output: results/noise_audit/candidates.pkl.gz (DataFrame), candidates_top.tsv.
"""
import collections
import gzip
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_audit_lib import FIRST_REAL_SIGN, OUT, load_train, parse_osl, read_class  # noqa: E402

A, T, LS, M = load_train()
rv, sv, sn = M["read_vocab"], M["sign_vocab"], M["sign_names"]
ok = (A[0] >= FIRST_REAL_SIGN) & (A[1] > 1)
S, R, TX = A[0][ok].astype(np.int64), A[1][ok].astype(np.int64), T[ok].astype(np.int64)
NTOK = len(S)
src3 = np.where(M["t_proj"] == "cdli", "cdli", np.where(M["t_proj"] == "oracc:cdli", "oracc:cdli", "oracc"))
lang2 = np.where(np.isin(M["t_lang"], ["sux", "akk"]), M["t_lang"], "oth")
# (sign, read, text) counts
key = (S * len(rv) + R) * len(M["texts"]) + TX
u, cnt = np.unique(key, return_counts=True)
ut = u % len(M["texts"])
usr = u // len(M["texts"])
D = pd.DataFrame({"s": usr // len(rv), "r": usr % len(rv), "t": ut, "n": cnt})
D["lang"] = lang2[D.t.values]
D["src"] = src3[D.t.values]
print("sign-read-text cells", len(D), "tokens", NTOK, flush=True)
tot = D.groupby(["s", "r", "lang"]).n.sum()
tot_src = D.groupby(["s", "r", "lang", "src"]).n.sum()

# ---- both-source pairs
P = pd.read_pickle(os.path.join(OUT, "pairs_matched.pkl.gz"))
PT = pickle.load(gzip.open(os.path.join(OUT, "pairs_texts.pkl.gz"), "rb"))["texts"]
for c in ("oid", "o_read", "c_read", "tid"):
    P[c] = P[c].astype(str)
P = P[~P.oid.str.startswith("<")]
P["proj"] = P.tid.map({k: v["proj"] for k, v in PT.items()})
P["lang"] = P.tid.map({k: v["lang"] for k, v in PT.items()})
P.loc[~P.lang.isin(["sux", "akk"]), "lang"] = "oth"
P = P[P.proj != "cdli"]
sid = {o: i for i, o in enumerate(sv)}
rid = {r: i for i, r in enumerate(rv)}
pc = P.groupby(["lang", "oid", "o_read", "c_read"]).size()
side_o = P.groupby(["lang", "oid", "o_read"]).size().to_dict()
side_c = P.groupby(["lang", "oid", "c_read"]).size().to_dict()
pcd = pc.to_dict()

cands = collections.defaultdict(set)          # (lang, sign idx, a, b) with a < b -> origins
for (lang, oid, a, b), n in pcd.items():
    if a != b and n >= 2 and oid in sid and a and b:
        cands[(lang, sid[oid], *sorted((a, b)))].add("mined")
by_sign = collections.defaultdict(list)
for (s, r, lang), n in tot.items():
    if n >= 5:
        by_sign[(lang, s)].append(rv[r])
for (lang, s), rs in by_sign.items():
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if read_class(rs[i], rs[j]) in ("index", "eng", "auslaut"):
                cands[(lang, s, *sorted((rs[i], rs[j])))].add("formal")
print("candidates", len(cands), flush=True)

osl, sysmap = parse_osl()
sysset = {(o, a, b) for o, a, b in sysmap} | {(o, b, a) for o, a, b in sysmap}
# per (lang, sign, read): text -> n
Dl = D.set_index(["lang", "s", "r"]).sort_index()
rows = []
for (lang, s, a, b), orig in cands.items():
    ra, rb = rid.get(a), rid.get(b)
    na = int(tot.get((s, ra, lang), 0)) if ra is not None else 0
    nb = int(tot.get((s, rb, lang), 0)) if rb is not None else 0
    cooc = np.nan
    if na and nb:
        ta = Dl.loc[(lang, s, ra)].set_index("t").n
        tb = Dl.loc[(lang, s, rb)].set_index("t").n
        minor, other = (ta, tb) if na <= nb else (tb, ta)
        cooc = float(minor[minor.index.isin(other.index)].sum() / minor.sum())
    oid = sv[s]
    pab, pba = pcd.get((lang, oid, a, b), 0), pcd.get((lang, oid, b, a), 0)
    paa, pbb = pcd.get((lang, oid, a, a), 0), pcd.get((lang, oid, b, b), 0)
    so_a, sc_a = side_o.get((lang, oid, a), 0), side_c.get((lang, oid, a), 0)
    so_b, sc_b = side_o.get((lang, oid, b), 0), side_c.get((lang, oid, b), 0)
    agree_a = (2 * paa + pab + pba) / (so_a + sc_a) if so_a + sc_a else np.nan
    agree_b = (2 * pbb + pab + pba) / (so_b + sc_b) if so_b + sc_b else np.nan
    v = osl.get(oid, {}).get("values", {})
    row = {"lang": lang, "sign": sn[s], "s": s, "a": a, "b": b, "cls": read_class(a, b), "origin": "+".join(sorted(orig)),
           "n_a": na, "n_b": nb, "cooc": cooc, "pair_ab": pab, "pair_ba": pba, "pair_aa": paa, "pair_bb": pbb,
           "agree_a": agree_a, "agree_b": agree_b,
           "osl_a": a in v, "osl_b": b in v, "osl_dep": ("dep" in v.get(a, ())) or ("dep" in v.get(b, ())),
           "osl_sys": (oid, a, b) in sysset}
    for g in ("cdli", "oracc:cdli", "oracc"):
        row["a_" + g] = int(tot_src.get((s, ra, lang, g), 0)) if ra is not None else 0
        row["b_" + g] = int(tot_src.get((s, rb, lang, g), 0)) if rb is not None else 0
    rows.append(row)
C = pd.DataFrame(rows)
C["minor"] = C[["n_a", "n_b"]].min(axis=1)
C["pairs"] = C.pair_ab + C.pair_ba
C = C.sort_values(["minor", "pairs"], ascending=False)
C.to_pickle(os.path.join(OUT, "candidates.pkl.gz"))
C.head(1500).to_csv(os.path.join(OUT, "candidates_top.tsv"), sep="\t", index=False, float_format="%.3f")
pickle.dump({"ntok": NTOK}, open(os.path.join(OUT, "cache/ntok.pkl"), "wb"))
print(C.cls.value_counts())
