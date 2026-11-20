"""Noise audit Q2: within-source inconsistency of readings (TRAIN split of the corpus, CUN_PROC / default v3).

A. Reading entropy per sign given (project, language, period bin); signs whose reading distribution differs
   most between projects inside one (language, period bin) cell: n x I(reading; project | sign, cell), top 50,
   with the dominant readings of the most divergent projects.
B. Convention profile of each project: for every reading pair of the rule tables (T1 safe, T2 scoring-only),
   the tokens of each form per project and language -> which project writes gud and which gu₄.
   A project's profile = share of its pair tokens that use the form CDLI prefers; projects fall into families.
C. Project-convention mass: share of reading tokens that differ from the majority reading of their
   (sign, language, period bin) cell but agree with their own project's majority reading for that cell,
   before and after the safe rules (T1 + determinatives): what a rule per (sign, language) removes and what
   only a 'convention' input (project family) could explain.
Output: results/noise_audit/q2_report.md, q2_tables.json, convention_by_project.json.
"""
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_audit_lib import FIRST_REAL_SIGN, OUT, load_train  # noqa: E402

A, T, LS, M = load_train()
rv, sv, sn = M["read_vocab"], M["sign_vocab"], M["sign_names"]
rid = {r: i for i, r in enumerate(rv)}
sidn = {n: i for i, n in enumerate(sn)}
tx = M["texts"]
proj = np.array([t["src"].replace("oracc:", "") if t["src"] != "cdli" else "CDLI" for t in tx])
proj[np.array([t["src"] == "oracc:cdli" for t in tx])] = "oracc:cdli"
src3 = np.where(proj == "CDLI", "cdli", np.where(proj == "oracc:cdli", "oracc:cdli", "oracc"))
lang = np.where(np.isin(M["t_lang"], ["sux", "akk"]), M["t_lang"], "oth")
pbin = np.array([t.get("period_bin") or "(none)" for t in tx])
ok = (A[0] >= FIRST_REAL_SIGN) & (A[1] > 1)
S, R, D3, TX = A[0][ok], A[1][ok].copy(), A[3][ok], T[ok]
N = len(S)

# ---- canonical readings under the safe rules
safe = json.load(open(os.path.join(OUT, "rules_reading_safe.json")))["rules"]
dets = json.load(open(os.path.join(OUT, "rules_determinatives.json")))["rules"]
Rc = R.copy()
ltok = lang[TX]
ext = list(rv)
for d in dets:
    m = (S == sidn[d["sign"]]) & (D3 > 0) & np.isin(R, [rid[x] for x in d["from"] if x in rid])
    Rc[m] = rid[d["to"]]
for r in safe:
    if r["from"] in rid:
        if r["to"] not in rid:
            rid[r["to"]] = len(ext)
            ext.append(r["to"])
        m = (S == sidn[r["sign"]]) & (R == rid[r["from"]]) & (ltok == r["lang"]) & (Rc == R)
        Rc[m] = rid[r["to"]]

df = pd.DataFrame({"s": S, "r": R, "rc": Rc, "proj": pd.Categorical(proj[TX]), "lang": pd.Categorical(ltok),
                   "pb": pd.Categorical(pbin[TX]), "src": pd.Categorical(src3[TX])})


def ent(c):
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


# ---- C. project-convention mass
def mass(col):
    g = df.groupby(["lang", "pb", "s", "proj", col], observed=True).size().rename("n").reset_index()
    cell = g.groupby(["lang", "pb", "s", col], observed=True).n.sum().rename("n").reset_index()
    cmaj = cell.sort_values("n", ascending=False).drop_duplicates(["lang", "pb", "s"]).rename(columns={col: "cmaj"})[["lang", "pb", "s", "cmaj"]]
    pmaj = g.sort_values("n", ascending=False).drop_duplicates(["lang", "pb", "s", "proj"]).rename(columns={col: "pmaj"})[["lang", "pb", "s", "proj", "pmaj"]]
    g = g.merge(cmaj, on=["lang", "pb", "s"]).merge(pmaj, on=["lang", "pb", "s", "proj"])
    g["k"] = np.where(g[col] == g.cmaj, "cell majority", np.where(g[col] == g.pmaj, "project majority only", "neither"))
    g["src"] = np.where(g.proj == "CDLI", "cdli", np.where(g.proj == "oracc:cdli", "oracc:cdli", "oracc"))
    t = g.groupby(["src", "lang", "k"], observed=True).n.sum().unstack(fill_value=0)
    t2 = g.groupby("k").n.sum()
    return t, t2, g


t_raw, a_raw, g_raw = mass("r")
t_can, a_can, g_can = mass("rc")
conv = pd.DataFrame({"tokens": t_raw.sum(axis=1),
                     "project-majority-only % (raw)": (100 * t_raw["project majority only"] / t_raw.sum(axis=1)).round(2),
                     "after safe rules %": (100 * t_can["project majority only"] / t_can.sum(axis=1)).round(2),
                     "neither % (raw)": (100 * t_raw["neither"] / t_raw.sum(axis=1)).round(2),
                     "neither % after": (100 * t_can["neither"] / t_can.sum(axis=1)).round(2)})
conv_all = {"raw": (100 * a_raw / N).round(3).to_dict(), "after_safe_rules": (100 * a_can / N).round(3).to_dict()}
# per project: its own convention mass, raw and after
pp = g_raw[g_raw.k == "project majority only"].groupby(["proj", "lang"], observed=True).n.sum()
pa = g_can[g_can.k == "project majority only"].groupby(["proj", "lang"], observed=True).n.sum()
pn = g_raw.groupby(["proj", "lang"], observed=True).n.sum()
proj_tab = pd.DataFrame({"tokens": pn, "conv_raw_%": (100 * pp / pn).round(2), "conv_after_%": (100 * pa / pn).round(2)}).fillna(0)
proj_tab = proj_tab[proj_tab.tokens >= 20000].sort_values("conv_raw_%", ascending=False)

# ---- A. n x I(reading; project | sign, cell): top signs
rows = []
gg = df.groupby(["lang", "pb", "s", "proj", "r"], observed=True).size().rename("n").reset_index()
for (lg, pb, s), g in gg.groupby(["lang", "pb", "s"], observed=True):
    n = g.n.sum()
    if n < 200:
        continue
    pj = g.groupby("proj", observed=True).n.sum()
    pj = pj[pj >= 30]
    if len(pj) < 2:
        continue
    g = g[g.proj.isin(pj.index)]
    n = g.n.sum()
    h_all = ent(g.groupby("r").n.sum().values)
    h_cond = sum(pj[p] / n * ent(x.n.values) for p, x in g.groupby("proj", observed=True))
    rows.append((lg, pb, s, int(n), h_all, h_cond, n * (h_all - h_cond)))
MI = pd.DataFrame(rows, columns=["lang", "pb", "s", "n", "H", "H_given_project", "bits"]).sort_values("bits", ascending=False)
top = []
for r in MI.head(50).itertuples():
    g = gg[(gg.lang == r.lang) & (gg.pb == r.pb) & (gg.s == r.s)]
    ex = []
    pj = g.groupby("proj", observed=True).n.sum().sort_values(ascending=False)
    for p in pj.index[:6]:
        x = g[g.proj == p].sort_values("n", ascending=False).head(3)
        ex.append(f"{p}: " + " ".join(f"{rv[a]} {100 * b / pj[p]:.0f}%" for a, b in zip(x.r, x.n)) + f" (n={pj[p]})")
    top.append({"lang": r.lang, "period_bin": r.pb, "sign": sn[r.s], "n": r.n, "H": round(r.H, 3), "H_given_project": round(r.H_given_project, 3),
                "bits_total": round(r.bits), "projects": ex})
# the same after the safe rules (how much of the top list is removed)
bits_raw = MI.bits.sum()
rows = []
gc = df.groupby(["lang", "pb", "s", "proj", "rc"], observed=True).size().rename("n").reset_index()
for (lg, pb, s), g in gc.groupby(["lang", "pb", "s"], observed=True):
    n = g.n.sum()
    if n < 200:
        continue
    pj = g.groupby("proj", observed=True).n.sum()
    pj = pj[pj >= 30]
    if len(pj) < 2:
        continue
    g = g[g.proj.isin(pj.index)]
    n = g.n.sum()
    rows.append(n * (ent(g.groupby("rc").n.sum().values) - sum(pj[p] / n * ent(x.n.values) for p, x in g.groupby("proj", observed=True))))
bits_can = float(np.sum(rows))

# ---- B. convention pairs by project
C = pd.read_pickle(os.path.join(OUT, "candidates_tiered.pkl.gz"))
C = C[C.tier.isin(["T1", "T2a", "T2b"]) & (C.n_a > 0) & (C.n_b > 0) & C.lang.isin(["sux", "akk"])]
cnt = df.groupby(["lang", "s", "r", "proj"], observed=True).size()
byproj, prof = [], collections.defaultdict(lambda: [0, 0])
for r in C.itertuples():
    ra, rb = rid[r.a], rid[r.b]
    try:
        ca = cnt.loc[(r.lang, r.s, ra)]
    except KeyError:
        ca = pd.Series(dtype=int)
    try:
        cb = cnt.loc[(r.lang, r.s, rb)]
    except KeyError:
        cb = pd.Series(dtype=int)
    t = pd.DataFrame({"a": ca, "b": cb}).fillna(0).astype(int)
    t = t[(t.a + t.b) > 0]
    cd = t.loc["CDLI"] if "CDLI" in t.index else None
    cdli_form = None
    if cd is not None and cd.a + cd.b >= 20:
        cdli_form = "a" if cd.a >= cd.b else "b"
        for p, x in t.iterrows():
            if r.tier == "T1":
                prof[(p, r.lang)][0] += int(x[cdli_form])
                prof[(p, r.lang)][1] += int(x.a + x.b)
    byproj.append({"lang": r.lang, "sign": r.sign, "a": r.a, "b": r.b, "tier": r.tier, "relation": r.rel,
                   "cdli_prefers": None if cdli_form is None else (r.a if cdli_form == "a" else r.b),
                   "projects": {p: [int(x.a), int(x.b)] for p, x in t.sort_values("a", ascending=False).iterrows() if x.a + x.b >= 5}})
byproj.sort(key=lambda d: -min(sum(v[0] for v in d["projects"].values()), sum(v[1] for v in d["projects"].values())))
P = pd.DataFrame([{"project": p, "lang": l, "pair_tokens": v[1], "cdli_form_%": round(100 * v[0] / v[1], 1)} for (p, l), v in prof.items() if v[1] >= 200])
P["family"] = np.where(P["cdli_form_%"] >= 80, "CDLI-style (short / index values)", np.where(P["cdli_form_%"] <= 20, "Oracc-style (long values, ŋ, sign names)", "mixed"))
P = P.sort_values(["lang", "cdli_form_%"])
json.dump({"corpus_version": M["version"], "note": "tokens [form a, form b] per project (>= 5), TRAIN split; CDLI = CDLI's own ATF, oracc:cdli = Oracc's copy of it",
           "project_profile": P.to_dict("records"), "pairs": byproj}, open(os.path.join(OUT, "convention_by_project.json"), "w"), ensure_ascii=False, indent=0)
json.dump({"convention_mass_pct_of_tokens": conv_all, "by_source_lang": conv.reset_index().to_dict("records"),
           "by_project": proj_tab.reset_index().to_dict("records"), "mi_bits_total": {"raw": float(bits_raw), "after_safe_rules": bits_can},
           "top50_signs": top}, open(os.path.join(OUT, "q2_tables.json"), "w"), ensure_ascii=False, indent=1, default=float)
o = ["# Q2 within-source inconsistency (train)\n", f"reading tokens {N}\n",
     "## C. tokens that follow their project's majority reading but not the (sign, language, period bin) majority, % of tokens\n",
     json.dumps(conv_all), "\n", conv.to_string(), "\n\n### by project (>= 20k tokens)\n", proj_tab.to_string(),
     f"\n\n## A. n x I(reading; project | sign, language, period bin): total {bits_raw:.0f} bits raw, {bits_can:.0f} after the safe rules\n",
     "### top 50 signs\n"]
for t in top:
    o.append(f"- {t['lang']} {t['period_bin']} {t['sign']} (n={t['n']}, H={t['H']}, H|project={t['H_given_project']}): " + " | ".join(t["projects"]))
o += ["\n## B. convention profile of the projects (T1 pairs; share of the pair tokens in the form CDLI prefers)\n", P.to_string(index=False),
      "\n\n### largest pairs: tokens [a, b] per project\n"]
for d in byproj[:60]:
    o.append(f"- {d['lang']} {d['sign']} {d['a']} / {d['b']} [{d['tier']}]: " + ", ".join(f"{p} {v[0]}/{v[1]}" for p, v in list(d["projects"].items())[:14]))
open(os.path.join(OUT, "q2_report.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
