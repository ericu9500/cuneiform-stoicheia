"""Noise audit Q3-Q5: join / determinative planes, sign plane, damage flags.

Train split of the corpus (CUN_PROC, default v3) for the per-source shares; the both-source alignment
(noise_audit_pairs.py; Oracc curated projects vs CDLI, train texts) for position-level comparisons.
Output: results/noise_audit/q345_report.md, q345_tables.json.
"""
import collections
import gzip
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_audit_lib import FIRST_REAL_SIGN, NUM, OUT, load_train  # noqa: E402

A, T, LS, M = load_train()
rv, sv, sn, joins = M["read_vocab"], M["sign_vocab"], M["sign_names"], M["joins"]
tx = M["texts"]
src3 = np.where(M["t_proj"] == "cdli", "cdli", np.where(M["t_proj"] == "oracc:cdli", "oracc:cdli", "oracc"))
lang = np.where(np.isin(M["t_lang"], ["sux", "akk"]), M["t_lang"], "oth")
pbin = np.array([t.get("period_bin") or "(none)" for t in tx])
N = A.shape[1]
S, R, J, D, B = A
g_src, g_lang, g_pb = src3[T], lang[T], pbin[T]
res, o = {}, ["# Q3-Q5: joins, determinatives, signs, damage\n"]


def pct(df, by, col):
    t = df.groupby(by + [col], observed=True).size().unstack(fill_value=0)
    out = (100 * t.div(t.sum(axis=1), axis=0)).round(2)
    out.insert(0, "n", t.sum(axis=1))
    return out


# ---------------- Q3 joins
F = pd.DataFrame({"src": g_src, "lang": g_lang, "pb": g_pb, "join": pd.Categorical.from_codes(J, ["w", "-", ".", ":", "+", "''"]),
                  "det": pd.Categorical.from_codes(D, ["none", "pre", "post"]), "brk": pd.Categorical.from_codes(B, ["intact", "damaged", "missing"])})
real = S >= FIRST_REAL_SIGN
jt = pct(F[real], ["src", "lang"], "join")
dt = pct(F[real], ["src", "lang"], "det")
# word length (signs per word) per source and language
line_id = np.zeros(N, np.int64)
line_id[LS[1:-1]] = 1
line_id = np.cumsum(line_id)
wend = (J == 0)
wend[LS[1:] - 1] = True
wid = np.concatenate([[0], np.cumsum(wend)[:-1]])
wl = pd.DataFrame({"src": g_src, "lang": g_lang, "w": wid}).groupby(["src", "lang"]).agg(signs=("w", "size"), words=("w", "nunique"))
wl["signs_per_word"] = (wl.signs / wl.words).round(3)
o += ["## Q3 joiner after each sign, % by source and language (train)\n", jt.to_string(), "\n\nsigns per word\n", wl.to_string(),
      "\n\ndeterminative flag, % of signs\n", dt.to_string()]
res["join_share"] = jt.reset_index().to_dict("records")
res["signs_per_word"] = wl.reset_index().to_dict("records")
res["det_share"] = dt.reset_index().to_dict("records")

P = pd.read_pickle(os.path.join(OUT, "pairs_matched.pkl.gz"))
PT = pickle.load(gzip.open(os.path.join(OUT, "pairs_texts.pkl.gz"), "rb"))
names, PTX = PT["names"], PT["texts"]
for c in ("oid", "o_read", "c_read", "tid", "o_det", "c_det", "o_join", "c_join", "o_brk", "c_brk"):
    P[c] = P[c].astype(str)
P["proj"] = P.tid.map({k: v["proj"] for k, v in PTX.items()})
P["lang"] = P.tid.map({k: v["lang"] for k, v in PTX.items()})
P.loc[~P.lang.isin(["sux", "akk"]), "lang"] = "oth"
Pall = P[P.proj != "cdli"]
P = Pall[~Pall.oid.str.startswith("<")].copy()
P["sign"] = P.oid.map(names)
for c in ("o_join", "c_join"):
    P[c] = P[c].replace({"": "''", "—": "-", "–": "-", "/": "-"})
jc = pd.crosstab([P.lang, P.o_join], P.c_join)
jdis = P[P.o_join != P.c_join]
res["both_source_join"] = {"positions": int(len(P)), "exact_agreement_pct": round(100 * float((P.o_join == P.c_join).mean()), 2),
                           "word_boundary_agreement_pct": round(100 * float(((P.o_join == "w") == (P.c_join == "w")).mean()), 2),
                           "by_lang_exact": (100 * (P.o_join == P.c_join).groupby(P.lang).mean()).round(2).to_dict(),
                           "by_lang_word_boundary": (100 * ((P.o_join == "w") == (P.c_join == "w")).groupby(P.lang).mean()).round(2).to_dict(),
                           "disagreements_oracc|cdli": {f"{l} {a}|{b}": int(n) for (l, a, b), n in jdis.groupby(["lang", "o_join", "c_join"]).size().sort_values(ascending=False).head(24).items()}}
ex = {}
for (l, a, b), g in jdis.groupby(["lang", "o_join", "c_join"]):
    if len(g) >= 150:
        ex[f"{l} {a}|{b}"] = [f"{s}:{r} {n}" for (s, r), n in g.groupby(["sign", "o_read"]).size().sort_values(ascending=False).head(10).items()]
res["both_source_join"]["examples (sign:Oracc reading before the joiner)"] = ex
o += [f"\n\n### both-source positions ({len(P)}): joiner agreement {res['both_source_join']['exact_agreement_pct']}%, word-boundary agreement "
      f"{res['both_source_join']['word_boundary_agreement_pct']}%\n", "rows Oracc, columns CDLI\n", jc.to_string(), "\n\nlargest disagreement cells, signs before the joiner:\n"]
o += [f"- {k}: " + ", ".join(v) for k, v in ex.items()]
# determinatives
P["od"] = P.o_det.str.split(":").str[0].replace("", "none")
P["cd"] = P.c_det.str.split(":").str[0].replace("", "none")
P["ok"] = P.o_det.str.split(":").str[-1]
P["ck"] = P.c_det.str.split(":").str[-1]
dc = pd.crosstab([P.lang, P.od], P.cd)
ddis = P[(P.od == "none") != (P.cd == "none")]
d1 = ddis[ddis.od != "none"].groupby(["lang", "sign", "o_read", "ok"]).size().sort_values(ascending=False).head(25)
d2 = ddis[ddis.cd != "none"].groupby(["lang", "sign", "c_read", "ck"]).size().sort_values(ascending=False).head(25)
phon = pd.crosstab(P.ok.replace("", "none"), P.ck.replace("", "none"))
res["both_source_det"] = {"flag_agreement_pct": round(100 * float(((P.od == "none") == (P.cd == "none")).mean()), 3),
                          "position_agreement_pct": round(100 * float((P.od == P.cd).mean()), 3),
                          "det_only_in_oracc": int((ddis.od != "none").sum()), "det_only_in_cdli": int((ddis.cd != "none").sum()),
                          "det_only_in_oracc_top": {f"{l} {s}:{r} ({k})": int(n) for (l, s, r, k), n in d1.items()},
                          "det_only_in_cdli_top": {f"{l} {s}:{r} ({k})": int(n) for (l, s, r, k), n in d2.items()},
                          "kind_oracc_x_cdli": phon.to_dict()}
o += [f"\n\n### determinatives on both-source positions: flag agreement {res['both_source_det']['flag_agreement_pct']}%, position (pre/post) agreement "
      f"{res['both_source_det']['position_agreement_pct']}%\n", dc.to_string(), "\n\nkind (rows Oracc, columns CDLI; 'phonetic' = phonetic complement / gloss)\n", phon.to_string(),
      "\n\ndeterminative only in Oracc: " + ", ".join(f"{k} {v}" for k, v in res["both_source_det"]["det_only_in_oracc_top"].items()),
      "\n\ndeterminative only in CDLI: " + ", ".join(f"{k} {v}" for k, v in res["both_source_det"]["det_only_in_cdli_top"].items())]
# phonetic complements in train: det readings that are syllables after a logogram, per source (post-determinatives that are not ki / meš etc.)

# ---------------- Q4 sign plane
BL = pickle.load(gzip.open(os.path.join(OUT, "pairs_blocks.pkl.gz"), "rb"))
strip = lambda n: re.sub(r"[@~][a-z0-9]+", "", n)  # noqa: E731
srds, comp, variant, other11, sizes = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
PROJ_CDLI = None
for k, n in BL.items():
    if k[0] == "big":
        sizes["big"] += n
        continue
    lg, ob, cb = k
    ob = tuple(x for x in ob)
    cb = tuple(x for x in cb)
    sizes[(min(len(ob), 3), min(len(cb), 3))] += n
    if len(ob) == 1 and len(cb) == 1:
        (on, orr, _), (cn, cr, _) = ob[0], cb[0]
        if not on or not cn or orr.startswith("<") or cr.startswith("<"):
            continue
        if strip(on) == strip(cn):
            variant[(lg, on, cn, orr, cr)] += n
        elif orr == cr:
            srds[(lg, orr, on, cn)] += n
        else:
            other11[(lg, on, orr, cn, cr)] += n
    elif (len(ob) == 1) != (len(cb) == 1) and ob and cb:
        one, many, side = (ob[0], cb, "Oracc compound | CDLI sequence") if len(ob) == 1 else (cb[0], ob, "CDLI compound | Oracc sequence")
        nm = one[0]
        if nm.startswith("|") and all(m[0] for m in many):
            parts = [strip(x) for x in re.split(r"[.×+&%]", nm.strip("|").replace("(", "").replace(")", ""))]
            if sorted(parts) == sorted(strip(m[0]).strip("|") for m in many) or "".join(parts) == "".join(strip(m[0]).strip("|") for m in many):
                comp[(side, lg, nm, one[1], " ".join(m[1] + (m[2] if m[2] != "w" else " ") for m in many).strip())] += n
n_blocks = sum(BL.values())
res["sign_plane"] = {"non_matching_blocks": int(n_blocks),
                     "same_reading_different_sign": {"positions": int(sum(srds.values())), "top (lang, reading, Oracc sign, CDLI sign)": [[*k, v] for k, v in srds.most_common(40)]},
                     "variant_form_split (same base sign, other @/~ form)": {"positions": int(sum(variant.values())), "top (lang, Oracc sign, CDLI sign, readings)": [[*k, v] for k, v in variant.most_common(40)]},
                     "compound_vs_sequence": {"positions": int(sum(comp.values())), "by_side": {s: int(sum(v for k, v in comp.items() if k[0] == s)) for s in {k[0] for k in comp}},
                                              "top": [[*k, v] for k, v in comp.most_common(40)]},
                     "other_one_to_one_substitutions": {"positions": int(sum(other11.values())), "top": [[*k, v] for k, v in other11.most_common(30)]}}
o += [f"\n\n## Q4 sign plane (both-source, Oracc curated vs CDLI; {n_blocks} non-matching alignment blocks)\n",
      f"same reading, different sign id: {sum(srds.values())} positions: " + ", ".join(f"{k[1]}: {k[2]} | {k[3]} {v}" for k, v in srds.most_common(30)),
      f"\n\nvariant form of the same base sign (@g @t ~a ...) in one source only: {sum(variant.values())} positions: " + ", ".join(f"{k[1]} | {k[2]} ({k[3]}) {v}" for k, v in variant.most_common(30)),
      f"\n\ncompound in one source, sequence of its parts in the other: {sum(comp.values())} positions ({res['sign_plane']['compound_vs_sequence']['by_side']}): "
      + ", ".join(f"[{k[0][:5]}] {k[2]} {k[3]} | {k[4]} {v}" for k, v in comp.most_common(30)),
      f"\n\nother 1:1 substitutions (different sign and reading: edition differences): {sum(other11.values())}: " + ", ".join(f"{k[1]}:{k[2]} | {k[3]}:{k[4]} {v}" for k, v in other11.most_common(20))]
# numerals in train: sign ids per numeral reading and source
isnum = np.array([bool(NUM.match(r)) for r in rv])
m = isnum[R] & real
nd = pd.DataFrame({"r": R[m], "s": S[m], "src": g_src[m]})
t = nd.groupby(["r", "s", "src"]).size().rename("n").reset_index()
num_rows, n_split_tok = [], 0
for r, g in t.groupby("r"):
    tot = g.n.sum()
    by_s = g.groupby("s").n.sum().sort_values(ascending=False)
    if len(by_s) > 1 and by_s.iloc[1] >= 20:
        n_split_tok += int(tot - by_s.iloc[0])
        num_rows.append({"reading": rv[r], "tokens": int(tot), "signs": {sn[s]: {a: int(b) for a, b in g[g.s == s].groupby("src").n.sum().items()} for s in by_s.index[:4]}})
num_rows.sort(key=lambda d: -d["tokens"])
multi = np.array([" " in r for r in rv])[R] & real
nm_src = pd.Series(g_src[isnum[R] & real]).value_counts()
numword = collections.Counter()
res["numerals"] = {"numeral_reading_tokens": int(m.sum()), "share_of_reading_tokens_pct": round(100 * float(m.sum()) / int(((R > 1) & real).sum()), 2),
                   "numeral_share_by_source_pct": (100 * pd.Series(isnum[R] & real).groupby(g_src).mean()).round(2).to_dict(),
                   "readings_with_several_signs": num_rows[:40], "tokens_not_on_the_main_sign": n_split_tok,
                   "compound_reading_on_one_sign (Oracc only)": {"tokens": int(multi.sum()), "by_source": pd.Series(g_src[multi]).value_counts().to_dict(),
                                                                 "top": [[rv[a], int(b)] for a, b in pd.Series(R[multi]).value_counts().head(12).items()]}}
o += [f"\n\n### numerals (train): {m.sum()} numeral reading tokens; compound numbers kept as ONE sign with a multi-part reading (Oracc only): {multi.sum()} tokens",
      f"\nnumeral readings that still sit on several sign ids (second sign >= 20 tokens): {len(num_rows)} readings, {n_split_tok} tokens off the main sign\n"]
o += [f"- {d['reading']} ({d['tokens']}): {json.dumps(d['signs'], ensure_ascii=False)}" for d in num_rows[:25]]

# ---------------- Q5 damage
bt = pct(F, ["src", "lang"], "brk")
F["kind"] = np.where(S == 4, "x", np.where(S == 5, "gap", np.where(S == 3, "unk-sign", "sign")))
kt = pct(F, ["src", "lang"], "kind")
bp = pct(F[F.kind == "sign"], ["src", "pb"], "brk")
bp = bp[bp.n >= 50000]
# word-level uniformity: among words of >= 2 identified signs with any flag, are all signs flagged alike?
W = pd.DataFrame({"w": wid[real], "b": B[real], "src": g_src[real]})
wg = W.groupby("w").agg(n=("b", "size"), mn=("b", "min"), mx=("b", "max"), src=("src", "first"))
wg = wg[(wg.n >= 2) & (wg.mx > 0)]
wg["uniform"] = wg.mn == wg.mx
uni = wg.groupby("src").uniform.agg(["mean", "size"])
# restored (missing with a reading) vs lost
Pd = P.copy()
bc = pd.crosstab([Pd.lang, Pd.o_brk.replace("", "intact")], Pd.c_brk.replace("", "intact"))
bc_all = pd.crosstab(Pd.o_brk.replace("", "intact"), Pd.c_brk.replace("", "intact"))
agree = float((Pd.o_brk == Pd.c_brk).mean())
# does one source flag a whole word where the other flags single signs? words (Oracc segmentation) with mixed flags, each side
Pd["wkey"] = (Pd.o_join == "w").shift(1, fill_value=True).cumsum()
ww = Pd.groupby("wkey").agg(n=("o_brk", "size"), o_n=("o_brk", lambda x: x.nunique()), c_n=("c_brk", lambda x: x.nunique()),
                            o_any=("o_brk", lambda x: (x != "").any()), c_any=("c_brk", lambda x: (x != "").any()))
ww = ww[(ww.n >= 2) & (ww.o_any | ww.c_any)]
res["damage"] = {"flag_share_by_source_lang": bt.reset_index().to_dict("records"), "token_kind_by_source_lang": kt.reset_index().to_dict("records"),
                 "flag_share_by_source_period_bin": bp.reset_index().to_dict("records"),
                 "words_with_a_flag_all_signs_same_flag": {k: {"share": round(float(v["mean"]), 3), "words": int(v["size"])} for k, v in uni.iterrows()},
                 "both_source": {"agreement_pct": round(100 * agree, 2), "oracc_x_cdli": bc_all.to_dict(),
                                 "flagged_words_mixed_in_oracc_pct": round(100 * float((ww.o_n > 1).mean()), 2),
                                 "flagged_words_mixed_in_cdli_pct": round(100 * float((ww.c_n > 1).mean()), 2), "flagged_words": int(len(ww))}}
o += ["\n\n## Q5 damage flags (train), % of all positions by source and language\n", bt.to_string(), "\n\nkind of position (identified sign / x / gap / unknown sign)\n", kt.to_string(),
      "\n\nidentified signs only, by source and period bin (>= 50k signs)\n", bp.to_string(),
      "\n\nwords (>= 2 identified signs) carrying a flag: share whose signs all carry the same flag\n", uni.to_string(),
      f"\n\n### both-source positions: damage flag agreement {100 * agree:.2f}% (rows Oracc, columns CDLI)\n", bc_all.to_string(), "\n", bc.to_string(),
      f"\n\nflagged words (Oracc word segmentation, >= 2 matched signs; {len(ww)}): mixed flags inside the word in Oracc {100 * (ww.o_n > 1).mean():.1f}%, in CDLI {100 * (ww.c_n > 1).mean():.1f}%"]
json.dump(res, open(os.path.join(OUT, "q345_tables.json"), "w"), ensure_ascii=False, indent=1, default=str)
open(os.path.join(OUT, "q345_report.md"), "w").write("\n".join(o) + "\n")
print("\n".join(o))
