"""Shared loading for the error diagnosis (DEV only): instance table, slot table, helpers."""
from __future__ import annotations

import gzip
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN, GAP, UNK_SIGN, X_SIGN  # noqa: E402

OUT = os.path.join(ROOT, "results/diag/errors")
SPLIT = "dev"                                    # never anything else
SYL = re.compile(r"^[^aeiu]?[aeiu][^aeiu]?[₀-₉ₓ]*$")
NUM = re.compile(r"^(\d|n\(|\d+/\d+)")


def pk(path):
    return pickle.load(gzip.open(path, "rb"))


def period_norm(p):
    if p is None:
        return "(none)"
    p = re.sub(r"\s*\(.*\)", "", p).strip().rstrip("?").strip()
    p = p.replace("Lagaš II", "Lagash II")
    if p.startswith("ED ") or p == "Early Dynastic":
        return "Early Dynastic"
    return p


def genre_norm(g):
    if g is None:
        return "(none)"
    g = g.strip().rstrip("?").strip().lower()
    if "letter" in g:
        return "letter"
    if g.startswith("admin"):
        return "administrative"
    if g.startswith("legal"):
        return "legal"
    if g.startswith("royal"):
        return "royal"
    if g.startswith("lexical") or g == "school":
        return "lexical/school"
    if g.startswith("liter"):
        return "literary"
    if any(k in g for k in ("astro", "diary", "goal-year", "scientific", "mathemat", "omen", "extispicy", "divination",
                            "medical", "scholar", "almanac", "commentary")):
        return "scholarly/science"
    if any(k in g for k in ("ritual", "incantation", "prayer", "hymn", "lament", "liturg", "religious")):
        return "religious/ritual"
    return "other"


def src_norm(s):
    if s == "cdli":
        return "cdli"
    if s == "oracc:cdli":
        return "oracc:cdli"
    return "oracc"


def load():
    ev = evalset.load(ROOT, SPLIT)
    cache = pk(os.path.join(OUT, "cache.pkl.gz"))
    assert cache["version"] == ev["version"]
    P = {}
    b = pk(os.path.join(ROOT, f"results/gaps/base-long__{SPLIT}.pkl.gz"))
    for k in ("known/indep", "known/iter", "unknown/indep", "unknown/iter", "translit"):
        P[k] = b[k]
    P["unknown/iter-w"] = pk(os.path.join(ROOT, f"results/gaps/base-long-w__{SPLIT}.pkl.gz"))["unknown/iter-w"]
    n = pk(os.path.join(ROOT, f"results/gaps/ngram__{SPLIT}.pkl.gz"))
    P["known/4gram"] = n["known/4gram"]
    P["translit_ngram"] = n["translit"]
    P["polyvalent"] = set(n["polyvalent_signs"])
    return ev, cache, P


def chunk_offsets(ev):
    """Per chunk: (offset of the chunk in its text, number of signs of the text, n chunks, index of the chunk)."""
    tot, cnt = {}, {}
    for ti, lg, a in ev["chunks"]:
        tot[ti] = tot.get(ti, 0) + a.shape[1]
        cnt[ti] = cnt.get(ti, 0) + 1
    seen, k = {}, {}
    out = []
    for ti, lg, a in ev["chunks"]:
        out.append((seen.get(ti, 0), tot[ti], cnt[ti], k.get(ti, 0)))
        seen[ti] = seen.get(ti, 0) + a.shape[1]
        k[ti] = k.get(ti, 0) + 1
    return out


def sign_type(ev, lang, sign, read, join_prev, join, det):
    """Proxy classes of one gold sign occurrence.
    numeral: the reading is a number notation (1(diš), 1/2(diš), n(...)), or the sign has no reading and its
             name is one; det: marked determinative; for Akkadian: 'logo' = joined with '.' to a neighbour
             (logogram compound) or a reading that is not shaped (C)V(C); 'logo?/1-sign word' = a (C)V(C)
             reading standing alone as a word; 'syll' = (C)V(C) reading inside a multi-sign word.
    For other languages: 'word-alone' / 'in-word'."""
    r = ev["read_vocab"][read] if read > 1 else ev["sign_names"][sign].lower()
    if NUM.match(r):
        return "numeral"
    if det in (1, 2):
        return "determinative"
    alone = join_prev in (0, 6, -1) and join in (0, 6)
    if lang == "akk":
        if read <= 1:
            return "akk:no-reading"
        if join == 2 or join_prev == 2 or not SYL.match(r):
            return "akk:logogram"
        return "akk:(C)V(C) alone (ambig.)" if alone else "akk:syllabic"
    if lang == "sux":
        return "sux:1-sign word" if alone else "sux:in multi-sign word"
    return "other-lang"


def tables(ev, cache, P):
    inst = ev["instances"]
    offs = chunk_offsets(ev)
    langs = ev["langs"]
    rows, slots = [], []
    XID = ev["sign_names"].index("X")
    for i, x in enumerate(inst):
        ti, lg, a = ev["chunks"][x["chunk"]]
        s, L = x["start"], x["L"]
        n = a.shape[1]
        m = cache["dev_meta"][ti]
        off, tn, nch, ci = offs[x["chunk"]]
        gs, gr, gj, gd = evalset.gold(ev, x)
        # Oracc's illegible 'x' is the real sign id 'X' (CDLI's is the special <x>): both count as unknown here
        dmg = np.isin(a[0], (UNK_SIGN, X_SIGN, GAP, XID)) | (a[4] > 0)
        unk = np.isin(a[0], (UNK_SIGN, X_SIGN, GAP, XID))
        ctx = np.r_[np.arange(max(0, s - 10), s), np.arange(s + L, min(n, s + L + 10))]
        ctx3 = np.r_[np.arange(max(0, s - 3), s), np.arange(s + L, min(n, s + L + 3))]
        rest = np.r_[np.arange(0, s), np.arange(s + L, n)]
        jp = int(a[2, s - 1]) if s > 0 else -1
        r = {"i": i, "L": L, "novel": x["novel"], "extra": x["extra"], "text": ti, "lang": langs[lg],
             "period": period_norm(m["period"]), "genre": genre_norm(m["genre"]), "src": src_norm(m["src"]),
             "src_raw": m["src"], "text_n": tn, "n_chunks": nch, "chunk_i": ci, "chunk_n": n, "start": s,
             "d_start": s, "d_end": n - s - L, "cut_left": ci > 0, "cut_right": ci < nch - 1,
             "t_start": off + s, "t_end": tn - off - s - L,
             "dmg10": int(dmg[ctx].sum()), "unk10": int(unk[ctx].sum()), "dmg3": int(dmg[ctx3].sum()),
             "unk3": int(unk[ctx3].sum()), "dmg_chunk": float(dmg[rest].mean()), "unk_chunk": float(unk[rest].mean()),
             "brk_chunk": float((a[4][rest] > 0).mean()),
             "x_line": bool((gj[:-1] == 6).any()), "x_word": bool(np.isin(gj[:-1], (0, 6)).any()),
             "starts_word": jp in (0, 6, -1), "ends_word": int(gj[-1]) in (0, 6),
             "starts_line": jp in (6, -1), "ends_line": int(gj[-1]) == 6}
        for key in ("known/iter", "known/indep", "known/4gram"):
            p = P[key][i]
            ps = np.asarray(p["sign"], np.int64)
            tag = key.split("/")[1]
            r[f"ser_{tag}"] = Levenshtein.distance(ps.tolist(), gs.tolist()) / L
            r[f"ham_{tag}"] = float((ps != gs).mean())
            r[f"exact_{tag}"] = float((ps == gs).all())
            g_str = evalset.render(ev, gs, gr, gj, gd)
            r[f"cer_{tag}"] = Levenshtein.distance(evalset.render(ev, p["sign"], p["read"], p["join"], p["det"]),
                                                   g_str) / max(1, len(g_str))
        if L == 1:
            tk = list(P["known/indep"][i]["topk"])
            r["top1"], r["top10"] = float(int(gs[0]) in tk[:1]), float(int(gs[0]) in tk[:10])
            tk = list(P["known/4gram"][i]["topk"])
            r["top1_4g"], r["top10_4g"] = float(int(gs[0]) in tk[:1]), float(int(gs[0]) in tk[:10])
        rows.append(r)
        pi, pn, p4 = P["known/iter"][i], P["known/indep"][i], P["known/4gram"][i]
        for k in range(L):
            slots.append({"i": i, "k": k, "L": L, "lang": langs[lg], "novel": x["novel"], "g": int(gs[k]),
                          "gr": int(gr[k]), "gj": int(gj[k]), "gd": int(gd[k]),
                          "p": int(pi["sign"][k]), "pr": int(pi["read"][k]), "pj": int(pi["join"][k]),
                          "pd": int(pi["det"][k]), "pn": int(pn["sign"][k]), "p4": int(p4["sign"][k]),
                          "type": sign_type(ev, langs[lg], int(gs[k]), int(gr[k]),
                                            int(gj[k - 1]) if k else jp, int(gj[k]), int(gd[k]))})
    I = pd.DataFrame(rows)
    S = pd.DataFrame(slots)
    S["ok"] = S.g == S.p
    S["ok_n"] = S.g == S.pn
    S["ok_4"] = S.g == S.p4
    lgi = {l: j for j, l in enumerate(cache["langs"])}
    S["f_all"] = cache["f_sign"].sum(0)[S.g.values]
    S["f_lang"] = cache["f_sign"][S.lang.map(lgi).values, S.g.values]
    return I, S


def cell(df, col, n_col=None):
    """mean in % with n"""
    return f"{100 * df[col].mean():.1f}"


def tab(df, by, cols, min_n=1, sort=None, names=None, extra=None):
    """Markdown table: group means (in %) of cols with n."""
    g = df.groupby(by, observed=True, dropna=False)
    t = g[cols].mean() * 100
    t.insert(0, "n", g.size())
    if extra:
        for k, f in extra.items():
            t[k] = g.apply(f)
    t = t[t.n >= min_n]
    if sort == "n":
        t = t.sort_values("n", ascending=False)
    elif sort:
        t = t.sort_values(sort)
    if names:
        t = t.rename(columns=names)
    return md(t)


def flag(n):
    return "" if n >= 100 else " (n<100)"


def md(t, index=True, nd=1):
    """Markdown table of a DataFrame (no tabulate here)."""
    t = t.reset_index() if index else t

    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() and abs(v) >= 1000 else f"{v:.{nd}f}")
        return str(v)
    L = ["| " + " | ".join(str(c) for c in t.columns) + " |", "|" + "---|" * len(t.columns)]
    for row in t.itertuples(index=False):
        L.append("| " + " | ".join(f(v) for v in row) + " |")
    return "\n".join(L)


_SUB = str.maketrans("", "", "₀₁₂₃₄₅₆₇₈₉ₓ")
_VOI = str.maketrans({"b": "p", "d": "t", "ṭ": "t", "g": "k", "q": "k", "z": "s", "ṣ": "s", "e": "i"})


def read_diff(g, p):
    """Class of a reading error (same sign, gold reading g, predicted reading p; strings).
    'index'     same letters, other index (bi / bi₂ style: same value, other numbering)
    'auslaut'   one is the other plus a final consonant or vowel (du₁₁ / dug₄, gu₄ / gud, tu₄ / tum, zi / zid):
                long vs short form of the same value
    'numeral'   both number notations (1(diš) / 1(aš), 1(u) / 10 ...)
    'voicing'   equal after merging b/p, d/t/ṭ, g/k/q, z/s/ṣ and e/i (bu / pu, di / ṭi, be / bi): the same syllable
                family; the choice is lexical (real, but an Akkadian editor decides it from the word)
    'different' anything else (a different value of the sign: logogram vs syllable, other word)"""
    if NUM.match(g) and NUM.match(p):
        return "numeral"
    if NUM.match(g) or NUM.match(p):
        return "different"
    a, b = g.translate(_SUB), p.translate(_SUB)
    if a == b:
        return "index"
    s, l = sorted((a, b), key=len)
    if l.startswith(s) and len(l) - len(s) == 1 and len(s) >= 2:
        return "auslaut"
    if a.translate(_VOI) == b.translate(_VOI):
        return "voicing"
    return "different"
