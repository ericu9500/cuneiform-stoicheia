"""Noise audit, step 3 (Q1 + Q6): reading-convention classes, their size, and prototype normalisation tables.

Tiers of a reading pair (a, b) of one sign in texts of one language:
  T1 safe          the same value under two conventional names; attested as an edition difference on >= 3
                   both-source positions; the two names (almost) never stand in the same text (cooc <= 0.25);
                   neither is CDLI's minority form with >= 20% of CDLI's tokens of the pair;
                   in Akkadian texts the replaced name must be logographic (>= 90% of its both-source tokens are
                   logograms or determinatives). -> training targets and scoring.
  T2 scoring-only  the same formal relation and both-source evidence, but the two names contrast inside texts
                   (one of them is also a syllabic / more general value): sense-level credit only.
                   T2a cooc <= 0.5 or syllabic use in Akkadian; T2b cooc > 0.5: one form is the sign's general
                   value (mu for gu₁₀, zi for zid): lenient credit; T2c Akkadian logogram written by its sign name
                   in Oracc and by a value in CDLI (GI / sig₁₇, AG / na₃; >= 20 both-source positions, >= 90% of
                   them capitals in Oracc): lenient credit.
  T3 keep          everything else (different values, Akkadian syllabic voicing and e/i, mimation CVm / CV, ...).
Eligible relations: same letters with another index, the ŋ / geminate convention, long vs short form
(read_class: index, eng, auslaut), e / i vowel colour in word values (>= 3 letters), and a hand list of
synonymous value names (uru / iri, erim / erin₂ ...) and consonant variants (gurušda / kurušda ...).
Determinatives: m / diš / 1(diš) on DIŠ, d / dingir on AN, f / munus / mi₂ / sal on SAL in determinative position.

Writes results/noise_audit/: rules_reading_safe.json, rules_reading_scoring_only.json, rules_determinatives.json,
rules_numerals.json, rules_do_not_normalise.json, q1_tables.json, q1_report.md, samples_precision.tsv.
Reads train (cache), both-source pairs, candidates, and the DEV evalset + dev predictions (never test).
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
from noise_audit_lib import FIRST_REAL_SIGN, NUM, OUT, ROOT, letters, load_train, read_class  # noqa: E402

SEED = 20261120
COOC_MAX, PAIRS_MIN, LOGO_MIN = 0.25, 3, 0.9
SYN = [  # synonymous names of one value (hand list, each checked against the both-source pairs)
    "uru/iri", "erim/erin₂", "gim/gin₇", "sum/šum₂", "uruda/urudu", "urud/urudu", "ud₅/uzud", "ud₅/uz₃", "uz₃/uzud",
    "sag₁₀/sig₅", "nigir/nimgir", "gidri/gidru", "gešnimbar/nimbar", "gešimmar/gešnimbar", "gešimmar/nimbar",
    "gišimmar/gešnimbar", "šagina/šakkan₆", "taskarin/taškarin", "ses/šeš", "duda/tud₂", "sumun₂/sun₂", "kešda/keš₂",
    "banda₃/ban₃", "sig₂/siki", "ašnan/ezinam₂", "sig₄/šeg₁₂", "bugin/bunig", "buginₓ/bunigₓ", "tumu/tu₁₅", "iti/itud",
    "iti/itu", "itu/itud", "inda₃/ninda", "nigin₂/ni₁₀", "erim₂/|ne.ru|", "kimin/|ki.min|", "|ki.lam|/šakanka",
    "geškim/giskim", "giškim/giskim", "numdun/nundun", "luhša/luhšu", "umum/umun₂", "muh/ugu", "lak175/saga₂",
    "se₃/sig₁₀", "ge₆/gi₆", "ge₄/gi₄", "gurušda/kurušda", "dub₂/tub₂", "dah/tah", "dan₃/tan₃", "sizkur/siskur",
    "sizkur₂/siskur₂", "gis/geš", "gis/giš", "šidim/šitim", "libiš/lipiš"]
SYN = {frozenset(x.split("/")) for x in SYN}
BLACK_AKK = {frozenset(("gin", "gen"))}      # GIN (kânu) and GEN (alāku) are different logograms of DU in Akkadian
BLACK = {frozenset(x.split("/")) for x in ["gi₆/mi", "ge₆/mi", "ge₆/me₂", "gi₆/me₂", "ne/ge₉", "giš/niš₂", "gis/nis₂"]}   # formal look-alikes, different values
SYN_SUX = {frozenset(x.split("/")) for x in ["ge/gi", "ge₄/gi₄", "ge₆/gi₆"]}      # e / i colour of short Sumerian values
DET = {"DIŠ": ("m", {"m", "diš", "1(diš)"}), "AN": ("d", {"d", "dingir", "digir"}), "SAL": ("f", {"f", "munus", "mi₂", "sal"})}
MOD = re.compile(r"@[a-z0-9]+$")


def e_i(a, b):
    x, y = letters(a), letters(b)
    return len(x) == len(y) >= 3 and x != y and x.replace("e", "i") == y.replace("e", "i")


def mimation(a, b):
    s, l = sorted((letters(a), letters(b)), key=len)
    return l == s + "m" and len(s) == 2


def relation(a, b, lang="sux"):
    """-> eligible relation name or None"""
    if (lang == "akk" and frozenset((a, b)) in BLACK_AKK) or frozenset((a, b)) in BLACK or \
            (lang == "sux" and frozenset((a, b)) == frozenset(("sum", "šum₂"))):      # sum 'garlic' vs šum₂ 'to give' (precision sample)
        return None
    if frozenset((a, b)) in SYN or (lang == "sux" and frozenset((a, b)) in SYN_SUX):
        return "synonym"
    if MOD.sub("", a) == MOD.sub("", b):
        return "modifier"
    c = read_class(a, b)
    if c == "auslaut" and mimation(a, b):
        return None
    if c in ("index", "eng", "auslaut"):
        return c
    if e_i(a, b):
        return "e/i"
    return None


def main():
    rng = np.random.default_rng(SEED)
    A, T, LS, M = load_train()
    rv, sv, sn = M["read_vocab"], M["sign_vocab"], M["sign_names"]
    rid = {r: i for i, r in enumerate(rv)}
    sidn = {n: i for i, n in enumerate(sn)}
    src3 = np.where(M["t_proj"] == "cdli", "cdli", np.where(M["t_proj"] == "oracc:cdli", "oracc:cdli", "oracc"))
    lang2 = np.where(np.isin(M["t_lang"], ["sux", "akk"]), M["t_lang"], "oth")
    ok = (A[0] >= FIRST_REAL_SIGN) & (A[1] > 1)
    NTOK = int(ok.sum())
    C = pd.read_pickle(os.path.join(OUT, "candidates.pkl.gz"))
    P = pd.read_pickle(os.path.join(OUT, "pairs_matched.pkl.gz"))
    PT = pickle.load(gzip.open(os.path.join(OUT, "pairs_texts.pkl.gz"), "rb"))
    names, PTX = PT["names"], PT["texts"]
    for c in ("oid", "o_read", "c_read", "tid", "o_det", "c_det"):
        P[c] = P[c].astype(str)
    P = P[~P.oid.str.startswith("<")].copy()
    P["proj"] = P.tid.map({k: v["proj"] for k, v in PTX.items()})
    P["lang"] = P.tid.map({k: v["lang"] for k, v in PTX.items()})
    P.loc[~P.lang.isin(["sux", "akk"]), "lang"] = "oth"
    P["period"] = P.tid.map({k: v["period"] for k, v in PTX.items()})
    P_all = P
    P = P[P.proj != "cdli"].copy()
    P["sign"] = P.oid.map(names)
    # logographic share per (lang, oid, reading): Oracc capitals / CDLI _.._ / determinative, both sides pooled
    # (Oracc side, Akkadian words only: there capitals mark logograms reliably; CDLI's _.._ is not used by every editor)
    Pa = P[P.wlang.astype(str) == "akk"]
    lo = pd.DataFrame({"lang": Pa.lang, "oid": Pa.oid, "r": Pa.o_read, "logo": Pa.o_logo | (Pa.o_det != "")})
    lg = lo.groupby(["lang", "oid", "r"]).logo.agg(["mean", "size"])
    # T2c: Akkadian logogram named by its sign in Oracc and by a value in CDLI
    pl = P[(P.lang == "akk") & (P.o_read != P.c_read)].groupby(["oid", "o_read", "c_read"]).o_logo.agg(["mean", "size"])
    signname = {(o, *sorted((a, b))) for (o, a, b), v in pl.iterrows() if v["size"] >= 20 and v["mean"] >= 0.9}
    logo_share = {k: (v["mean"], v["size"]) for k, v in lg.iterrows()}

    # ---------------- tiers
    pooled = C.groupby(["s", "a", "b"]).pairs.sum().to_dict()
    cnt_all = collections.Counter()          # pooled train count of (sign, reading)
    for r in C.itertuples():
        cnt_all[(r.s, r.a)] = max(cnt_all[(r.s, r.a)], 0)
    tot_sr = collections.Counter()
    key = A[0][ok].astype(np.int64) * len(rv) + A[1][ok]
    u, n = np.unique(key, return_counts=True)
    for k, v in zip(u, n):
        tot_sr[(int(k // len(rv)), rv[int(k % len(rv))])] = int(v)
    rows = []
    for r in C.itertuples():
        rel = relation(r.a, r.b, r.lang)
        pall = pooled[(r.s, r.a, r.b)]
        na, nb = tot_sr[(r.s, r.a)], tot_sr[(r.s, r.b)]
        canon, repl = (r.a, r.b) if na >= nb else (r.b, r.a)
        ls = logo_share.get((r.lang, sv[r.s], repl), (np.nan, 0))
        tier, why = "T3", ""
        if rel == "e/i" and r.lang == "akk" and not (ls[1] >= 20 and ls[0] >= LOGO_MIN):
            rel = None
        if rel is None:
            why = "mimation (CVm / CV)" if (r.cls == "auslaut" and mimation(r.a, r.b)) else "different value / lexical choice"
            if r.lang == "akk" and (sv[r.s], r.a, r.b) in signname and r.cls not in ("numeral", "num/word") and not mimation(r.a, r.b):
                tier, why = "T2c", "Akkadian logogram: sign name (Oracc) vs value (CDLI)"
        elif pall < PAIRS_MIN and rel not in ("modifier",) and not (rel == "index" and ("ₓ" in r.a or "ₓ" in r.b)):
            tier, why = "none", "no both-source evidence"
        elif r.n_a == 0 or r.n_b == 0:
            tier, why = ("T1", "one form absent in this language (no token changes)") if r.lang != "akk" else ("T2a", "one form absent")
        elif not (r.cooc <= COOC_MAX):
            tier, why = ("T2a" if r.cooc <= 0.5 else "T2b"), f"forms contrast inside texts (cooc {r.cooc:.2f})"
        elif r.a_cdli + r.b_cdli >= 100 and min(r.a_cdli, r.b_cdli) >= 0.2 * (r.a_cdli + r.b_cdli):
            tier, why = "T2a", "both forms are frequent inside CDLI's own texts (one convention uses them for different things)"
        elif r.lang == "akk" and ls[1] >= 20 and ls[0] < LOGO_MIN:
            tier, why = "T2a", f"replaced form is also syllabic in Akkadian (logographic share {ls[0]:.2f})"
        elif r.lang == "oth":
            tier, why = "T2a", "language other than sux / akk"
        else:
            tier = "T1"
        rows.append({"tier": tier, "why": why, "rel": rel or r.cls, "canon": canon, "repl": repl, "pairs_all": pall,
                     "logo_repl": ls[0], "logo_n": ls[1], "n_repl": r.n_a if repl == r.a else r.n_b,
                     "n_canon": r.n_b if repl == r.a else r.n_a})
    C = pd.concat([C.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    C.to_pickle(os.path.join(OUT, "candidates_tiered.pkl.gz"))
    print(C.groupby(["lang", "tier"]).agg(pairs=("a", "size"), tokens_repl=("n_repl", "sum")))

    # ---------------- groups (union-find) per (lang, sign): T1 -> canonical map; T1+T2 -> scoring classes
    def groups(df):
        par = {}

        def f(x):
            while par.setdefault(x, x) != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for r in df.itertuples():
            par[f((r.lang, r.s, r.a))] = f((r.lang, r.s, r.b))
        g = collections.defaultdict(list)
        for x in list(par):
            g[f(x)].append(x)
        return list(g.values())

    lang_tot = collections.Counter()         # (lang, sign, reading) -> train tokens
    lk = np.array(["sux", "akk", "oth"])
    li = np.searchsorted(np.array(["akk", "oth", "sux"]), lang2[T[ok]])
    key3 = (li.astype(np.int64) * len(sv) + A[0][ok]) * len(rv) + A[1][ok]
    u, n = np.unique(key3, return_counts=True)
    for k, v in zip(u, n):
        lang_tot[(["akk", "oth", "sux"][int(k // (len(sv) * len(rv)))], int(k // len(rv) % len(sv)), rv[int(k % len(rv))])] = int(v)
    safe_map = {}                            # (lang, sign idx, from) -> to
    safe_rules = []
    for g in groups(C[C.tier == "T1"]):
        lang, s = g[0][0], g[0][1]
        canon = max(g, key=lambda x: (tot_sr[(s, x[2])], x[2]))[2]
        for (_, _, r) in g:
            if r != canon:
                safe_map[(lang, s, r)] = canon
                ev = C[(C.lang == lang) & (C.s == s) & (C.tier == "T1") & ((C.a == r) | (C.b == r))].iloc[0]
                safe_rules.append({"lang": lang, "sign": sn[s], "oid": sv[s], "from": r, "to": canon, "relation": ev.rel,
                                   "train_tokens_changed": lang_tot[(lang, s, r)], "train_tokens_canonical": lang_tot[(lang, s, canon)],
                                   "both_source_pairs": int(ev.pairs_all), "cooc": None if pd.isna(ev.cooc) else round(float(ev.cooc), 3),
                                   "osl_lists_both": bool(ev.osl_a and ev.osl_b)})
    safe_rules.sort(key=lambda r: -r["train_tokens_changed"])
    strict_cls = {}                          # T1 + T2a classes
    for gi, g in enumerate(groups(C[C.tier.isin(["T1", "T2a"])])):
        for x in g:
            strict_cls[x] = gi
    score_cls = {}                           # (lang, sign idx, reading) -> class id (T1 + T2a + T2b + T2c)
    score_groups = []
    for gi, g in enumerate(groups(C[C.tier.isin(["T1", "T2a", "T2b", "T2c"])])):
        lang, s = g[0][0], g[0][1]
        for x in g:
            score_cls[x] = gi
        t2 = C[(C.lang == lang) & (C.s == s) & C.tier.isin(["T2a", "T2b", "T2c"]) & C.a.isin([x[2] for x in g])]
        score_groups.append({"lang": lang, "sign": sn[s], "oid": sv[s], "readings": sorted(x[2] for x in g),
                             "train_tokens": {x[2]: lang_tot[x] for x in g},
                             "scoring_only_pairs": [[r.a, r.b, r.tier, r.why] for r in t2.itertuples()]})
    score_groups = [g for g in score_groups if g["scoring_only_pairs"]]
    score_groups.sort(key=lambda g: -sorted(g["train_tokens"].values())[-2] if len(g["train_tokens"]) > 1 else 0)

    # ---------------- apply to train (counting only): per token class of change
    S, R, D3, TXI = A[0], A[1], A[3], T
    lang_tok = lang2[TXI]
    canon_r = R.copy()                       # reading id after det + T1 rules (ids; -1 = canonical form outside vocab)
    cls = np.zeros(R.shape, np.int8)         # 0 none, 1 det, 2 T1, 3 T2 (non-majority member of a scoring class), 4 compound numeral
    for nm, (cn, st) in DET.items():
        m = (S == sidn[nm]) & (D3 > 0) & np.isin(R, [rid[x] for x in st if x in rid and x != cn])
        cls[m] = 1
        canon_r[m] = rid[cn]
    for (lang, s, r), to in safe_map.items():
        if r not in rid:
            continue
        m = (S == s) & (R == rid[r]) & (lang_tok == lang) & (cls == 0)
        cls[m] = 2
        canon_r[m] = rid.get(to, -1)
    t2major = {}
    for g in score_groups:
        s = sidn[g["sign"]]
        mem = [safe_map.get((g["lang"], s, r), r) for r in g["readings"]]
        cnt = collections.Counter()
        for r, r0 in zip(mem, g["readings"]):
            cnt[r] += g["train_tokens"][r0]
        major = cnt.most_common(1)[0][0]
        for r0, r in zip(g["readings"], mem):
            if r != major and r0 in rid:
                m = (S == s) & (R == rid[r0]) & (lang_tok == g["lang"]) & (cls == 0)
                cls[m] = 3
    multi = np.array([" " in x for x in rv])
    cls[multi[R] & (cls == 0)] = 4
    per = M["t_period"][TXI]
    big = pd.Series(per[ok]).value_counts()
    per = np.where(np.isin(per, big.index[:12]), per, "other periods")
    tab = pd.DataFrame({"src": src3[TXI][ok], "lang": lang_tok[ok], "period": per[ok], "cls": cls[ok]})
    names_c = {0: "unchanged", 1: "det", 2: "T1_safe", 3: "T2_scoring_only", 4: "compound_numeral"}

    def shares(by):
        g = tab.groupby(by + ["cls"]).size().unstack(fill_value=0)
        g = g.rename(columns=names_c)
        n = g.sum(axis=1)
        out = (100 * g.div(n, axis=0)).round(3)
        out.insert(0, "tokens", n)
        return out.drop(columns="unchanged")
    sh = {"all": shares(["src"]).reset_index(), "src_lang": shares(["src", "lang"]).reset_index(),
          "src_lang_period": shares(["src", "lang", "period"]).reset_index()}
    sh["src_lang_period"] = sh["src_lang_period"][sh["src_lang_period"].tokens >= 30000]
    overall = (100 * tab.cls.value_counts(normalize=True)).rename(names_c).round(3).to_dict()
    # vocabulary effect of T1 + det
    used_before = np.unique(R[ok])
    used_after = np.unique(canon_r[ok])

    # ---------------- both-source classification (Oracc curated vs CDLI, sign-matched positions)
    P["diff"] = P.o_read != P.c_read
    sid = {o: i for i, o in enumerate(sv)}

    def classify(lang, oid, a, b, isdet, ologo, clogo, sign):
        s = sid.get(oid)
        if isdet and sign in DET and a in DET[sign][1] and b in DET[sign][1]:
            return "d determinative reading"
        na, nb = bool(NUM.match(a)), bool(NUM.match(b))
        if na and nb:
            return "b numeral: unsplit compound (Oracc one sign '1(u) 5(diš)')" if (" " in a or " " in b) else "b numeral: other notation / system"
        if na or nb:
            return "b numeral vs number word (min / 2(diš), 3(u) / sin)"
        if s is not None:
            ca, cb = safe_map.get((lang, s, a), a), safe_map.get((lang, s, b), b)
            if ca == cb:
                return "a value name: T1 safe"
            ga, gb = strict_cls.get((lang, s, a)), strict_cls.get((lang, s, b))
            if ga is not None and ga == gb:
                return "a value name: T2a scoring-only"
            ga, gb = score_cls.get((lang, s, a)), score_cls.get((lang, s, b))
            if ga is not None and ga == gb:
                return "a/c T2b+c lenient (general value or sign name vs specific value)"
        if relation(a, b, lang) is not None:
            return "a value name: formal relation, no rule (rare)"
        if ologo != clogo or (ologo and clogo):
            return "c logogram named differently / logogram vs syllabic"
        if read_class(a, b) == "auslaut" and mimation(a, b):
            return "e genuine: mimation (CVm / CV)"
        if read_class(a, b) == "voicing":
            return "e genuine: voicing or e/i of a syllabic value"
        return "e genuine: other value"
    d = P[P["diff"]].copy()
    d["isdet"] = (d.o_det != "") | (d.c_det != "")
    d["cls"] = [classify(*x) for x in zip(d.lang, d.oid, d.o_read, d.c_read, d.isdet, d.o_logo, d.c_logo, d.sign)]
    n_lang = P.groupby("lang").size()
    bs = d.groupby(["cls", "lang"]).size().unstack(fill_value=0)
    bs["all"] = bs.sum(axis=1)
    bs_pct = bs.copy().astype(float)
    for c in bs.columns:
        bs_pct[c] = (100 * bs[c] / (len(P) if c == "all" else n_lang[c])).round(3)
    top_c = {k: [[f"{s}: {a} | {b}", int(v)] for (s, a, b), v in g.groupby(["sign", "o_read", "c_read"]).size().sort_values(ascending=False).head(25).items()]
             for k, g in d.groupby("cls")}
    # the same on Oracc's own copy of CDLI (project cdli): converter-only differences
    pc = P_all[P_all.proj == "cdli"]
    cdli_copy = {"positions": int(len(pc)), "reading_differs_pct": round(100 * float((pc.o_read != pc.c_read).mean()), 3)}
    # near-identical editions only (reading ratio from matched share): texts with matched/o_signs >= 0.97
    near = {k for k, v in PTX.items() if v["o_signs"] and v["matched"] / v["o_signs"] >= 0.97 and v["proj"] != "cdli"}
    dn = d[d.tid.isin(near)]
    pn = P[P.tid.isin(near)]
    near_tab = (100 * dn.groupby("cls").size() / len(pn)).round(3).to_dict()

    # ---------------- DEV: errors of the model's bare-sign transliteration under the rules (dev only)
    # (model base-long belongs to corpus v2: its dev set = v3's devA + devB texts, v2 sign ids and vocabularies)
    ev = pickle.load(gzip.open(os.path.join(ROOT, "data/proc/evalset_dev.pkl.gz"), "rb"))
    pred = pickle.load(gzip.open(os.path.join(ROOT, "results/gaps/base-long__dev.pkl.gz"), "rb"))["translit"]
    erv, esn = ev["read_vocab"], ev["sign_names"]
    v2 = json.load(open(os.path.join(OUT, "v2/cache/meta.json")))          # sign ids of corpus v2 (vocabulary only)
    assert v2["version"] == ev["version"] and v2["sign_names"] == esn
    esv = v2["sign_vocab"]
    dmeta = pickle.load(gzip.open(os.path.join(ROOT, "results/diag/errors/cache.pkl.gz"), "rb"))["dev_meta"]
    dv_src, n_src = collections.Counter(), collections.Counter()
    dv = collections.Counter()
    dv_ex = collections.defaultdict(collections.Counter)
    n_pos = collections.Counter()
    for (ti, lgi, a), p in zip(ev["chunks"], pred):
        lang = ev["langs"][lgi]
        l2 = lang if lang in ("sux", "akk") else "oth"
        m = (a[1] > 1) & (a[0] >= FIRST_REAL_SIGN)
        pr = np.asarray(p["con"])
        n_pos[l2] += int(m.sum())
        sr = dmeta[ti]["src"]
        g3 = ("cdli" if sr == "cdli" else "oracc:cdli" if sr == "oracc:cdli" else "oracc") + " " + l2
        n_src[g3] += int(m.sum())
        for i in np.flatnonzero(m & (pr != a[1])):
            s, g, q = int(a[0][i]), erv[int(a[1][i])], erv[int(pr[i])]
            c = classify(l2, esv[s], g, q, a[3][i] > 0, False, False, esn[s])
            dv[(l2, c)] += 1
            dv_src[(g3, c[:1] if c[:1] != "a" else c[:16])] += 1
            dv_ex[c][f"{esn[s]}: {g} -> {q}"] += 1
    dev = pd.Series(dv).unstack(0).fillna(0).astype(int)
    dev["all"] = dev.sum(axis=1)
    npos = sum(n_pos.values())
    dev_pct = (100 * dev["all"] / npos).round(3)
    dsrc = pd.Series(dv_src).unstack(0).fillna(0)
    dsrc = (100 * dsrc / pd.Series(n_src)).round(3)
    dsrc.loc["TOTAL error %"] = dsrc.sum()
    dsrc.loc["positions"] = pd.Series(n_src)

    # ---------------- precision samples: 50 changed train tokens per rule family, with the line as context
    ls_of = np.searchsorted(LS, np.arange(0), side="right")
    joins = M["joins"]

    def context(i):
        k = int(np.searchsorted(LS, i, side="right") - 1)
        a, b = int(LS[k]), int(LS[k + 1])
        out = []
        for j in range(a, b):
            r = rv[R[j]] if R[j] > 1 else ("x" if S[j] == 4 else "…" if S[j] == 5 else "?")
            if D3[j]:
                r = "{" + r + "}"
            if j == i:
                r = "【" + r + "】"
            jn = joins[A[2][j]]
            out.append(r + (" " if jn == "w" else jn))
        return "".join(out).strip()
    samp = []
    for fam, mask in (("det", cls == 1), ("T1 sux", (cls == 2) & (lang_tok == "sux")), ("T1 akk", (cls == 2) & (lang_tok == "akk")),
                      ("T2 scoring-only", cls == 3)):
        idx = np.flatnonzero(mask & ok)
        for i in rng.choice(idx, min(50, len(idx)), replace=False):
            tx = M["texts"][TXI[i]]
            to = rv[canon_r[i]] if (cls[i] in (1, 2) and canon_r[i] >= 0) else "(class)"
            samp.append({"family": fam, "text": tx["id"], "src": tx["src"], "lang": tx["lang"], "period": tx["period"],
                         "sign": sn[S[i]], "from": rv[R[i]], "to": to, "line": context(i)})
    pd.DataFrame(samp).to_csv(os.path.join(OUT, "samples_precision.tsv"), sep="\t", index=False)

    # ---------------- write
    meta = {"corpus_version": M["version"], "split": "train only", "thresholds": {"cooc_max": COOC_MAX, "pairs_min": PAIRS_MIN, "logo_min": LOGO_MIN}}
    json.dump({**meta, "apply_to": "training targets and scoring", "key": "(text language, sign, reading) -> canonical reading",
               "rules": safe_rules}, open(os.path.join(OUT, "rules_reading_safe.json"), "w"), ensure_ascii=False, indent=0)
    json.dump({**meta, "apply_to": "scoring only (sense-level credit): a prediction inside the gold reading's class counts as correct",
               "note": "classes also contain the T1 pairs of the sign; listed are classes with at least one scoring-only pair",
               "classes": score_groups}, open(os.path.join(OUT, "rules_reading_scoring_only.json"), "w"), ensure_ascii=False, indent=0)
    det_counts = {}
    for nm, (cn, st) in DET.items():
        for x in st:
            if x in rid:
                m = (S == sidn[nm]) & (D3 > 0) & (R == rid[x])
                det_counts[f"{nm}:{x}"] = {k: int(v) for k, v in pd.Series(src3[TXI][m]).value_counts().items()}
    json.dump({**meta, "apply_to": "training targets and scoring", "condition": "det plane != 0 (pre or post) and sign as given",
               "rules": [{"sign": nm, "oid": sv[sidn[nm]], "from": sorted(st - {cn}), "to": cn} for nm, (cn, st) in DET.items()],
               "train_tokens_by_source": det_counts}, open(os.path.join(OUT, "rules_determinatives.json"), "w"), ensure_ascii=False, indent=1)
    isdetpair = np.array([r.sign in DET and r.a in DET[r.sign][1] and r.b in DET[r.sign][1] for r in C.itertuples()])
    t3 = C[(C.tier == "T3") & (C.pairs >= 5) & ~C.cls.isin(["numeral", "num/word"]) & ~isdetpair].sort_values("pairs", ascending=False)   # det pairs: rules_determinatives
    json.dump({**meta, "note": "reading pairs that differ between editions but must NOT be merged; pairs = both-source positions",
               "pairs": [{"lang": r.lang, "sign": r.sign, "a": r.a, "b": r.b, "reason": r.why, "class": r.cls, "pairs": int(r.pairs),
                          "train_a": int(r.n_a), "train_b": int(r.n_b), "cooc": None if pd.isna(r.cooc) else round(float(r.cooc), 3)}
                         for r in t3.itertuples()]}, open(os.path.join(OUT, "rules_do_not_normalise.json"), "w"), ensure_ascii=False, indent=0)
    numw = C[C.cls == "num/word"].sort_values("pairs", ascending=False)
    json.dump({**meta,
               "compound_numerals": {"rule": "builder (cun/oracc.py): an Oracc numeral grapheme WITH an oid whose reading (sexified form) has several "
                                             "space-separated parts ('1(u) 5(diš)') is one sign token today; split it into one sign per part, as C14(3) "
                                             "already does for numerals without an oid",
                                     "train_tokens": int((cls == 4).sum() + 0), "apply_to": "training targets and scoring (sign and reading planes)"},
               "number_words": {"apply_to": "scoring only", "pairs": [{"lang": r.lang, "sign": r.sign, "a": r.a, "b": r.b, "pairs": int(r.pairs),
                                                                         "train_a": int(r.n_a), "train_b": int(r.n_b), "cooc": None if pd.isna(r.cooc) else round(float(r.cooc), 3)}
                                                                        for r in numw.itertuples() if r.pairs >= 5]}},
              open(os.path.join(OUT, "rules_numerals.json"), "w"), ensure_ascii=False, indent=0)
    tables = {"train_reading_tokens": NTOK, "train_share_changed_pct": overall,
              "train_share_by_source": sh["all"].to_dict("records"), "train_share_by_source_lang": sh["src_lang"].to_dict("records"),
              "train_share_by_source_lang_period": sh["src_lang_period"].to_dict("records"),
              "read_vocab_used_in_train": int(len(used_before)), "after_T1_and_det": int(len(used_after)),
              "rules": {"T1_pairs": len(safe_rules), "T1_tokens": int((cls == 2).sum()), "det_tokens": int((cls == 1).sum()),
                        "T2_classes": len(score_groups), "T2_pairs_by_tier": C[C.tier.str.startswith("T2")].tier.value_counts().to_dict(), "T2_tokens_non_majority": int((cls == 3).sum()), "compound_numeral_tokens": int((cls == 4).sum())},
              "both_source": {"positions": int(len(P)), "by_lang_positions": n_lang.to_dict(), "differ_pct": round(100 * float(P["diff"].mean()), 3),
                              "differ_pct_by_lang": (100 * P.groupby("lang")["diff"].mean()).round(3).to_dict(),
                              "classes_n": bs.reset_index().to_dict("records"), "classes_pct_of_positions": bs_pct.reset_index().to_dict("records"),
                              "top_pairs": top_c, "oracc_copy_of_cdli": cdli_copy,
                              "near_identical_texts": {"texts": len(near), "positions": int(len(pn)), "differ_pct": round(100 * len(dn) / max(1, len(pn)), 3),
                                                       "classes_pct": near_tab}},
              "dev_translit": {"positions": npos, "errors": int(dev["all"].sum()), "error_pct": round(100 * float(dev["all"].sum()) / npos, 3),
                               "positions_by_lang": dict(n_pos), "classes_n": dev.reset_index().to_dict("records"),
                               "classes_pct_points": dev_pct.to_dict(), "by_source_lang_pct_points": dsrc.to_dict(), "top": {k: v.most_common(15) for k, v in dv_ex.items()}}}
    json.dump(tables, open(os.path.join(OUT, "q1_tables.json"), "w"), ensure_ascii=False, indent=1, default=int)
    o = ["# Q1 reading plane: convention classes\n", f"train reading tokens: {NTOK}\n", "## share of train reading tokens changed by each rule family (%)\n",
         json.dumps(overall), "\n### by source\n", sh["all"].to_string(index=False), "\n### by source x language\n", sh["src_lang"].to_string(index=False),
         "\n### by source x language x period (cells >= 30k tokens)\n", sh["src_lang_period"].to_string(index=False),
         f"\n## both-source texts (Oracc curated vs CDLI), {len(P)} sign-matched positions; readings differ: {100 * P['diff'].mean():.2f}%\n",
         "counts\n", bs.to_string(), "\n% of positions (per language column)\n", bs_pct.to_string(),
         f"\nnear-identical editions ({len(near)} texts, {len(pn)} positions): differ {100 * len(dn) / max(1, len(pn)):.2f}%: {json.dumps(near_tab, ensure_ascii=False)}",
         f"\nOracc project 'cdli' (Oracc's own copy of the CDLI ATF): {json.dumps(cdli_copy)}",
         f"\n## dev: bare-sign transliteration errors of base-long by class ({npos} positions, error {100 * dev['all'].sum() / npos:.2f}%)\n",
         dev.to_string(), "\npoints of error rate\n", dev_pct.to_string(),
         "\npoints of error rate by source group and language (a = value-name classes, b numerals, c logogram naming, d determinative, e genuine)\n", dsrc.to_string(), "\n## top pairs per class (both-source: Oracc | CDLI)\n"]
    for k, v in top_c.items():
        o.append(f"- {k}: " + ", ".join(f"{a} {n}" for a, n in v))
    o.append("\n## top dev confusions per class (gold -> predicted)\n")
    for k, v in dv_ex.items():
        o.append(f"- {k}: " + ", ".join(f"{a} {n}" for a, n in v.most_common(15)))
    open(os.path.join(OUT, "q1_report.md"), "w").write("\n".join(o) + "\n")
    print("\n".join(o))


if __name__ == "__main__":
    main()
