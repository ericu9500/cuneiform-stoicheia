"""Build the sign-level pretraining corpus from Oracc + CDLI.

1. Oracc: every project zip -> texts (sign dicts with OSL oids). A text id seen in several projects keeps
   the version with the most signs.
2. CDLI: every text of the 2022 dump in a cuneiform language other than proto-cuneiform / proto-Elamite,
   converted with cun.cdli; a P-number already in Oracc keeps the Oracc version (curated sign ids).
3. Held out (never in pretraining): every text the UR study holds out (annotated tablets incl. the excluded
   gold files, OOD sets, adaptation-validation) and their near-duplicates / parallel exemplars (NearDup).
4. Splits (one label per text, stored in the corpus): dev (2,000 texts; checkpoint and schedule
   decisions), test (2,000 texts; never used for a choice), train; train texts that near-duplicate a dev
   or test text (line-set Jaccard >= 0.9 or sign 8-gram containment >= 0.5) are 'excluded'. Admissible
   readings per sign (sign_reads) are counted on train texts only. `version` hashes texts, splits and
   vocabularies; every derived file records it and is checked against it.
Oracc graphemes exported without an oid are mapped through their reading (SignMap.lookup_oracc); all
readings are normalised to one convention (signmap.norm_read: lower case, g for ŋ).

Output: data/proc/corpus.pkl.gz (vocabularies, per-text metadata, per-line plane arrays), stats json.

`--round 3` (corpus v3, PLAN_v3.md stage 0.1; needs CUN_PROC, e.g. CUN_PROC=data/proc_v3): the same texts,
lines and vocabularies, built by the same code path up to the v2 split, then relabelled (round3()):
  test1     v2's test texts (seen once; never trained on)
  devA/devB v2's dev texts halved by text (seed 20261120); near-duplicates of each other and texts of one
            composite / object stay in the same half
  test2     2,000 texts drawn from v2's train texts that share no composite / object with dev or test1;
            stratified (cun.strata) in proportion to the strata's share of texts, at least 250 per main
            stratum; at most one text per composite
  excluded  v2's excluded texts, plus train texts that near-duplicate a held-out text (NearDup) or share a
            composite (Q-number) or a museum number with one (cun.composites)
  train     the rest
Per-text metadata added: stratum, genre_class, period_bin, dialect, composites, museum (+ the source of each).
The flag `dev` marks devA + devB (the dev loss of cun.train). Also written: reserved_ids.json (P-number,
composites and museum numbers of every held-out text, for deduplicating future data sources).
The default (--round 2) rebuilds corpus v2 unchanged.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import gzip
import json
import os
import pickle
import random
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UR = "/nobackup/proj/disk/dionysus/shared/UR"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(UR, "src"))
from cun import composites as comp_mod, strata  # noqa: E402
from cun import paths  # noqa: E402
from cun.cdli import convert_line  # noqa: E402
from cun.oracc import iter_project  # noqa: E402
from cun.signmap import SignMap, norm_read  # noqa: E402
from ur.atf import parse_atf  # noqa: E402

PROC = paths.PROC        # data/proc unless CUN_PROC is set
SEED = 20261118          # dev draw
SEED_TEST = 20261119     # test draw
SEED3 = 20261120         # round 3: devA / devB halves; SEED3 + 1: test2 draw
N_TEST2 = 2000
MIN_STRATUM = 250        # round 3: least number of test2 texts per main stratum
N_DEV = 2000
N_TEST = 2000
JACCARD = 0.9            # line-set Jaccard: same text (other edition / duplicate)
CONTAIN_RARE = 0.5       # share of the smaller text's RARE sign 8-grams found in the other: parallel exemplar
CONTAIN_ALL = 0.8        # share of ALL its 8-grams (exemplars of widely copied inscriptions, whose 8-grams are common)
DF_RARE = 20             # an 8-gram in more than 20 texts is a stock formula (year names, seals), ignored above
NGRAM = 8
EXCLUDE_LANG = {"qpc", "qpe", "nlc", "und"}       # proto-cuneiform, proto-Elamite, no linguistic content, undetermined
JOINS = ["w", "-", ".", ":", "+", ""]
DETS = ["", "pre", "post"]
BRKS = ["", "damaged", "missing"]
JOIN_ALIAS = {"—": "-", "–": "-", "/": "-"}      # Oracc in-word joiner of compound names; alternatives


def log(*a):
    print(*a, flush=True)


def coarse_lang(l):
    if not l:
        return "und"
    l = l.split("-")[0].lower()
    return {"sux": "sux", "akk": "akk", "hit": "hit", "xhu": "xhu", "elx": "elx", "xur": "xur", "uga": "uga",
            "qeb": "qeb", "qcu": "qcu", "arc": "arc", "peo": "peo", "egy": "egy", "grc": "grc"}.get(l, "oth")


class NearDup:
    """Near-duplicates of query texts among candidate texts: line-set Jaccard >= JACCARD; or containment of
    rare sign 8-grams (in <= DF_RARE texts) >= CONTAIN_RARE (shared / those of the smaller text, >= 5 of them);
    or containment of all 8-grams >= CONTAIN_ALL (>= 10 of them); or the same sequence of identified signs
    (>= 3; short inscriptions in many exemplars). Lines of only x / lacunae are ignored. The 8-gram tests catch parallel exemplars
    with different lineation or damage; stock formulae shared by distinct texts do not count. Signs are keyed by oid, else
    reading, else x/gap; 8-grams made only of x/gap are ignored."""

    def __init__(self, texts):
        tok = {}
        self.skel, self.skel_of = {}, {}
        rows, cols, grows, gcols = [], [], [], []
        lv = {}
        for i, t in enumerate(texts):
            seen = set()
            seq = []
            skel = []
            for l in t["lines"]:
                key = tuple(s["oid"] or s["read"] or s["unk"] for s in l["signs"])
                if all(s["unk"] for s in l["signs"]):  # a line of only x / lacunae identifies nothing
                    continue
                skel += [s["oid"] or s["read"] for s in l["signs"] if not s["unk"]]
                j = lv.setdefault(key, len(lv))
                if j not in seen:
                    seen.add(j)
                    rows.append(i)
                    cols.append(j)
            if len(skel) >= 3:                         # identical identified-sign sequence = same text
                self.skel.setdefault(hash(tuple(skel)), []).append(i)
                self.skel_of[i] = hash(tuple(skel))
                seq += [tok.setdefault(s["oid"] or s["read"] or "#unk", len(tok)) for s in l["signs"]]
            a = np.array(seq, np.uint64)
            if len(a) >= NGRAM:
                h = np.zeros(len(a) - NGRAM + 1, np.uint64)
                for k in range(NGRAM):
                    h = h * np.uint64(1000003) + a[k:len(a) - NGRAM + 1 + k] + np.uint64(1)
                unk = tok.get("#unk", -1)
                ok = np.ones(len(h), bool)
                if unk >= 0:
                    isu = (a == unk).astype(np.int64)
                    ok = np.convolve(isu, np.ones(NGRAM, np.int64), "valid") < NGRAM
                h = np.unique(h[ok])
                grows.append(np.full(len(h), i, np.int64))
                gcols.append(h)
        n = len(texts)
        self.L = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, len(lv)))
        self.Lsize = np.asarray(self.L.sum(1)).ravel()
        self.LT = self.L.T.tocsr()
        gr = np.concatenate(grows) if grows else np.zeros(0, np.int64)
        gc = np.concatenate(gcols) if gcols else np.zeros(0, np.uint64)
        u, inv = np.unique(gc, return_inverse=True)
        self.G = sp.csr_matrix((np.ones(len(gr), np.float32), (gr, inv)), shape=(n, len(u)))
        self.Gsize = np.asarray(self.G.sum(1)).ravel()
        self.GT = self.G.T.tocsr()
        df = np.asarray(self.G.sum(0)).ravel()            # texts per 8-gram
        rare = np.flatnonzero(df <= DF_RARE)
        self.R = self.G[:, rare].tocsr()                    # 8-grams that are not stock formulae
        self.Rsize = np.asarray(self.R.sum(1)).ravel()
        self.RT = self.R.T.tocsr()

    def find(self, queries, candidates):
        cand = np.zeros(self.L.shape[0], bool)
        cand[list(candidates)] = True
        q = np.asarray(sorted(queries), np.int64)
        out = set()
        for i in q.tolist():
            for j in self.skel.get(self.skel_of.get(i), ()):
                if j != i and cand[j]:
                    out.add(j)
        for s in range(0, len(q), 256):
            qq = q[s:s + 256]
            inter = (self.L[qq] @ self.LT).tocoo()
            qi = qq[inter.row]
            jac = inter.data / (self.Lsize[qi] + self.Lsize[inter.col] - inter.data)
            m = (jac >= JACCARD) & cand[inter.col] & (inter.col != qi)
            out.update(inter.col[m].tolist())
            for M, size, thr, min_n in ((self.R, self.Rsize, CONTAIN_RARE, 5), (self.G, self.Gsize, CONTAIN_ALL, 10)):
                inter = (M[qq] @ (self.RT if M is self.R else self.GT)).tocoo()
                qi = qq[inter.row]
                mn = np.minimum(size[qi], size[inter.col])
                m = (mn >= min_n) & (inter.data / np.maximum(mn, 1) >= thr) & cand[inter.col] & (inter.col != qi)
                out.update(inter.col[m].tolist())
        return out


def allocate(avail, n_total, main, min_main):
    """Texts per stratum: proportional to `avail` (largest remainder), then at least `min_main` (or all that
    are available) for the main strata, the excess taken from the largest stratum."""
    tot = sum(avail.values())
    exact = {s: n_total * v / tot for s, v in avail.items()}
    q = {s: int(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda s: (q[s] - exact[s], s))[:n_total - sum(q.values())]:
        q[s] += 1
    for s in main:
        if s in q:
            q[s] = max(q[s], min(min_main, avail[s]))
    while sum(q.values()) > n_total:
        q[max(q, key=lambda s: (q[s], s))] -= 1
    return q


def round3(texts, kept_idx, split2, dd, cdli_cat, stats):
    """Corpus v3 labels and per-text metadata from the v2 split (see the module docstring).
    Returns (split: index -> label, extra: index -> metadata dict, reserved: object for reserved_ids.json)."""
    ident = comp_mod.Identity(os.path.join(ROOT, "data", "raw", "oracc"), os.path.join(UR, "data/proc/texts_meta.jsonl.gz"))
    st = {}
    # ---------- strata: genre class, period bin, dialect
    extra = {}
    un_g, un_p = collections.Counter(), collections.Counter()
    for i in kept_idx:
        t = texts[i]
        proj = t["src"].partition(":")[2]
        c = cdli_cat.get(t["cdli"] or "", {})
        ents = ident.entries(t)
        dia = strata.dialect(t["lang"], t.get("tags"))
        # order of trust: the text's own catalogue entry, the CDLI catalogue (by P-number), other Oracc catalogues
        pb, pb_from = strata.period_bin([("catalogue", t["period"]), ("cdli", c.get("period"))] +
                                        [("oracc-other", e.get("period")) for _, e in ents],
                                        proj, t["lang"], dia, un_p)
        gc, gc_from = strata.genre_class([("catalogue", t["genre"]), ("cdli", c.get("genre"))] +
                                         [("oracc-other", e.get("genre")) for _, e in ents],
                                         c.get("subgenre") or t.get("subgenre"), proj, t["lang"], pb, un_g)
        mus = ident.museum(t)
        qs, how = ident.composites(t, mus)
        extra[i] = {"stratum": strata.stratum(t["lang"], gc), "genre_class": gc, "genre_from": gc_from,
                    "period_bin": pb, "period_from": pb_from, "dialect": dia,
                    "composites": qs, "composite_from": sorted(how), "museum": mus,
                    "museum_raw": ident.museum_raw(t)}
    st["genre_unmapped"], st["period_unmapped"] = dict(un_g), dict(un_p)
    # ---------- composites that are legends, not compositions: a seal legend or year formula catalogued as a
    # composite lists every administrative tablet that carries it (one has 2,399 Ur III tablets). A composite
    # most of whose texts are documentary is ignored (the analogue of the stock-formula rule of NearDup, C14).
    by_q = collections.defaultdict(list)
    for i in kept_idx:
        for q in extra[i]["composites"]:
            by_q[q].append(i)
    legend = {q for q, m in by_q.items() if sum(extra[i]["genre_class"] in strata.DOC for i in m) > 0.5 * len(m)}
    mus_n = collections.Counter(k for i in kept_idx for k in extra[i]["museum"])
    for i in kept_idx:
        e = extra[i]
        e["legend_composites"] = sorted(e["composites"] & legend)
        e["composites"] = sorted(e["composites"] - legend)
        e["museum"] = sorted(e["museum"])
        # identity keys: composites and museum numbers (a number on more than MUS_MAX texts is a collection)
        e["_keys"] = {"Q:" + q for q in e["composites"]} | {"M:" + k for k in e["museum"] if mus_n[k] <= comp_mod.MUS_MAX}
    st["composites"] = {"texts_with_composite": sum(bool(extra[i]["composites"]) for i in kept_idx),
                        "composites": len(by_q) - len(legend), "legend_composites_ignored": len(legend),
                        "texts_in_legend_composites": len({i for q in legend for i in by_q[q]}),
                        "link_kinds": dict(collections.Counter(h for i in kept_idx if extra[i]["composites"]
                                                               for h in extra[i]["composite_from"])),
                        "texts_with_museum_no": sum(bool(extra[i]["museum"]) for i in kept_idx),
                        "museum_keys_ignored_as_collections": sorted(k for k, v in mus_n.items() if v > comp_mod.MUS_MAX)}

    def sharing(held, cands):
        keys = set().union(*(extra[i]["_keys"] for i in held)) if held else set()
        return {i for i in cands if extra[i]["_keys"] & keys}

    dev = sorted(i for i in kept_idx if split2[i] == "dev")
    test1 = sorted(i for i in kept_idx if split2[i] == "test")
    train2 = sorted(i for i in kept_idx if split2[i] == "train")
    excl2 = {i for i in kept_idx if split2[i] == "excluded"}
    # ---------- devA / devB: halves by text; near-duplicates and texts of one composite / object stay together
    parent = {i: i for i in dev}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    n_pairs = 0
    for i in dev:
        for j in dd.find([i], dev):
            n_pairs += 1
            parent[root(j)] = root(i)
    first = {}
    for i in dev:
        for k in extra[i]["_keys"]:
            if k in first:
                parent[root(i)] = root(first[k])
            else:
                first[k] = i
    groups = collections.defaultdict(list)
    for i in dev:
        groups[root(i)].append(i)
    groups = [groups[k] for k in sorted(groups)]
    random.Random(SEED3).shuffle(groups)
    half = {"devA": [], "devB": []}
    for g in groups:                                   # a group goes to the half that has fewer texts so far
        half["devA" if len(half["devA"]) <= len(half["devB"]) else "devB"].extend(g)
    st["dev_halves"] = {"near_duplicate_pairs_in_dev": n_pairs // 2, "groups": len(groups),
                        "groups_of_several": sum(len(g) > 1 for g in groups), "largest_group": max(map(len, groups))}
    # ---------- test2: from v2 train texts that share no composite / object with dev or test1
    e_comp1 = sharing(dev + test1, train2)
    pool = [i for i in train2 if i not in e_comp1]
    by_s = collections.defaultdict(list)
    for i in pool:
        by_s[extra[i]["stratum"]].append(i)
    avail = {s: len(v) for s, v in by_s.items()}
    quota = allocate(avail, N_TEST2, strata.MAIN_STRATA, MIN_STRATUM)
    rng = random.Random(SEED3 + 1)
    test2, used, skipped = [], set(), collections.Counter()
    rest = []
    for s in strata.STRATA:
        cand = by_s.get(s, [])[:]
        rng.shuffle(cand)
        n = 0
        for k, i in enumerate(cand):
            if n >= quota.get(s, 0):
                rest += cand[k:] if s == max(avail, key=avail.get) else []
                break
            if extra[i]["_keys"] & used:               # one text per composite / object
                skipped[s] += 1
                continue
            test2.append(i)
            used |= extra[i]["_keys"]
            n += 1
    for i in rest:                                     # a stratum fell short of its quota: fill from the largest
        if len(test2) >= N_TEST2:
            break
        if not extra[i]["_keys"] & used:
            test2.append(i)
            used |= extra[i]["_keys"]
    test2_set = set(test2)
    st["test2"] = {"pool_texts": avail, "quota": quota, "skipped_same_composite": dict(skipped),
                   "drawn": dict(collections.Counter(extra[i]["stratum"] for i in test2)),
                   "near_duplicates_within_test2": len(dd.find(sorted(test2_set), sorted(test2_set)))}
    # ---------- excluded: near-duplicates (NearDup) or same composite / object as any held-out text
    held = sorted(set(dev) | set(test1) | test2_set)
    cand = [i for i in train2 if i not in test2_set]
    e_dup = dd.find(held, cand)
    e_comp2 = sharing(test2, cand)
    e_comp1 &= set(cand)
    new_excl = e_dup | e_comp1 | e_comp2
    held_q = {k for j in held for k in extra[j]["_keys"] if k[0] == "Q"}
    split = {}
    in_a = set(half["devA"])
    for i in kept_idx:
        s2 = split2[i]
        split[i] = ("devA" if i in in_a else "devB") if s2 == "dev" else "test1" if s2 == "test" else \
            "excluded" if s2 == "excluded" or i in new_excl else "test2" if i in test2_set else "train"
    sg = lambda ix: int(sum(texts[i]["n"] for i in ix))
    st["excluded"] = {
        "v2_excluded": [len(excl2), sg(excl2)], "newly_excluded": [len(new_excl), sg(new_excl)],
        "new_near_duplicate_of_held_out": [len(e_dup), sg(e_dup)],
        "new_same_composite_or_object_as_dev_test1": [len(e_comp1), sg(e_comp1)],
        "new_same_composite_or_object_as_test2": [len(e_comp2), sg(e_comp2)],
        "new_composite_only_not_near_duplicate": [len((e_comp1 | e_comp2) - e_dup), sg((e_comp1 | e_comp2) - e_dup)],
        "new_by_museum_number_only": len({i for i in new_excl - e_dup if not (extra[i]["_keys"] & held_q)}),
        "by_stratum": dict(collections.Counter(extra[i]["stratum"] for i in new_excl)),
        "v2_train": [len(train2), sg(train2)]}
    st["held_out_with_composite"] = {lab: [sum(bool(extra[i]["composites"]) for i in ix), len(ix)] for lab, ix in
                                     (("devA", half["devA"]), ("devB", half["devB"]), ("test1", test1), ("test2", test2))}
    # ---------- tables: texts / signs per split x stratum and split x period bin; coverage of the mappings
    for name in ("stratum", "period_bin"):
        tab = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for i in kept_idx:
            c = tab[extra[i][name]][split[i]]
            c[0] += 1
            c[1] += texts[i]["n"]
        st["by_" + name] = {k: dict(v) for k, v in tab.items()}
    for name in ("genre_from", "period_from", "genre_class", "dialect"):
        tab = collections.defaultdict(lambda: [0, 0])
        for i in kept_idx:
            c = tab[str(extra[i][name])]
            c[0] += 1
            c[1] += texts[i]["n"]
        st[name] = dict(tab)
    no_genre = [i for i in kept_idx if not texts[i]["genre"]]
    st["no_genre_in_own_catalogue"] = [len(no_genre), sg(no_genre)]
    st["no_genre_resolved_by"] = dict(collections.Counter(extra[i]["genre_from"] for i in no_genre))
    stats["round3"] = st
    reserved = [{"id": texts[i]["id"], "split": split[i], "src": texts[i]["src"], "p": texts[i]["cdli"],
                 "q": extra[i]["composites"], "q_legend": extra[i]["legend_composites"],
                 "museum": extra[i]["museum_raw"], "museum_keys": extra[i]["museum"]} for i in held]
    for e in extra.values():
        del e["_keys"], e["museum_raw"]
    return split, extra, reserved


def apply_v3_fixes(keep, ksplit, smap):
    """Apply the encoding changes marked in s['_v3'] (see main); returns counts for the stats."""
    n = collections.Counter()
    num = collections.defaultdict(collections.Counter)
    for t, sp_ in zip(keep, ksplit):
        src = t["src"].split(":")[0]
        for l in t["lines"]:
            for s in l["signs"]:
                fix = s.get("_v3")
                if not fix:
                    continue
                if fix[0] == "x":
                    n[f"x:{src}"] += 1
                    n[f"x:split:{sp_}"] += 1
                    n["x:damaged" if s["brk"] == "damaged" else "x:missing" if s["brk"] == "missing" else "x:intact"] += 1
                    s.update(oid="", sign="", read="", unk="x")
                else:
                    n[f"numeral:{src}"] += 1
                    n[f"numeral:split:{sp_}"] += 1
                    num[f"{t['lang']} {fix[2]}"][f"{src}: {smap.names.get(s['oid'], s['oid'] or '<unk-sign>')} -> "
                                                 f"{smap.names[fix[1]]}"] += 1
                    s.update(oid=fix[1], sign=smap.names[fix[1]], read=norm_read(fix[2]))
    return {"x_oid": smap.oid_x, "numeral_canon": dict(smap.num_canon),
            "numeral_canon_names": {r: smap.names[o] for r, o in smap.num_canon.items()},
            "numeral_readings_split_in_oracc": smap.num_split, "counts": dict(n),
            "numeral_changes": {k: dict(v) for k, v in sorted(num.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=2, choices=(2, 3),
                    help="2: corpus v2 (default, data/proc); 3: corpus v3 labels and metadata (set CUN_PROC)")
    args = ap.parse_args()
    if args.round == 3 and PROC == paths.DEFAULT_PROC:
        raise SystemExit("--round 3 must not overwrite corpus v2: set CUN_PROC (e.g. CUN_PROC=data/proc_v3)")
    os.makedirs(PROC, exist_ok=True)
    stats = {}
    # ---------- Oracc
    oracc = {}
    pairs = collections.defaultdict(collections.Counter)
    lang_pairs = collections.defaultdict(collections.Counter)
    n_proto = collections.Counter()
    for z in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "oracc", "*.zip"))):
        n = 0
        for tid, proj, cat, lines in iter_project(z):
            ns = sum(len(l["signs"]) for l in lines)
            if not ns:
                continue
            lc = collections.Counter(s["lang"] for l in lines for s in l["signs"])
            period = (cat.get("period") or "").lower()
            if period.startswith(("uruk", "proto-elamite", "proto-cuneiform")) or \
                    (lc.get("qpc", 0) + lc.get("qpe", 0)) >= 0.2 * ns:
                n_proto[proj] += 1                     # proto-cuneiform / proto-Elamite (C3): never used
                continue
            if tid not in oracc or ns > oracc[tid]["n"]:
                langs = collections.Counter(s["lang"] for l in lines for s in l["signs"] if s["lang"])
                oracc[tid] = {"id": tid, "src": "oracc:" + proj, "n": ns, "lines": lines,
                              "lang": coarse_lang(langs.most_common(1)[0][0] if langs else ""),
                              "period": cat.get("period"), "genre": cat.get("genre"),
                              "cdli": cat.get("cdli_id") or (tid if tid.startswith("P") else None),
                              "subgenre": cat.get("subgenre"), "tags": cat.get("_lang_tags")}   # round 3 only
            for l in lines:
                for s in l["signs"]:
                    if s["oid"] and s["read"]:
                        pairs[s["read"]][s["oid"]] += 1
                        lang_pairs[(coarse_lang(s["lang"]), s["read"])][s["oid"]] += 1
            n += 1
        log(os.path.basename(z), n)
    stats["oracc_proto_skipped"] = dict(n_proto)
    stats["oracc_texts"] = len(oracc)
    log("oracc texts", len(oracc))
    smap = SignMap(pairs, os.path.join(ROOT, "data", "raw", "osl.asl"), lang_pairs)
    # graphemes that some Oracc exports give without an oid (mostly Neo-Assyrian projects): map the reading
    n_no, n_map = 0, 0
    for t in oracc.values():
        for l in t["lines"]:
            for s in l["signs"]:
                if not s["oid"] and not s["unk"] and s["read"]:
                    n_no += 1
                    o = smap.lookup_oracc(s["read"], coarse_lang(s["lang"]))
                    if o:
                        s["oid"] = o
                        n_map += 1
    stats["oracc_no_oid"] = n_no
    stats["oracc_no_oid_mapped"] = n_map
    stats["oracc_no_oid_unmapped_top"] = smap.miss_oracc.most_common(50)
    log("oracc graphemes without oid", n_no, "mapped", n_map)

    # ---------- CDLI
    cat = {}
    for l in gzip.open(os.path.join(UR, "data/proc/texts_meta.jsonl.gz"), "rt"):
        d = json.loads(l)
        cat[d["pid"]] = d
    oracc_p = {t["cdli"] for t in oracc.values() if t["cdli"]}
    cdli = {}
    n_skip = collections.Counter()
    for t in parse_atf(os.path.join(UR, "data/raw/cdli/cdliatf_unblocked.atf")):
        lang = (t.lang or "und").replace("other:", "")
        if lang in EXCLUDE_LANG:
            n_skip[lang] += 1
            continue
        if t.pid in oracc_p:
            n_skip["in_oracc"] += 1
            continue
        lines = []
        for l in t.lines:
            sg = convert_line(l.raw, smap, coarse_lang(lang))
            if sg:
                lines.append({"label": l.label, "signs": sg})
        ns = sum(len(l["signs"]) for l in lines)
        if not ns:
            continue
        m = cat.get(t.pid, {})
        cdli[t.pid] = {"id": t.pid, "src": "cdli", "n": ns, "lines": lines, "lang": coarse_lang(lang),
                       "period": m.get("period"), "genre": m.get("genre"), "cdli": t.pid}
    stats["cdli_texts"] = len(cdli)
    stats["cdli_skipped"] = dict(n_skip)
    stats["unmapped_cdli_readings_top"] = smap.miss.most_common(50)
    stats["unmapped_cdli_tokens"] = sum(smap.miss.values())
    log("cdli texts", len(cdli), dict(n_skip))
    texts = list(oracc.values()) + list(cdli.values())
    if args.round == 3:
        # v3 encoding fixes, decided here on the source readings but APPLIED after the splits (apply_v3_fixes),
        # so that the text selection, the near-duplicate tests and the v2 split are those of corpus v2:
        # (1) the illegible sign: Oracc 'x' graphemes carry the OSL sign X (or no oid) and became a real sign;
        #     they become the special <x>, as CDLI's x. X inside compounds (|KA×X|) is a different sign: kept.
        # (2) numeral readings whose sign Oracc itself codes in two ways (SignMap.num_canon) -> OSL's sign.
        for t in texts:
            for l in t["lines"]:
                for s in l["signs"]:
                    if s["unk"]:
                        continue
                    if s["oid"] == smap.oid_x or (not s["oid"] and s["read"] in ("x", "X")):
                        s["_v3"] = ("x", None, None)
                    else:
                        o, r = smap.numeral_v3(s["read"])
                        if o and (o != s["oid"] or r != s["read"].lower()):
                            s["_v3"] = ("num", o, r)
    for t in texts:                                   # one reading convention for both sources
        for l in t["lines"]:
            for s in l["signs"]:
                s["read"] = norm_read(s["read"])

    # ---------- near-duplicate machinery: line sets (Jaccard) and sign 8-grams (containment)
    dd = NearDup(texts)

    # ---------- held out (UR study): never in pretraining, nor their near-duplicates / parallel exemplars
    spl = json.load(open(os.path.join(UR, "data/proc/splits.json")))
    held = set(p for c in spl["annotated_clusters"] for p in c) | set(spl["excluded_gold"])
    for v in spl["ood"].values():
        held |= set(v)
    held |= set(spl["adapt_val"])
    q = [i for i, t in enumerate(texts) if (t["cdli"] in held) or (t["id"] in held)]
    remove = set(q) | dd.find(q, range(len(texts)))
    stats["heldout_texts_found"] = len(q)
    stats["removed_total"] = len(remove)
    log("held out found", len(q), "removed incl. near-duplicates / parallels", len(remove))
    # texts that are almost entirely unidentified (fragments: < 2 identified signs or < 50% identified)
    def n_ident(t):
        return sum(1 for l in t["lines"] for s in l["signs"] if s["oid"] and not s["unk"])
    weak = {i for i, t in enumerate(texts) if n_ident(t) < 2 or n_ident(t) < 0.5 * t["n"]}
    stats["weak_texts_dropped"] = len(weak - remove)
    log("weak (mostly unidentified) texts dropped", len(weak - remove))
    kept_idx = [i for i in range(len(texts)) if i not in remove and i not in weak]

    # ---------- dev and test splits; train texts that near-duplicate one of them are excluded
    r = random.Random(SEED)
    order = kept_idx[:]
    r.shuffle(order)
    dev_set = set(order[:N_DEV])
    rt = random.Random(SEED_TEST)
    pool = order[N_DEV:]
    dev_dups = dd.find(sorted(dev_set), pool)          # test texts must not near-duplicate dev texts either
    test_set = set(rt.sample([i for i in pool if i not in dev_dups], N_TEST))
    cand = [i for i in kept_idx if i not in dev_set and i not in test_set]
    excl = dd.find(sorted(dev_set | test_set), cand)
    split = {}
    for i in kept_idx:
        split[i] = "dev" if i in dev_set else "test" if i in test_set else "excluded" if i in excl else "train"
    stats["splits"] = dict(collections.Counter(split.values()))
    stats["excluded_signs"] = int(sum(texts[i]["n"] for i in excl))
    log("splits", stats["splits"], "excluded signs", stats["excluded_signs"])
    extra = None
    if args.round == 3:
        stats["splits_v2"] = stats["splits"]
        split, extra, reserved = round3(texts, kept_idx, split, dd, cat, stats)
        stats["splits"] = dict(collections.Counter(split.values()))
        log("round 3 splits", stats["splits"])
    keep = [texts[i] for i in kept_idx]
    ksplit = [split[i] for i in kept_idx]
    if args.round == 3:
        stats["round3"]["fixes"] = apply_v3_fixes(keep, ksplit, smap)
        log("round 3 fixes", json.dumps(stats["round3"]["fixes"], ensure_ascii=False))
        # the fixes change sign sequences (an illegible sign no longer distinguishes two texts), so the
        # near-duplicate tests are repeated on the fixed texts: train texts that now near-duplicate a
        # held-out text are excluded
        dd3 = NearDup(keep)
        held3 = [i for i, sp_ in enumerate(ksplit) if sp_ in ("devA", "devB", "test1", "test2")]
        cand3 = [i for i, sp_ in enumerate(ksplit) if sp_ == "train"]
        ex3 = dd3.find(held3, cand3)
        for i in ex3:
            ksplit[i] = "excluded"
        stats["round3"]["excluded_after_fixes"] = len(ex3)
        stats["splits"] = dict(collections.Counter(ksplit))
        log("excluded after fixes", len(ex3), "splits", stats["splits"])
        del dd3

    # ---------- vocabularies and arrays
    oid_c = collections.Counter(s["oid"] for t in keep for l in t["lines"] for s in l["signs"] if s["oid"])
    sign_vocab = ["<pad>", "<mask>", "<blank>", "<unk-sign>", "<x>", "<gap>"] + [o for o, c in oid_c.most_common() if c >= 2]
    sid = {o: i for i, o in enumerate(sign_vocab)}
    read_c = collections.Counter((s["oid"], s["read"]) for t in keep for l in t["lines"] for s in l["signs"] if s["oid"])
    read_vocab = ["<none>", "<mask>"] + sorted({rd for (o, rd), c in read_c.items() if c >= 2})
    rid = {x: i for i, x in enumerate(read_vocab)}
    # admissible readings per sign (constrain decoders and HMM states): from TRAIN texts only
    train_read_c = collections.Counter((s["oid"], s["read"]) for t, sp_ in zip(keep, ksplit) if sp_ == "train"
                                       for l in t["lines"] for s in l["signs"] if s["oid"])
    sign_reads = collections.defaultdict(set)
    for (o, rd), c in train_read_c.items():
        if c >= 2 and o in sid and rd in rid:
            sign_reads[sid[o]].add(rid[rd])
    langs = sorted({t["lang"] for t in keep})
    out_lines = []
    meta = []
    unknown_join = collections.Counter()
    for ti, t in enumerate(keep):
        meta.append({k: t[k] for k in ("id", "src", "lang", "period", "genre", "cdli", "n")}
                    | {"split": ksplit[ti], "dev": ksplit[ti] in ("dev", "devA", "devB")}
                    | (extra[kept_idx[ti]] if extra else {}))
        for l in t["lines"]:
            sg = l["signs"]
            a = np.zeros((5, len(sg)), np.int32)
            for k2, s in enumerate(sg):
                if s["unk"]:
                    a[0, k2] = sid["<gap>"] if s["unk"] == "gap" else sid["<x>"]
                else:
                    a[0, k2] = sid.get(s["oid"], sid["<unk-sign>"])
                a[1, k2] = rid.get(s["read"], 0) if s["oid"] in sid else 0
                j = JOIN_ALIAS.get(s["join"], s["join"])
                if j not in JOINS:
                    unknown_join[s["join"]] += 1
                    j = "-"
                a[2, k2] = JOINS.index(j)
                a[3, k2] = DETS.index(s["det"].split(":")[0]) if s["det"] else 0
                a[4, k2] = BRKS.index(s["brk"]) if s["brk"] in BRKS else 0
            out_lines.append((ti, a))
    stats["unknown_joins"] = dict(unknown_join)
    from cun.oracc import STATS as ORACC_STATS
    stats["oracc_reader"] = dict(ORACC_STATS)
    stats.update(texts_kept=len(keep), lines=len(out_lines),
                 signs=int(sum(a.shape[1] for _, a in out_lines)), sign_vocab=len(sign_vocab),
                 read_vocab=len(read_vocab), langs=dict(collections.Counter(t["lang"] for t in keep)),
                 by_src=dict(collections.Counter(t["src"].split(":")[0] for t in keep)))
    sb = collections.Counter()
    for t in keep:
        sb[t["lang"]] += t["n"]
    stats["signs_by_lang"] = dict(sb)
    hsh = hashlib.sha1(json.dumps([[m["id"], m["split"]] for m in meta] + [sign_vocab, read_vocab],
                                  ensure_ascii=False).encode())
    for ti, a in out_lines:                             # line contents too: a converter change changes the version
        hsh.update(np.int64(ti).tobytes())
        hsh.update(np.ascontiguousarray(a).tobytes())
    version = hsh.hexdigest()[:12]
    stats["version"] = version
    obj = {"version": version, "sign_vocab": sign_vocab, "sign_names": [smap.names.get(o, o) for o in sign_vocab],
           "read_vocab": read_vocab, "sign_reads": {k: sorted(v) for k, v in sign_reads.items()},
           "joins": JOINS, "dets": DETS, "brks": BRKS, "langs": langs, "texts": meta, "lines": out_lines}
    with gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    json.dump(stats, open(os.path.join(PROC, "corpus_stats.json"), "w"), indent=1, default=str)
    if args.round == 3:
        json.dump({"version": version,
                   "note": "identifiers of every held-out text of corpus v3 (devA, devB, test1, test2): P-number, "
                           "composites (Q-numbers; q_legend = seal legends etc., not used for deduplication), museum "
                           "numbers as in the catalogues and normalised (cun.composites.museum_keys)",
                   "p_numbers": sorted({r["p"] for r in reserved if r["p"]}),
                   "q_numbers": sorted({q for r in reserved for q in r["q"]}),
                   "museum_keys": sorted({k for r in reserved for k in r["museum_keys"]}),
                   "texts": reserved}, open(os.path.join(PROC, "reserved_ids.json"), "w"), indent=0, ensure_ascii=False)
    log(json.dumps({k: v for k, v in stats.items() if not k.startswith("unmapped")}, indent=1, default=str))


if __name__ == "__main__":
    main()
