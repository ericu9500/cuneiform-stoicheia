"""Head-to-head, step 1 (torch-free; login node): positions of the UR study (system B) where the masked
WordPiece token is exactly one whole sign of the line, with the sign-model (system A) inputs.

  HF_HOME=/nobackup/proj/disk/dionysus/shared/hf_cache \
  /nobackup/proj/disk/dionysus/shared/UR/envs/x86/bin/python scripts/h2h_positions.py

Alignment. A UR line is " ".join(words) (cleaned ATF words), tokenised by mBERT WordPiece with character
offsets (re-tokenised here; the ids must equal the stored ids). The same string is tokenised into signs by
cun.cdli.tokenize_line; every sign text is located in the string left to right (only separators, braces
and spaces may lie between consecutive signs, else the line is not aligned). A position is MATCHED when
the character span of its token equals the character span of one sign, that sign converts to exactly one
sign dict (cun.cdli.convert_line) with a sign id of the corpus vocabulary, and it is not x / '...'.
Output: results/headtohead/positions.pkl.gz (+ signmap.pkl cache, alignment_sample.txt, positions_stats.json)
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UR = "/nobackup/proj/disk/dionysus/shared/UR"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(UR, "src"))
from cun import cdli  # noqa: E402
from cun.consts import FIRST_REAL_SIGN  # noqa: E402
from cun.oracc import iter_project  # noqa: E402
from cun.signmap import SignMap, norm_read  # noqa: E402

from cun.paths import PROC, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "headtohead")
JOINS = ["w", "-", ".", ":", "+", ""]
DETS = ["", "pre", "post"]
JOIN_ALIAS = {"—": "-", "–": "-", "/": "-"}
LINE_END = 6
BETWEEN = set("-.:+{}_ ")
LANG = "sux"
_orig_tokenize = cdli.tokenize_line


def log(*a):
    print(*a, flush=True)


def coarse_lang(l):                      # as scripts/build_corpus.py
    if not l:
        return "und"
    l = l.split("-")[0].lower()
    return {"sux": "sux", "akk": "akk", "hit": "hit", "xhu": "xhu", "elx": "elx", "xur": "xur", "uga": "uga",
            "qeb": "qeb", "qcu": "qcu", "arc": "arc", "peo": "peo", "egy": "egy", "grc": "grc"}.get(l, "oth")


def build_signmap():
    """The SignMap exactly as scripts/build_corpus.py builds it (reading -> oid pairs of every Oracc text
    that is not proto-cuneiform)."""
    p = os.path.join(OUT, "signmap.pkl")
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    pairs = collections.defaultdict(collections.Counter)
    lang_pairs = collections.defaultdict(collections.Counter)
    for z in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "oracc", "*.zip"))):
        for tid, proj, cat, lines in iter_project(z):
            ns = sum(len(l["signs"]) for l in lines)
            if not ns:
                continue
            lc = collections.Counter(s["lang"] for l in lines for s in l["signs"])
            period = (cat.get("period") or "").lower()
            if period.startswith(("uruk", "proto-elamite", "proto-cuneiform")) or \
                    (lc.get("qpc", 0) + lc.get("qpe", 0)) >= 0.2 * ns:
                continue
            for l in lines:
                for s in l["signs"]:
                    if s["oid"] and s["read"]:
                        pairs[s["read"]][s["oid"]] += 1
                        lang_pairs[(coarse_lang(s["lang"]), s["read"])][s["oid"]] += 1
        log("oracc", os.path.basename(z))
    smap = SignMap(pairs, os.path.join(ROOT, "data", "raw", "osl.asl"), lang_pairs)
    pickle.dump(smap, open(p, "wb"), protocol=5)
    return smap


class Conv:
    def __init__(self, corpus, smap):
        self.smap = smap
        self.sid = {o: i for i, o in enumerate(corpus["sign_vocab"])}
        self.rid = {x: i for i, x in enumerate(corpus["read_vocab"])}
        self.v3 = corpus["version"] != "5da416adb1cc"

    def fix(self, signs):
        """Corpus v3 encoding (scripts/build_corpus.py, round 3): the illegible sign becomes <x>, numeral
        readings that Oracc codes in two ways take OSL's sign. In place; applying it twice changes nothing."""
        if not self.v3:
            return signs
        sm = self.smap
        for s in signs:
            if s["unk"]:
                continue
            if s["oid"] == sm.oid_x or (not s["oid"] and s["read"] in ("x", "X")):
                s.update(oid="", sign="", read="", unk="x")
            else:
                o, r = sm.numeral_v3(s["read"])
                if o and (o != s["oid"] or r != s["read"].lower()):
                    s.update(oid=o, sign=sm.names[o], read=norm_read(r))
        return signs

    def planes(self, signs):
        """(4, n) int32 array sign / read / join / det, as scripts/build_corpus.py."""
        sid, rid = self.sid, self.rid
        self.fix(signs)
        a = np.zeros((4, len(signs)), np.int32)
        for k, s in enumerate(signs):
            if s["unk"]:
                a[0, k] = sid["<gap>"] if s["unk"] == "gap" else sid["<x>"]
            else:
                a[0, k] = sid.get(s["oid"], sid["<unk-sign>"])
            a[1, k] = rid.get(norm_read(s["read"]), 0) if s["oid"] in sid else 0
            j = JOIN_ALIAS.get(s["join"], s["join"])
            a[2, k] = JOINS.index(j if j in JOINS else "-")
            a[3, k] = DETS.index(s["det"].split(":")[0]) if s["det"] else 0
        return a

    def line(self, text):
        """-> (sign dicts, per ASCII sign: (char start, char end, first dict, n dicts)) or (dicts, None) when
        the sign texts cannot be located in the string."""
        items = _orig_tokenize(text)
        dicts, spans, cur, ok = [], [], 0, True
        for it in items:
            cdli.tokenize_line = lambda _c, it=it: [list(it)]
            try:
                d = cdli.convert_line(None, self.smap, LANG)
            finally:
                cdli.tokenize_line = _orig_tokenize
            s = it[0]
            i = text.find(s, cur) if ok else -1
            if i < 0 or not set(text[cur:i]) <= BETWEEN:
                ok = False
            else:
                spans.append((i, i + len(s), len(dicts), len(d)))
                cur = i + len(s)
            dicts += self.fix(d)
        if ok and not set(text[cur:]) <= BETWEEN:
            ok = False
        return dicts, (spans if ok else None)


def check_signmap(corpus, conv, n=400):
    """The rebuilt SignMap must reproduce the corpus: convert CDLI train texts from the ATF dump again and
    compare the plane arrays line by line."""
    from ur.atf import parse_atf
    want = {t["id"]: ti for ti, t in enumerate(corpus["texts"]) if t["src"] == "cdli"}
    by = collections.defaultdict(list)
    for ti, a in corpus["lines"]:
        by[ti].append(a)
    nt = nl = bad = 0
    for t in parse_atf(os.path.join(UR, "data/raw/cdli/cdliatf_unblocked.atf")):
        if t.pid not in want:
            continue
        lang = coarse_lang((t.lang or "und").replace("other:", ""))
        arrs = []
        for l in t.lines:
            sg = cdli.convert_line(l.raw, conv.smap, lang)
            if sg:
                arrs.append(conv.planes(sg))
        ref = by[want[t.pid]]
        nt += 1
        if len(arrs) != len(ref):
            bad += 1
            continue
        for a, b in zip(arrs, ref):
            nl += 1
            bad += not np.array_equal(a, b[:4])
        if nt >= n:
            break
    return {"texts": nt, "lines": nl, "mismatching_lines_or_texts": bad}


def main():
    import ur.data as U
    from ur.tok import get_tokenizer
    stats = {}
    smap = build_signmap()
    corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    assert corpus["version"] in ("5da416adb1cc", "2d838c475606"), corpus["version"]
    conv = Conv(corpus, smap)
    stats["signmap_check"] = check_signmap(corpus, conv)
    log("signmap check", stats["signmap_check"])
    assert stats["signmap_check"]["mismatching_lines_or_texts"] == 0

    D = U.load()
    L = D["lines"]
    sets = [f"fold{k}/test" for k in range(5)] + sorted(k for k in D["positions"] if k.startswith("ood/"))
    lids = sorted({int(l) for s in sets for l, _ in D["positions"][s]})
    tok = get_tokenizer()
    texts = [" ".join(L[l]["words"]) for l in lids]
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    recur = set()
    with gzip.open(os.path.join(UR, "data/proc/adapt_UrIII.txt.gz"), "rt") as fh:
        for ln in fh:
            recur.add(ln.rstrip("\n"))
    log("adapt_UrIII lines", len(recur))

    # ---- tablets (all their lines, raw ATF content converted as in pretraining)
    tab = {}
    for f in ("annotated", "ood_texts"):
        with gzip.open(os.path.join(UR, "data/proc", f + ".jsonl.gz"), "rt") as fh:
            for ln in fh:
                d = json.loads(ln)
                tab.setdefault(d["pid"], d)
    need = sorted({L[l]["pid"] for l in lids})
    lid_tl = {}                                           # lid -> index of the line in its tablet
    by_pid = collections.defaultdict(list)
    for l in lids:
        by_pid[L[l]["pid"]].append(l)
    n_unloc = 0
    for pid in need:
        tl = tab[pid]["lines"]
        used = set()
        for l in by_pid[pid]:
            x = L[l]
            c = [i for i, t in enumerate(tl) if i not in used and t["words"] == x["words"]
                 and t["label"] == x["label"] and t["section"] == x["section"]]
            if not c:
                n_unloc += 1
                continue
            lid_tl[l] = c[0]
            used.add(c[0])
    stats["lines"] = len(lids)
    stats["lines_not_located_in_tablet"] = n_unloc
    log("lines", len(lids), "not located in their tablet", n_unloc)

    # ---- lines: tokens, signs, alignment
    line_rec = {}
    n_ids_diff = n_unaligned = n_inline = 0
    for l, text, ids, offs in zip(lids, texts, enc["input_ids"], enc["offset_mapping"]):
        x = L[l]
        stored = [int(v) for v in x["ids"]]
        if list(ids)[:len(stored)] != stored:
            n_ids_diff += 1
            continue
        if "$" in text:                                   # UR D26: lines with inline comments are excluded
            n_inline += 1
            continue
        if l not in lid_tl:
            continue
        dicts, spans = conv.line(text)
        assert [(d["oid"], d["read"], d["join"], d["det"]) for d in dicts] == \
               [(d["oid"], d["read"], d["join"], d["det"]) for d in conv.fix(cdli.convert_line(text, smap, LANG))]
        if spans is None:
            n_unaligned += 1
            continue
        toks = tok.convert_ids_to_tokens(stored)
        for (a, b), t in zip(offs, toks):                 # offsets reproduce the token ('##' = continuation)
            assert text[a:b] == (t[2:] if t.startswith("##") else t) or t == "[UNK]", (text, t, a, b)
        line_rec[l] = {"text": text, "offs": [tuple(o) for o in offs][:len(stored)], "toks": toks, "dicts": dicts,
                       "spans": spans, "planes": conv.planes(dicts)}
    stats.update(lines_ids_differ=n_ids_diff, lines_inline_comment=n_inline, lines_signs_not_located=n_unaligned)
    log("ids differ", n_ids_diff, "inline comment", n_inline, "signs not located", n_unaligned)

    # ---- tablets as sign sequences; the evaluated line enters in its 'words' form (what B sees)
    tablets = {}
    for pid in need:
        tl = tab[pid]["lines"]
        own = {lid_tl[l]: l for l in by_pid[pid] if l in line_rec}
        arrs, start = [], {}
        n = 0
        for i, t in enumerate(tl):
            if i in own:
                a = line_rec[own[i]]["planes"].copy()
                start[own[i]] = n
            else:
                sg = cdli.convert_line(t["raw"], smap, LANG)
                if not sg:
                    continue
                a = conv.planes(sg)
            if a.shape[1] == 0:
                continue
            a[2, -1] = LINE_END
            arrs.append(a)
            n += a.shape[1]
        tablets[pid] = {"planes": np.concatenate(arrs, 1) if arrs else np.zeros((4, 0), np.int32), "start": start}

    # ---- positions
    rv = corpus["read_vocab"]
    sign_reads = {int(k): set(v) for k, v in corpus["sign_reads"].items()}
    pos_out = []
    why = collections.Counter()
    by_set = collections.defaultdict(collections.Counter)
    for s in sets:
        for l, p in D["positions"][s]:
            l, p = int(l), int(p)
            cat = L[l]["cat"][p]
            by_set[s]["all"] += 1
            by_set[s]["all/" + cat] += 1
            r = line_rec.get(l)
            if r is None:
                why["line dropped"] += 1
                continue
            a, b = r["offs"][p]
            hit = [(k, sp) for k, sp in enumerate(r["spans"]) if sp[0] == a and sp[1] == b]
            if not hit:
                why["token is not one whole sign"] += 1
                continue
            k, (_, _, d0, nd) = hit[0]
            if nd != 1:
                why["ASCII sign converts to several signs"] += 1
                continue
            d = r["dicts"][d0]
            g = int(r["planes"][0, d0])
            if d["unk"] or g < FIRST_REAL_SIGN:
                why["sign without a sign id"] += 1
                continue
            gr = norm_read(d["read"])
            grid = conv.rid.get(gr, 0)
            x = L[l]
            pos_out.append({"set": s, "lid": l, "pos": p, "token": r["toks"][p], "cat": cat, "slot": d0,
                            "gold_sign": g, "gold_read": gr, "gold_read_id": grid,
                            "read_in_vocab": grid > 1, "read_admissible": grid in sign_reads.get(g, ()),
                            "pid": x["pid"], "cluster": int(x["cluster"]), "period": x["period"], "genre": x["genre"],
                            "recurring": r["text"] in recur, "tab_slot": tablets[x["pid"]]["start"][l] + d0})
            by_set[s]["matched"] += 1
            by_set[s]["matched/" + cat] += 1
    stats["unmatched_reasons"] = dict(why)
    stats["by_set"] = {s: dict(c) for s, c in by_set.items()}
    stats["n_matched"] = len(pos_out)
    used = sorted({x["lid"] for x in pos_out})
    obj = {"version": corpus["version"], "sets": sets, "positions": pos_out,
           "lines": {l: {"text": line_rec[l]["text"], "planes": line_rec[l]["planes"], "pid": L[l]["pid"],
                         "toks": line_rec[l]["toks"],
                         "signs": [(d["oid"], d["sign"], d["read"], d["join"], d["det"], d["unk"]) for d in line_rec[l]["dicts"]]}
                     for l in used},
           "tablets": {pid: tablets[pid]["planes"] for pid in sorted({x["pid"] for x in pos_out})},
           "lang": corpus["langs"].index(LANG), "stats": stats}
    with gzip.open(os.path.join(OUT, "positions.pkl.gz"), "wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    json.dump(stats, open(os.path.join(OUT, "positions_stats.json"), "w"), indent=1)
    log(json.dumps(stats, indent=1))

    # ---- sanity (a): 50 random matched positions, to be read by eye
    rng = np.random.default_rng(20261120)
    names = corpus["sign_names"]
    with open(os.path.join(OUT, "alignment_sample.txt"), "w") as fh:
        for i in rng.choice(len(pos_out), 50, replace=False):
            x = pos_out[int(i)]
            r = line_rec[x["lid"]]
            mk = " ".join(("[" + t + "]") if j == x["pos"] else t for j, t in enumerate(r["toks"]))
            sg = " ".join(("[" + (d["read"] or d["unk"]) + "]") if j == x["slot"] else (d["read"] or d["unk"])
                          for j, d in enumerate(r["dicts"]))
            ln = (f"{x['set']:18s} {x['pid']} | {r['text']}\n    tokens: {mk}\n    signs:  {sg}\n"
                  f"    token={x['token']!r} cat={x['cat']} sign={names[x['gold_sign']]} ({corpus['sign_vocab'][x['gold_sign']]}) "
                  f"reading={x['gold_read']!r} in_vocab={x['read_in_vocab']} admissible={x['read_admissible']}\n")
            fh.write(ln)
            print(ln, end="")

    # ---- sanity (b): no evaluated tablet in the sign model's train split
    pids = {x["pid"] for x in pos_out}
    inc = collections.Counter()
    for t in corpus["texts"]:
        if t.get("cdli") in pids or t.get("id") in pids:
            inc[t["split"]] += 1
    stats["evaluated_tablets"] = len(pids)
    stats["evaluated_tablets_in_corpus_by_split"] = dict(inc)
    json.dump(stats, open(os.path.join(OUT, "positions_stats.json"), "w"), indent=1)
    log("evaluated tablets", len(pids), "found in corpus (by split):", dict(inc))


if __name__ == "__main__":
    main()
