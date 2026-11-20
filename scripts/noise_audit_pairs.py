"""Noise audit, step 1: sign-level comparison of the two sources on texts present in BOTH (P-number in CDLI's dump
and in an Oracc project), restricted to texts whose corpus entry (the Oracc version) has split == 'train'.

Both versions are produced as scripts/build_corpus.py --round 3 produces them (same SignMap, v3 fixes for x and numerals, cun.oracc reader,
cun.cdli converter; Oracc graphemes without an oid mapped through their reading; readings through norm_read) and
aligned per text on sign ids with difflib (as scripts/check_cdli_conversion.py). Unlike that check, every text is
kept (no oid-cover filter, no sampling) and every aligned position is stored, with the planes of both sides.

Two extra flags that the corpus does not carry are recorded for classifying reading differences:
  o_logo  Oracc wrote the reading in capitals (role=logo / sign name)
  c_logo  the CDLI sign stands inside _..._ (Akkadian logogram markup); taken from a run-time copy of
          cun.cdli.tokenize_line that returns its `in_logo` state (the module itself is not changed)

Output (results/noise_audit/): pairs_matched.pkl.gz (DataFrame, one row per sign-matched position),
pairs_blocks.pkl.gz (non-matching alignment blocks, counted), pairs_texts.pkl.gz (per text: metadata and both
rendered sequences, for examples), pairs_summary.json.
  python scripts/noise_audit_pairs.py
"""
from __future__ import annotations

import collections
import difflib
import glob
import gzip
import inspect
import json
import os
import pickle
import sys
from multiprocessing import Pool

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UR = "/nobackup/proj/disk/dionysus/shared/UR"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(UR, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cun.cdli as cdli_mod  # noqa: E402
from cun.oracc import iter_project  # noqa: E402
from cun.signmap import SignMap, norm_read  # noqa: E402
from ur.atf import parse_atf  # noqa: E402

OUT = os.path.join(ROOT, "results/noise_audit")
NPROC = 8

# ---- run-time copy of tokenize_line that also returns in_logo (cun/cdli.py stays untouched)
_src = inspect.getsource(cdli_mod.tokenize_line)
_old = "out.append([s, det_kind(), join, brk, unk])"
assert _src.count(_old) == 1
_ns = dict(vars(cdli_mod))
exec(_src.replace("def tokenize_line(", "def tokenize_line_logo(").replace(_old, "out.append([s, det_kind(), join, brk, unk, in_logo])"), _ns)
tokenize_line_logo = _ns["tokenize_line_logo"]
HAND = cdli_mod.HAND


def convert_line_logo(content, smap, lang):
    """cun.cdli.convert_line, with the extra key 'logo'."""
    res = []
    for s, det, join, brk, unk, logo in tokenize_line_logo(content):
        base = {"det": det, "brk": brk, "lang": lang, "logo": logo}
        if unk:
            res.append({"oid": "", "sign": "", "read": "", "join": join, "unk": unk, **base})
            continue
        r = HAND.get(s, None)
        if r and "-" in r:
            parts = r.split("-")
            for k, p in enumerate(parts):
                oid = smap._get(p, lang)
                res.append({"oid": oid or "", "sign": smap.names.get(oid, ""), "read": p,
                            "join": join if k == len(parts) - 1 else "-", "unk": "", **base})
            continue
        if r:
            oid = smap._get(r, lang)
            read = r
        else:
            oid, read = smap.lookup(s, lang)
        if not oid and s.startswith("|") and s.endswith("|") and "." in s:
            parts = s[1:-1].split(".")
            got = [smap.lookup(p, lang) for p in parts]
            if all(o for o, _ in got):
                for k, (o, rd) in enumerate(got):
                    res.append({"oid": o, "sign": smap.names.get(o, ""), "read": rd,
                                "join": join if k == len(parts) - 1 else ".", "unk": "", **base})
                continue
        res.append({"oid": oid or "", "sign": smap.names.get(oid, "") if oid else "", "read": read, "join": join, "unk": "", **base})
    return res


def log(*a):
    print(*a, flush=True)


def coarse_lang(l):
    if not l:
        return "und"
    l = l.split("-")[0].lower()
    return {"sux": "sux", "akk": "akk", "hit": "hit", "xhu": "xhu", "elx": "elx", "xur": "xur", "uga": "uga",
            "qeb": "qeb", "qcu": "qcu", "arc": "arc", "peo": "peo", "egy": "egy", "grc": "grc"}.get(l, "oth")


WANT = None   # {(tid, project)} of train texts with a P-number, set in main before the pool starts (fork)


def load_zip(z):
    pairs = collections.defaultdict(collections.Counter)
    lpairs = collections.defaultdict(collections.Counter)
    texts = {}
    for tid, proj, cat, lines in iter_project(z):
        ns = sum(len(l["signs"]) for l in lines)
        if not ns:
            continue
        lc = collections.Counter(s["lang"] for l in lines for s in l["signs"])
        period = (cat.get("period") or "").lower()
        if period.startswith(("uruk", "proto-elamite", "proto-cuneiform")) or (lc.get("qpc", 0) + lc.get("qpe", 0)) >= 0.2 * ns:
            continue                                   # as build_corpus: no pairs from proto-cuneiform texts
        for l in lines:
            for s in l["signs"]:
                if s["oid"] and s["read"]:
                    pairs[s["read"]][s["oid"]] += 1
                    lpairs[(coarse_lang(s["lang"]), s["read"])][s["oid"]] += 1
        if (tid, proj) in WANT:
            texts[tid] = {"id": tid, "proj": proj, "n": ns, "lines": lines}
    return dict(pairs), dict(lpairs), texts


def tok(s, side):
    if s["unk"]:
        return "<" + s["unk"] + ">"
    return s["oid"] or f"<none-{side}>:{s['read']}"


def render(seq):
    """one string per sign: reading (or x / ...), {} for determinatives, + joiner (word end = space)."""
    out = []
    for s in seq:
        r = "…" if s["unk"] == "gap" else "x" if s["unk"] else (norm_read(s["read"]) or "?")
        if s["det"]:
            r = "{" + r + "}"
        if s["brk"] == "missing":
            r = "[" + r + "]"
        elif s["brk"] == "damaged":
            r += "#"
        out.append(r + (" " if s["join"] == "w" else (s["join"] or "")))
    return out


def compare(job):
    p, tid, o, c, names = job
    to = [tok(s, "o") for s in o]
    tc = [tok(s, "c") for s in c]
    sm = difflib.SequenceMatcher(None, to, tc, autojunk=False)
    rows, blocks = [], collections.Counter()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                a, b = o[i1 + k], c[j1 + k]
                rows.append((tid, i1 + k, j1 + k, a["oid"] if not a["unk"] else "<" + a["unk"] + ">",
                             norm_read(a["read"]), norm_read(b["read"]), a["read"], b["read"],
                             coarse_lang(a["lang"]), a["det"], b["det"], a["join"], b["join"], a["brk"], b["brk"],
                             bool(a["read"]) and a["read"].upper() == a["read"] and a["read"].lower() != a["read"], bool(b["logo"])))
        elif (i2 - i1) <= 6 and (j2 - j1) <= 6:
            ob = tuple((names.get(s["oid"], s["oid"]) or "", "<" + s["unk"] + ">" if s["unk"] else norm_read(s["read"]), s["join"]) for s in o[i1:i2])
            cb = tuple((names.get(s["oid"], s["oid"]) or "", "<" + s["unk"] + ">" if s["unk"] else norm_read(s["read"]), s["join"]) for s in c[j1:j2])
            lang = coarse_lang((o[i1] if i2 > i1 else o[min(i1, len(o) - 1)])["lang"])
            blocks[(lang, ob, cb)] += 1
        else:
            blocks[("big", i2 - i1, j2 - j1)] += 1
    nm = sum(1 for r in rows if not r[3].startswith("<"))
    return p, tid, rows, blocks, {"o_signs": sum(1 for s in o if not s["unk"]), "c_signs": sum(1 for s in c if not s["unk"]),
                                  "matched": nm, "o_x": sum(1 for s in o if s["unk"]), "c_x": sum(1 for s in c if s["unk"])}, render(o), render(c)


def main():
    global WANT
    meta = json.load(open(os.path.join(OUT, "cache/meta.json")))
    from noise_audit_lib import VERSION
    assert meta["version"] == VERSION
    tmeta = {(t["id"], t["src"].partition(":")[2]): t for t in meta["texts"]
             if t["split"] == "train" and t["src"].startswith("oracc") and t["cdli"]}
    WANT = set(tmeta)
    log("train Oracc texts with a P-number", len(WANT))
    zips = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "oracc", "*.zip")))
    pairs = collections.defaultdict(collections.Counter)
    lpairs = collections.defaultdict(collections.Counter)
    oracc = {}
    with Pool(NPROC) as pool:
        for k, (pz, lz, tz) in enumerate(pool.imap(load_zip, zips)):
            for rd, cnt in pz.items():
                pairs[rd].update(cnt)
            for rd, cnt in lz.items():
                lpairs[rd].update(cnt)
            oracc.update(tz)
    log("oracc texts loaded", len(oracc))
    smap = SignMap(pairs, os.path.join(ROOT, "data", "raw", "osl.asl"), lpairs)
    cover = {}
    for t in oracc.values():
        g = n = 0
        for l in t["lines"]:
            for s in l["signs"]:
                if s["unk"]:
                    continue
                g += 1
                if s["oid"]:
                    n += 1
                elif s["read"]:
                    s["oid"] = smap.lookup_oracc(s["read"], coarse_lang(s["lang"])) or ""
        cover[t["id"]] = n / g if g else 0.0
    byp = {tmeta[(t["id"], t["proj"])]["cdli"]: t for t in oracc.values()}
    jobs = []
    for t in parse_atf(os.path.join(UR, "data/raw/cdli/cdliatf_unblocked.atf"), pids=set(byp)):
        if not t.lines:
            continue
        ot = byp[t.pid]
        clang = coarse_lang((t.lang or "und").replace("other:", ""))
        o = [s for l in ot["lines"] for s in l["signs"]]
        c = [s for l in t.lines for s in convert_line_logo(l.raw, smap, clang)]
        for seq in (o, c):                             # corpus v3: illegible x -> <x>; numeral signs unified
            for sg in seq:
                if sg["unk"]:
                    continue
                if sg["oid"] == smap.oid_x or (not sg["oid"] and sg["read"] in ("x", "X")):
                    sg.update(oid="", read="", unk="x")
                else:
                    fo, fr = smap.numeral_v3(sg["read"])
                    if fo:
                        sg.update(oid=fo, read=fr)
        if o and c:
            jobs.append((t.pid, ot["id"], o, c, smap.names))
    log("texts in both sources (train)", len(jobs))
    rows, blocks, texts = [], collections.Counter(), {}
    with Pool(NPROC) as pool:
        for p, tid, r, b, st, ro, rc in pool.imap_unordered(compare, jobs, chunksize=8):
            rows += r
            blocks.update(b)
            m = tmeta[(tid, byp[p]["proj"])]
            texts[tid] = {"p": p, "proj": byp[p]["proj"], "lang": m["lang"], "period": m["period"], "genre": m["genre"],
                          "cover": round(cover[tid], 4), **st, "o": ro, "c": rc}
    df = pd.DataFrame(rows, columns=["tid", "io", "ic", "oid", "o_read", "c_read", "o_raw", "c_raw", "wlang", "o_det", "c_det",
                                     "o_join", "c_join", "o_brk", "c_brk", "o_logo", "c_logo"])
    for col in ("tid", "oid", "o_read", "c_read", "o_raw", "c_raw", "wlang", "o_det", "c_det", "o_join", "c_join", "o_brk", "c_brk"):
        df[col] = df[col].astype("category")
    df.to_pickle(os.path.join(OUT, "pairs_matched.pkl.gz"))
    pickle.dump(blocks, gzip.open(os.path.join(OUT, "pairs_blocks.pkl.gz"), "wb"))
    pickle.dump({"texts": texts, "names": smap.names}, gzip.open(os.path.join(OUT, "pairs_texts.pkl.gz"), "wb"))
    tot = collections.Counter()
    for t in texts.values():
        for k in ("o_signs", "c_signs", "matched"):
            tot[k] += t[k]
    summ = {"texts": len(texts), **tot, "matched_rows_incl_x": len(df),
            "by_project": dict(collections.Counter(t["proj"] for t in texts.values()).most_common()),
            "by_lang": dict(collections.Counter(t["lang"] for t in texts.values()))}
    json.dump(summ, open(os.path.join(OUT, "pairs_summary.json"), "w"), indent=1, ensure_ascii=False)
    log(json.dumps(summ, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
