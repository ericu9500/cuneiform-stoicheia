"""Accuracy of the CDLI -> OSL sign converter (cun.cdli.convert_line), measured on texts present in both
CDLI and Oracc. Oracc's sign ids are curated and serve as the reference.

Sample: ~3,000 P-numbers present in both sources whose Oracc version carries an oid for >= 95% of its
graphemes (x / ellipsis excluded). The SignMap is built exactly as in scripts/build_corpus.py.

Per text, the Oracc and converted CDLI sign sequences (lines concatenated in order) are aligned twice with
difflib.SequenceMatcher (autojunk off):
  * on oid tokens (x / ellipsis as their own tokens, unmapped CDLI signs as a token that never matches) ->
    oid recall of Oracc signs, substitution pairs (equal-length replace blocks, paired position by position),
    and reading / join / determinative agreement on oid-matched signs;
  * on normalised readings (norm_read, with ḫ folded to h) -> edition similarity (reading-sequence ratio) and, on positions
    whose readings agree, oid disagreements (these cannot be edition differences).
A text counts as near-identical when its reading-sequence ratio is >= 0.95 and the sign counts differ by
<= 5%; numbers are reported for all texts and for near-identical texts, split by language (sux / akk).

Output: logs/check_cdli_conversion.json and logs/check_cdli_conversion.txt. Reads only data/raw/.
"""
from __future__ import annotations

import collections
import difflib
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UR = "/nobackup/proj/disk/dionysus/shared/UR"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(UR, "src"))
from cun.cdli import convert_line  # noqa: E402
from cun.oracc import iter_project  # noqa: E402
from cun.signmap import SignMap, norm_read  # noqa: E402
from ur.atf import parse_atf  # noqa: E402

SEED = 20261119
N_SAMPLE = 3000
MIN_OID_COVER = 0.95
NEAR_RATIO = 0.95
NEAR_LEN = 0.05
NPROC = 8
LOGS = os.path.join(ROOT, "logs")


def log(*a):
    print(*a, flush=True)


def coarse_lang(l):
    if not l:
        return "und"
    l = l.split("-")[0].lower()
    return {"sux": "sux", "akk": "akk", "hit": "hit", "xhu": "xhu", "elx": "elx", "xur": "xur", "uga": "uga",
            "qeb": "qeb", "qcu": "qcu", "arc": "arc", "peo": "peo", "egy": "egy", "grc": "grc"}.get(l, "oth")


def load_zip(z):
    """-> (pairs reading->Counter(oid), {tid: text} for texts with a P-number)."""
    pairs = collections.defaultdict(collections.Counter)
    lpairs = collections.defaultdict(collections.Counter)
    texts = {}
    for tid, proj, cat, lines in iter_project(z):
        ns = sum(len(l["signs"]) for l in lines)
        if not ns:
            continue
        for l in lines:
            for s in l["signs"]:
                if s["oid"] and s["read"]:
                    pairs[s["read"]][s["oid"]] += 1
                    lpairs[(coarse_lang(s["lang"]), s["read"])][s["oid"]] += 1
        p = cat.get("cdli_id") or (tid if tid.startswith("P") else None)
        if not p:
            continue
        g = [s for l in lines for s in l["signs"] if not s["unk"]]
        cover = sum(1 for s in g if s["oid"]) / len(g) if g else 0.0
        langs = collections.Counter(s["lang"] for l in lines for s in l["signs"] if s["lang"])
        texts[tid] = {"id": tid, "proj": proj, "n": ns, "lines": lines, "cdli": p, "cover": cover,
                      "lang": coarse_lang(langs.most_common(1)[0][0] if langs else "")}
    return dict(pairs), dict(lpairs), texts


def tok_oid(s, side):
    if s["unk"]:
        return "<" + s["unk"] + ">"
    if s["oid"]:
        return s["oid"]
    return f"<none-{side}>:{s['read']}"          # never matches the other side


def loose(r):
    """norm_read plus h for ḫ (Oracc's Akkadian projects write ḫ, the converter h): a reading-convention
    difference, not an edition or converter difference."""
    return (norm_read(r) or "").replace("ḫ", "h")


def tok_read(s):
    if s["unk"]:
        return "<" + s["unk"] + ">"
    return loose(s["read"]) or "<empty>"


def jcls(j):
    return "w" if j == "w" else ("in" if j else "none")


def compare(o, c, names):
    """o, c: flat sign lists (Oracc, CDLI). -> per-text stats dict."""
    r = collections.Counter()
    subs = collections.Counter()           # (oracc name, cdli name, cdli read, oracc read)
    sameread_diffoid = collections.Counter()
    read_dis = collections.Counter()       # on oid-matched: (oracc read, cdli read)
    join_dis = collections.Counter()
    det_dis = collections.Counter()
    unmapped = collections.Counter()
    on = [s for s in o if not s["unk"]]
    cn = [s for s in c if not s["unk"]]
    r["o_signs"] = len(on)
    r["c_signs"] = len(cn)
    r["c_unmapped"] = sum(1 for s in cn if not s["oid"])
    for s in cn:
        if not s["oid"]:
            unmapped[s["read"]] += 1
    to = [tok_oid(s, "o") for s in o]
    tc = [tok_oid(s, "c") for s in c]
    sm = difflib.SequenceMatcher(None, to, tc, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for a, b in zip(o[i1:i2], c[j1:j2]):
                if a["unk"]:
                    continue
                r["oid_match"] += 1
                ra, rb = norm_read(a["read"]), norm_read(b["read"])
                r["read_eq"] += ra == rb
                r["read_eq_loose"] += loose(a["read"]) == loose(b["read"])
                if ra != rb:
                    read_dis[(ra, rb)] += 1
                r["join_eq"] += a["join"] == b["join"]
                r["joinw_eq"] += (a["join"] == "w") == (b["join"] == "w")
                if jcls(a["join"]) != jcls(b["join"]) or a["join"] != b["join"]:
                    join_dis[(a["join"] or "''", b["join"] or "''")] += 1
                da, db = a["det"].split(":")[0], b["det"].split(":")[0]
                r["det_eq"] += da == db
                r["isdet_eq"] += bool(da) == bool(db)
                if da != db:
                    det_dis[(da or "-", db or "-", a["det"].split(":")[-1] if a["det"] else "")] += 1
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            for a, b in zip(o[i1:i2], c[j1:j2]):
                if a["unk"] or b["unk"]:
                    continue
                if not b["oid"]:
                    continue
                r["sub_1to1"] += 1
                subs[(names.get(a["oid"], a["sign"] or a["oid"]), names.get(b["oid"], b["oid"]),
                      b["read"], norm_read(a["read"]))] += 1
    # reading alignment
    ro = [tok_read(s) for s in o]
    rc = [tok_read(s) for s in c]
    sm2 = difflib.SequenceMatcher(None, ro, rc, autojunk=False)
    m = 0
    for tag, i1, i2, j1, j2 in sm2.get_opcodes():
        if tag != "equal":
            continue
        m += i2 - i1
        for a, b in zip(o[i1:i2], c[j1:j2]):
            if a["unk"]:
                continue
            r["read_aligned"] += 1
            if not b["oid"]:
                r["read_aligned_unmapped"] += 1
            elif a["oid"] != b["oid"]:
                r["read_aligned_diffoid"] += 1
                sameread_diffoid[(loose(a["read"]), names.get(a["oid"], a["oid"]),
                                  names.get(b["oid"], b["oid"]))] += 1
    ratio = 2 * m / (len(ro) + len(rc)) if (ro or rc) else 1.0
    lendiff = abs(len(on) - len(cn)) / max(1, len(on))
    return {"r": r, "subs": subs, "srdo": sameread_diffoid, "read_dis": read_dis, "join_dis": join_dis,
            "det_dis": det_dis, "unmapped": unmapped, "ratio": ratio, "lendiff": lendiff,
            "near": ratio >= NEAR_RATIO and lendiff <= NEAR_LEN}


def rates(r):
    d = lambda a, b: round(r[a] / r[b], 4) if r[b] else None  # noqa: E731
    return {"texts": r["texts"], "oracc_signs": r["o_signs"], "cdli_signs": r["c_signs"],
            "oid_recall_of_oracc_signs": d("oid_match", "o_signs"),
            "oid_precision_of_cdli_signs": d("oid_match", "c_signs"),
            "cdli_unmapped_share": d("c_unmapped", "c_signs"),
            "reading_agreement_on_matched": d("read_eq", "oid_match"),
            "reading_agreement_on_matched_h_folded": d("read_eq_loose", "oid_match"),
            "join_exact_agreement_on_matched": d("join_eq", "oid_match"),
            "word_boundary_agreement_on_matched": d("joinw_eq", "oid_match"),
            "det_position_agreement_on_matched": d("det_eq", "oid_match"),
            "det_flag_agreement_on_matched": d("isdet_eq", "oid_match"),
            "one_to_one_substitutions_per_oracc_sign": d("sub_1to1", "o_signs"),
            "reading_aligned_share_of_oracc_signs": d("read_aligned", "o_signs"),
            "oid_differs_where_reading_agrees": d("read_aligned_diffoid", "read_aligned"),
            "unmapped_where_reading_agrees": d("read_aligned_unmapped", "read_aligned")}


def main():
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
            for tid, t in tz.items():
                if tid not in oracc or t["n"] > oracc[tid]["n"]:
                    oracc[tid] = t
            if k % 20 == 0:
                log("zip", k, len(zips), len(oracc))
    log("oracc texts with P-number", len(oracc))
    smap = SignMap(pairs, os.path.join(ROOT, "data", "raw", "osl.asl"), lpairs)
    # one Oracc version per P-number (most signs), as the corpus effectively does
    byp = {}
    for t in oracc.values():
        if t["cdli"] not in byp or t["n"] > byp[t["cdli"]]["n"]:
            byp[t["cdli"]] = t
    elig = {p: t for p, t in byp.items() if t["cover"] >= MIN_OID_COVER}
    log("P-numbers in Oracc", len(byp), "with oid cover >= 0.95", len(elig))
    for t in elig.values():                         # same treatment as build_corpus for the few oid-less
        for l in t["lines"]:
            for s in l["signs"]:
                if not s["oid"] and not s["unk"] and s["read"]:
                    s["oid"] = smap.lookup_oracc(s["read"]) or ""
    cdli = {}
    for t in parse_atf(os.path.join(UR, "data/raw/cdli/cdliatf_unblocked.atf"), pids=set(elig)):
        if t.lines:
            cdli[t.pid] = t
    both = sorted(set(cdli) & set(elig))
    log("in both, eligible", len(both))
    rnd = random.Random(SEED)
    sample = rnd.sample(both, min(N_SAMPLE, len(both)))

    agg = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    cnts = {k: collections.defaultdict(collections.Counter) for k in
            ("subs", "srdo", "read_dis", "join_dis", "det_dis", "unmapped", "subs_near")}
    ratios = collections.defaultdict(list)
    proj_c = collections.Counter()
    examples = collections.defaultdict(list)
    per_text = []
    smap.miss.clear()
    for p in sample:
        ot = elig[p]
        ct = cdli[p]
        lang = ot["lang"] if ot["lang"] in ("sux", "akk") else "other"
        clang = coarse_lang((ct.lang or "und").replace("other:", ""))
        o = [s for l in ot["lines"] for s in l["signs"]]
        c = [s for l in ct.lines for s in convert_line(l.raw, smap, clang)]
        if not o or not c:
            continue
        res = compare(o, c, smap.names)
        proj_c[ot["proj"]] += 1
        for grp in (lang, "all"):
            for sub in ("all", "near" if res["near"] else "far"):
                a = agg[grp][sub]
                a.update(res["r"])
                a["texts"] += 1
            ratios[grp].append(res["ratio"])
            for k in cnts:
                if k != "subs_near":
                    cnts[k][grp].update(res[k])
            if res["near"]:
                cnts["subs_near"][grp].update(res["subs"])
        for key in list(res["subs"])[:3] + list(res["srdo"])[:3]:
            if len(examples[str(key)]) < 3:
                examples[str(key)].append(p)
        per_text.append({"p": p, "oracc_id": ot["id"], "proj": ot["proj"], "lang": lang, "ratio": round(res["ratio"], 4),
                         "lendiff": round(res["lendiff"], 4), "near": res["near"],
                         "oid_recall": round(res["r"]["oid_match"] / max(1, res["r"]["o_signs"]), 4)})

    def top(counter, n=60):
        return [[list(k) if isinstance(k, tuple) else k, v] for k, v in counter.most_common(n)]

    out = {"settings": {"seed": SEED, "n_sample": len(sample), "min_oid_cover": MIN_OID_COVER,
                        "near_ratio": NEAR_RATIO, "near_lendiff": NEAR_LEN,
                        "eligible_in_both": len(both), "oracc_p_numbers": len(byp), "oracc_eligible": len(elig)},
           "sample_projects": dict(proj_c.most_common()),
           "rates": {g: {s: rates(a) for s, a in subs.items()} for g, subs in agg.items()},
           "near_identical_share": {g: round(sum(1 for t in per_text if (g == "all" or t["lang"] == g) and t["near"]) /
                                             max(1, sum(1 for t in per_text if g == "all" or t["lang"] == g)), 4)
                                    for g in agg},
           "ratio_quantiles": {g: [round(sorted(v)[int(q * (len(v) - 1))], 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
                               for g, v in ratios.items()},
           "top_substitutions_all": {g: top(c) for g, c in cnts["subs"].items()},
           "top_substitutions_near": {g: top(c) for g, c in cnts["subs_near"].items()},
           "same_reading_different_oid": {g: top(c) for g, c in cnts["srdo"].items()},
           "reading_disagreements_on_matched": {g: top(c) for g, c in cnts["read_dis"].items()},
           "join_disagreements_on_matched": {g: top(c, 30) for g, c in cnts["join_dis"].items()},
           "det_disagreements_on_matched": {g: top(c, 30) for g, c in cnts["det_dis"].items()},
           "unmapped_cdli_readings": {g: top(c, 80) for g, c in cnts["unmapped"].items()},
           "examples": examples, "per_text": per_text}
    os.makedirs(LOGS, exist_ok=True)
    json.dump(out, open(os.path.join(LOGS, "check_cdli_conversion.json"), "w"), indent=1, ensure_ascii=False)

    L = []
    st = out["settings"]
    L.append(f"CDLI -> OSL converter check: {st['n_sample']} texts sampled from {st['eligible_in_both']} P-numbers "
             f"in both sources with Oracc oid cover >= {MIN_OID_COVER}")
    L.append(f"near-identical = reading-sequence ratio >= {NEAR_RATIO} and sign-count difference <= {NEAR_LEN:.0%}")
    L.append("sample by Oracc project: " + ", ".join(f"{k} {v}" for k, v in proj_c.most_common(15)))
    for g in ("all", "sux", "akk", "other"):
        if g not in agg:
            continue
        L.append("")
        L.append(f"== {g}: near-identical texts {out['near_identical_share'][g]:.1%}; reading-ratio quantiles "
                 f"(10/25/50/75/90) {out['ratio_quantiles'][g]}")
        for s in ("all", "near", "far"):
            if s in out["rates"][g]:
                L.append(f"  [{s}] " + "; ".join(f"{k}={v}" for k, v in out["rates"][g][s].items()))
    for g in ("sux", "akk"):
        if g not in agg:
            continue
        L.append("")
        L.append(f"== {g}: top substitutions in near-identical texts (Oracc sign | CDLI sign | CDLI read | Oracc read : n)")
        for k, v in out["top_substitutions_near"].get(g, [])[:30]:
            L.append(f"  {k[0]} | {k[1]} | {k[2]} | {k[3]} : {v}")
        L.append(f"== {g}: same reading, different oid (read | Oracc sign | CDLI sign : n)")
        for k, v in out["same_reading_different_oid"][g][:25]:
            L.append(f"  {k[0]} | {k[1]} | {k[2]} : {v}")
        L.append(f"== {g}: unmapped CDLI readings")
        L.append("  " + ", ".join(f"{k} {v}" for k, v in out["unmapped_cdli_readings"][g][:50]))
        L.append(f"== {g}: reading disagreements on oid-matched signs (Oracc | CDLI : n)")
        L.append("  " + ", ".join(f"{k[0]}|{k[1]} {v}" for k, v in out["reading_disagreements_on_matched"][g][:30]))
        L.append(f"== {g}: join disagreements on matched (Oracc | CDLI : n)")
        L.append("  " + ", ".join(f"{k[0]}|{k[1]} {v}" for k, v in out["join_disagreements_on_matched"][g][:15]))
        L.append(f"== {g}: det disagreements on matched (Oracc | CDLI | Oracc kind : n)")
        L.append("  " + ", ".join(f"{k[0]}|{k[1]}|{k[2]} {v}" for k, v in out["det_disagreements_on_matched"][g][:15]))
    open(os.path.join(LOGS, "check_cdli_conversion.txt"), "w").write("\n".join(L) + "\n")
    log("\n".join(L))


if __name__ == "__main__":
    main()
