"""Description of the duplicate-witness sets (cun.witness) -> RESULTS/witness/REPORT.md.
witness-test is only counted (sizes and composition); nothing is predicted on it.

  CUN_PROC=data/proc_v3 python scripts/witness_report.py
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cun import witness as W  # noqa: E402
from cun.paths import RESULTS  # noqa: E402

LBINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20))


def lbin(L):
    return next(f"{a}-{b}" if a != b else str(a) for a, b in LBINS if a <= L <= b)


def comp_key(r):
    return r["composites"][0] if r["composites"] else "pair:%d-%d" % (min(r["a"], r["b"]), max(r["a"], r["b"]))


def sizes(obj, lac):
    return {"lacunae": len(lac), "pairs": len({(min(r["a"], r["b"]), max(r["a"], r["b"])) for r in lac}),
            "texts": len({r["a"] for r in lac} | {r["b"] for r in lac}), "texts A": len({r["a"] for r in lac}),
            "composites": len({q for r in lac for q in r["composites"]}),
            "groups (composite, or pair without one)": len({comp_key(r) for r in lac}),
            "answer signs": sum(r["L"] for r in lac)}


def dist(lac, key, order=None):
    c = collections.Counter(key(r) for r in lac)
    ks = [k for k in (order or []) if k in c] + sorted((k for k in c if k not in (order or [])), key=lambda k: -c[k])
    return [(k, c[k], 100 * c[k] / max(len(lac), 1)) for k in ks]


def md_dist(title, rows):
    return [f"**{title}**", "", "| | n | % |", "|---|---|---|"] + [f"| {k} | {n} | {p:.1f} |" for k, n, p in rows] + [""]


def pct(v):
    v = [bool(x) for x in v]
    return f"{100 * np.mean(v):.1f}% ({int(np.sum(v))} / {len(v)})" if v else "n/a"


def describe(obj, lac, out):
    out += md_dist("Answer length (signs of B)", dist(lac, lambda r: lbin(r["L"]), [lbin(a) for a, _ in LBINS]))
    out.append("Exact lengths: " + ", ".join(f"{L}: {n}" for L, n in sorted(collections.Counter(r["L"] for r in lac).items())) + "\n")
    out += md_dist("Stratum of A", dist(lac, lambda r: str(r["stratum"])))
    out += md_dist("Genre class of A", dist(lac, lambda r: str(r["genre_class"])))
    out += md_dist("Period bin of A", dist(lac, lambda r: str(r["period_bin"])))
    out += md_dist("Language of A", dist(lac, lambda r: r["lang"]))
    out += md_dist("Position in A's line", dist(lac, lambda r: r["line_pos"], ["initial", "medial", "final", "whole", "multi"]))
    out += md_dist("The lacuna as edited in A", dist(lac, lambda r: r["kind"], ["restored", "x", "ellipsis", "mixed"]))
    out += md_dist("Regime of the answer (verbatim lookup in train with 4 / 2 signs of context)",
                   dist(lac, lambda r: r.get("regime"), ["parallel", "formulaic", "novel"]))
    out += md_dist("Pair source", dist(lac, lambda r: r["source"] + ": " + "+".join(sorted((r["a_split"], r["b_split"])))))
    sl = [r for r in lac if r["a_slots"] is not None]
    out.append(f"**Width relation.** A gives explicit slots (restored signs or x) in {pct(r['a_slots'] is not None for r in lac)} "
               f"of the lacunae; the rest contain a '...' of unknown width. Where explicit: slots of A = signs of B in "
               f"{pct(r['a_slots'] == r['L'] for r in sl)}; A has fewer in {pct(r['a_slots'] < r['L'] for r in sl)}, more in "
               f"{pct(r['a_slots'] > r['L'] for r in sl)}; |difference| <= 1 in {pct(abs(r['a_slots'] - r['L']) <= 1 for r in sl)}.")
    for k in ("restored", "x", "mixed"):
        v = [r for r in sl if r["kind"] == k]
        if v:
            out.append(f"  - {k}: equal width {pct(r['a_slots'] == r['L'] for r in v)}, mean (A slots - B signs) {np.mean([r['a_slots'] - r['L'] for r in v]):+.2f}")
    out.append("")
    rest = [r for r in lac if r["kind"] == "restored"]
    out.append(f"**Editor.** Fully restored by A's editor: {pct(r['kind'] == 'restored' for r in lac)}; partly (mixed with x / ...): "
               f"{pct(r['kind'] == 'mixed' and r['n_restored'] > 0 for r in lac)}. The editor's restoration equals B's reading "
               f"exactly (sign level) in {pct(r['editor_exact'] for r in rest)} of the fully restored lacunae; sign error rate "
               f"{100 * np.mean([r['editor_dist'] / r['L'] for r in rest]) if rest else float('nan'):.1f}%. "
               f"Same number of signs but not identical: {pct(len(r['editor']) == r['L'] and not r['editor_exact'] for r in rest)}.")
    out.append("")
    out.append("| answer length | restored lacunae | editor = B exactly | editor SER % |")
    out.append("|---|---|---|---|")
    for a, b in LBINS:
        v = [r for r in rest if a <= r["L"] <= b]
        if v:
            out.append(f"| {lbin(a)} | {len(v)} | {100 * np.mean([r['editor_exact'] for r in v]):.1f}% | {100 * np.mean([r['editor_dist'] / r['L'] for r in v]):.1f} |")
    out.append("")
    out.append(f"**Novelty.** The answer with 2 signs of context on each side occurs nowhere in the train texts: {pct(r.get('novel') for r in lac)}.")
    out.append("")
    multi = [r for r in lac if r["n_wit"] > 1]
    share = f"{100 * np.mean([r['n_agree'] / r['n_wit'] for r in multi]):.1f}%" if multi else "n/a"
    out.append(f"**Witnesses.** B is an Oracc composite text (Q-number; its clean text is attested in some exemplar): "
               f"{pct(r['b_is_composite'] for r in lac)}; A is a composite text: {pct(r['a_is_composite'] for r in lac)}; "
               f"A and B are the same object (museum number / P-number; two editions or joins): {pct(r['same_object'] for r in lac)}. "
               f"Answered by more than one witness: {pct(r['n_wit'] > 1 for r in lac)}; of these all witnesses give the same signs: "
               f"{pct(r['n_agree'] == r['n_wit'] for r in multi)}; mean share of the witnesses that give the majority reading "
               f"(the answer): {share}; median number of witnesses {int(np.median([r['n_wit'] for r in multi])) if multi else 0}.")
    out.append("")
    c = collections.Counter(comp_key(r) for r in lac)
    top = c.most_common(8)
    out.append(f"**Concentration.** {len(c)} groups (composite, or pair without one); the largest 8 hold "
               f"{100 * sum(n for _, n in top) / max(len(lac), 1):.1f}% of the lacunae: " + ", ".join(f"{k} ({n})" for k, n in top)
               + f". Median lacunae per group {int(np.median(list(c.values()))) if c else 0}. "
               f"`cun.witness.select(max_per_composite=k)` caps a group.")
    out.append("")


def quality(lac, out):
    """Signs of systematic misalignment."""
    out.append("| subset | n | editor = B (restored) | witnesses agree (>= 2 witnesses) | x-runs: slots = signs of B | x-runs: within 1 |")
    out.append("|---|---|---|---|---|---|")

    def row(name, v):
        rest = [r for r in v if r["kind"] == "restored"]
        mw = [r for r in v if r["n_wit"] > 1]
        xs = [r for r in v if r["kind"] == "x"]
        out.append(f"| {name} | {len(v)} | {pct(r['editor_exact'] for r in rest)} | {pct(r['n_agree'] == r['n_wit'] for r in mw)} | "
                   f"{pct(r['a_slots'] == r['L'] for r in xs)} | {pct(abs(r['a_slots'] - r['L']) <= 1 for r in xs)} |")
    row("everything", lac)
    row("width conflict", [r for r in lac if r["width_conflict"]])
    lac = [r for r in lac if not r["width_conflict"]]
    row("without width conflicts", lac)
    row("clean flanks (strict)", [r for r in lac if r["clean_flanks"]])
    row("flanks identified but damaged", [r for r in lac if not r["clean_flanks"]])
    for lo, hi in ((3, 3), (4, 5), (6, 9), (10, 20)):
        row(f"shorter flank {lo}-{hi}" if lo != hi else f"shorter flank {lo}", [r for r in lac if lo <= min(r["left"], r["right"]) <= hi])
    for k in ("composite", "8gram"):
        row(f"source {k}", [r for r in lac if r["source"] == k])
    for a, b in LBINS:
        row(f"answer length {lbin(a)}", [r for r in lac if a <= r["L"] <= b])
    out.append("")


def examples(obj, lac, n, seed, out):
    rng = np.random.default_rng(seed)
    for k, i in enumerate(rng.permutation(len(lac))[:n]):
        r = lac[int(i)]
        A, B = obj["arrays"][r["a"]], obj["arrays"][r["b"]]
        s, e, bs, be = r["a_s"], r["a_e"], r["b_s"], r["b_e"]
        lo, lob = max(0, s - 10), max(0, bs - 10)
        ed = W.render(obj, A[:4, s:e]) if r["kind"] == "restored" else "none (" + r["kind"] + ")"
        out.append(f"{k + 1}. **{r['a_id']}** ({r['a_split']}) from **{r['b_id']}** ({r['b_split']}); {r['source']} "
                   f"{' '.join(r['composites'][:2])}; {r['stratum']}, {r['period_bin']}, {r['lang']}; {r['line_pos']}; "
                   f"flanks {r['left']} / {r['right']}{'' if r['clean_flanks'] else ' (damaged)'}; witnesses {r['n_agree']} / {r['n_wit']}; "
                   f"{r.get('regime')}")
        out.append(f"   - A: `{W.render(obj, A[:, lo:e + 10], (s - lo, e - lo))}`")
        out.append(f"   - B: `{W.render(obj, B[:, lob:be + 10], (bs - lob, be - lob))}`")
        out.append(f"   - editor of A: `{ed}`{'' if r['editor_exact'] is None else ' = B' if r['editor_exact'] else ' (differs from B)'}; "
                   f"answer (B): `{W.render(obj, B[:4, bs:be])}`; signs: {' '.join(obj['sign_names'][int(v)] for v in r['answer'][0])}")
    out.append("")


def main():
    dev, test = W.load("dev"), W.load("test")
    out = [f"# Duplicate-witness evaluation set (corpus {dev['version']})", "",
           "Real lacunae with attested answers: witness A of a composition is broken, witness B preserves the passage "
           "(cun/witness.py; format and rules in its docstring). No witness is a train text. "
           "`strict` (the default of `cun.witness.select` / `instances`) = the three flanking signs on each side are undamaged "
           "in both witnesses, and A's explicit width (restored signs or x) does not contradict B's; `all flanks` adds lacunae whose "
           "flanking signs are identified but marked damaged (`clean_only=False`); `everything` also keeps the width conflicts "
           "(`width_conflicts=True`).", ""]
    out += ["## Sizes", "", "| set | selection | lacunae | answer signs | pairs | texts (as A) | composites | groups |", "|---|---|---|---|---|---|---|---|"]
    for obj in (dev, test):
        sels = [("strict", dict()), ("strict, at most 30 per composite", dict(max_per_composite=30)),
                ("all flanks", dict(clean_only=False)), ("everything", dict(clean_only=False, width_conflicts=True))]
        if obj is test:
            sels = [("strict, pair has a test2 text", dict()), ("all flanks, pair has a test2 text", dict(clean_only=False)),
                    ("strict, incl. excluded-excluded pairs", dict(strict=False)),
                    ("all flanks, incl. excluded-excluded pairs", dict(clean_only=False, strict=False)),
                    ("everything", dict(clean_only=False, strict=False, width_conflicts=True))]
        for nm, sel in sels:
            z = sizes(obj, W.select(obj, **sel))
            out.append(f"| {obj['name']} | {nm} | {z['lacunae']} | {z['answer signs']} | {z['pairs']} | {z['texts']} ({z['texts A']}) | "
                       f"{z['composites']} | {z['groups (composite, or pair without one)']} |")
    out.append("")
    for obj in (dev, test):
        st = obj["stats"]
        out += [f"### Pair sources, {obj['name']}", "",
                "Candidate pairs (texts of one composite; or >= 0.5 containment of rare sign 8-grams without a common composite id), "
                "by the splits of the two texts:", "",
                "| source: splits | candidate pairs | aligned (>= 6 signs) | with >= 1 lacuna (everything) | lacunae (everything) | lacunae (strict) |", "|---|---|---|---|---|---|"]
        la = collections.Counter(r["source"] + ": " + "+".join(sorted((r["a_split"], r["b_split"]))) for r in obj["lacunae"])
        ls = collections.Counter(r["source"] + ": " + "+".join(sorted((r["a_split"], r["b_split"]))) for r in obj["lacunae"] if r["clean_flanks"] and not r["width_conflict"])
        for k, v in st["candidate_pairs"].items():
            out.append(f"| {k} | {v.get('pairs', 0)} | {v.get('aligned (>= 6 signs)', 0)} | {v.get('with lacuna', 0)} | {la.get(k, 0)} | {ls.get(k, 0)} |")
        out.append("")
        if obj is dev:
            out.append("Excluded texts by the set they may serve: " + json.dumps(st["excluded_side"]) + "; by the held-out splits they are "
                       "linked to (composite, museum number or 8-gram near-duplicate): " + json.dumps(st["excluded_links"])
                       + ". Pairs not used: " + json.dumps(st["dropped_pairs"]) + ".")
            out.append("")
        out += ["Lost runs of the paired texts (a run counts once, with the furthest stage reached with any witness):", "",
                "| outcome | runs |", "|---|---|"]
        for k in W.STAGES:
            out.append(f"| {k} | {st['runs'].get(k, 0)} |")
        out.append("")
    for name, sel in (("strict", dict()), ("all flanks", dict(clean_only=False))):
        out += [f"## witness-dev, {name}", ""]
        describe(dev, W.select(dev, **sel), out)
    out += ["## witness-test (composition only; no predictions)", ""]
    for name, sel in (("strict, pair has a test2 text", dict(clean_only=True)),):
        lac = W.select(test, **sel)
        out += md_dist(f"Stratum of A ({name})", dist(lac, lambda r: str(r["stratum"])))
        out += md_dist("Period bin of A", dist(lac, lambda r: str(r["period_bin"])))
        out += md_dist("Answer length", dist(lac, lambda r: lbin(r["L"]), [lbin(a) for a, _ in LBINS]))
        out += md_dist("The lacuna as edited in A", dist(lac, lambda r: r["kind"], ["restored", "x", "ellipsis", "mixed"]))
    out += ["## Alignment quality (witness-dev)", "",
            "If alignments were systematically wrong, the editor's restorations and further witnesses would disagree with B more "
            "often where the anchors are short, and x-runs (an independent statement of A's editor about the width) would not "
            "match B's width. Rows below 'without width conflicts' exclude the conflicts.", ""]
    quality(dev["lacunae"], out)
    ex = [r for r in dev["lacunae"] if r["a_slots"] is not None]
    out += ["### Width conflicts: B has material that A did not have", "",
            f"Where A states its width (restored signs or x: {len(ex)} of {len(dev['lacunae'])} lacunae), B's answer differs from it by more "
            f"than max(1, 34%) in {pct(r['width_conflict'] for r in ex)}; any difference: {pct(r['a_slots'] != r['L'] for r in ex)}. "
            "By eye these are extra entries or lines of B (lexical lists), an added gloss or translation column, dittography, or "
            "another recension, not misalignments: the flanks match. They are left out of the default selection. "
            "Lacunae edited as '...' cannot be checked this way: some share of them must be expected to carry "
            "material A never had, more where B's answer runs over a line end of B and in lexical and scholarly texts "
            "(`select(where=lambda r: r['b_line_ends'] == 0)` is the cautious subset):", "",
            "| explicit-width lacunae | n | width conflict | any difference |", "|---|---|---|---|"]
    for nm, f in [(f"A states {lbin(a)} slots", (lambda r, a=a, b=b: a <= r["a_slots"] <= b)) for a, b in LBINS[:-1] + ((11, 40),)] + \
            [("B's answer has no line end inside", lambda r: r["b_line_ends"] == 0), ("B's answer has a line end inside", lambda r: r["b_line_ends"] > 0)] + \
            [(f"stratum {st}", (lambda r, st=st: r["stratum"] == st)) for st in ("akk-lit", "sux-lit")] + \
            [(f"genre class {g}", (lambda r, g=g: r["genre_class"] == g)) for g in sorted({str(r["genre_class"]) for r in ex})]:
        v = [r for r in ex if f(r)]
        if v:
            out.append(f"| {nm} | {len(v)} | {pct(r['width_conflict'] for r in v)} | {pct(r['a_slots'] != r['L'] for r in v)} |")
    out.append("")
    el = [r for r in dev["lacunae"] if r["a_slots"] is None]
    out.append(f"Of the lacunae with a '...' ({len(el)}), B's answer has a line end inside in {pct(r['b_line_ends'] > 0 for r in el)} "
               f"(explicit-width lacunae: {pct(r['b_line_ends'] > 0 for r in ex)}).")
    out.append("")
    bp = os.path.join(RESULTS, "witness", "baselines_dev.json")
    if os.path.exists(bp):
        res = json.load(open(bp))
        if res.get("version") == dev["version"]:
            out += ["## Baselines on witness-dev, known width", "",
                    "4gram: Witten-Bell sign 4-gram on the train texts, language-specific, beam 8 (as cun.eval3.ngram). 'Exact, any witness' "
                    "also accepts the reading of a witness that differs from the majority answer. Editor: A's "
                    "restoration, scored only where the lacuna is fully restored; biased upwards, because editors often restore "
                    "FROM the duplicate (often from B itself). SER = Levenshtein distance / answer length, averaged over lacunae.", ""]
            for sname, v in res["sets"].items():
                out += [f"### {sname}", "", "| system | n | exact % | SER % | exact, any witness % |", "|---|---|---|---|---|"]
                for sysn, t in v.items():
                    a = t["all"]
                    if a["n"]:
                        out.append(f"| {sysn} | {a['n']} | {a['exact']:.1f} | {a['ser']:.1f} | {a['exact_any']:.1f} |")
                out.append("")
                for grp in ("by_length", "by_regime", "by_stratum", "by_kind", "by_line_pos"):
                    ks = [lbin(a) for a, _ in LBINS] if grp == "by_length" else sorted(v["4gram"][grp], key=str)
                    out += [f"{grp.replace('_', ' ')}:", "", "| | n | 4gram exact % | 4gram SER % | restored n | editor exact % | editor SER % | 4gram exact % (restored) |",
                            "|---|---|---|---|---|---|---|---|"]
                    for k in ks:
                        t = v["4gram"][grp].get(k)
                        if not t or not t["n"]:
                            continue
                        e = v["editor (restored lacunae)"][grp].get(k, {"n": 0})
                        g = v["4gram (restored lacunae)"][grp].get(k, {"n": 0})
                        out.append(f"| {k} | {t['n']} | {t['exact']:.1f} | {t['ser']:.1f} | {e['n']} | "
                                   + (f"{e['exact']:.1f} | {e['ser']:.1f} | {g['exact']:.1f} |" if e["n"] else " | | |"))
                    out.append("")
    d = W.select(dev)
    multi = [r for r in d if r["n_wit"] > 1]
    out += ["## Notes and limitations", "",
            "- **No sign of systematic misalignment.** Agreement of the editor's restorations and of further witnesses with B does not "
            "fall where the flanks are short or damaged, and x-runs of A have B's width in ~95% of cases (table above); the printed "
            "examples align by eye. Two kinds of unsafe candidates are removed: anchors that recur nearby or inside B's answer "
            "(rejected at extraction), and width conflicts (flagged, off by default).",
            f"- **Witnesses genuinely differ.** Where several witnesses answer, at least one deviates from the majority reading in "
            f"{pct(r['n_agree'] < r['n_wit'] for r in multi)} of the lacunae (on average {100 * np.mean([r['n_agree'] / r['n_wit'] for r in multi]):.1f}% "
            "of the witnesses give the answer). Most deviations are orthographic (u / u₂, mid / mi₃, tu / tum, logogram against "
            "syllabic spelling): scored as errors at sign level, so exact match has a ceiling below 100% and the editor's ~93% is close "
            "to it. `alts` keeps the deviant readings ('exact, any witness'); no sense-level normalisation is applied.",
            "- **B may hold more than A lost** (an extra entry, gloss or line; another recension): detectable only where A states a "
            "width (2% conflicts, removed); not detectable for '...' lacunae, which are 61% of the strict set.",
            "- **Known width is B's width**, information the editor of A did not have for a '...'; the 'edited' instances keep the "
            "lacuna as edited for unknown-width decoding.",
            "- **Coverage.** Only compositions with several exemplars: literary strata only (akk-lit, sux-lit; no administrative or "
            "documentary stratum, so the four-strata selection metric is undefined here), mostly royal inscriptions, and a few "
            "compositions hold most lacunae (use `max_per_composite`). 90% of the pairs are two excluded texts; excluded exemplars "
            "of test1 compositions are used in witness-dev.",
            f"- **B is often an Oracc composite text** ({pct(r['b_is_composite'] for r in d)}): an editor's composite of the exemplars, "
            "attested but not one tablet.",
            "- **Editor baseline is biased upwards**: restorations are often made FROM the duplicate (frequently B itself). It is "
            "scored only on fully restored lacunae, which are shorter and more formulaic than the rest.",
            "- **Strict flanks select line-edge lacunae** edited as '...' next to undamaged signs; `clean_only=False` gives five times "
            "as many lacunae with the same quality indicators.",
            f"- **Train parallels.** {pct(r['regime'] == 'parallel' for r in d)} of the answers occur verbatim with 4 signs of context in "
            "train texts (other inscriptions of the same king, stock phrases, lexical entries), although no text sharing a composite id with a witness is in train (unlinked "
            "duplicates in train cannot be ruled out); report by regime.",
            "- Chunks are windows of <= 256 signs centred on the lacuna (starting at a line start where possible), not the fixed "
            "256-sign grid of cun.evalset.eval_chunks.", ""]
    out += ["## Examples (witness-dev, strict; random, seed 0)", "",
            "`<<< >>>` marks the lacuna in A and the answer in B; `[ ]` restored by the editor, `#` damaged, `/` line end.", ""]
    examples(dev, W.select(dev), 20, 0, out)
    out += ["## Examples where the editor's restoration differs from B (witness-dev, strict; random, seed 1)", ""]
    examples(dev, [r for r in W.select(dev) if r["editor_exact"] is False], 12, 1, out)
    os.makedirs(os.path.join(RESULTS, "witness"), exist_ok=True)
    open(os.path.join(RESULTS, "witness", "REPORT.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
