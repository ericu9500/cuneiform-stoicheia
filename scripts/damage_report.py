"""Descriptive statistics of real damage in one split of the corpus (PLAN_v3 stage 0, item 2).

  python scripts/damage_report.py [--split train] [--out results/damage]

Writes <out>/REPORT.md and <out>/report.json. The corpus is read from cun.paths.PROC (CUN_PROC); only the
texts of the given split are used."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import damage as D  # noqa: E402
from cun.consts import GAP, X_SIGN  # noqa: E402
from cun.paths import PROC  # noqa: E402

MIN_TEXTS, MIN_SIGNS = 200, 50000
NUM = re.compile(r"^\d+(/\d+)?\(")


def pct(x, d=1):
    return "-" if x != x else f"{100 * x:.{d}f}"


def num(x, d=2):
    return "-" if x != x else f"{x:.{d}f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "damage"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb") as fh:
        corpus = pickle.load(fh)
    texts = corpus["texts"]
    F = D.flatten(corpus, a.split)
    read_vocab, sign_names, version = corpus["read_vocab"], corpus["sign_names"], corpus["version"]
    del corpus["lines"]
    flat, ll, lt = F["flat"], F["ll"], F["lt"]
    sign, read, join, det, brk = flat
    nl, N = len(ll), flat.shape[1]
    off = np.r_[0, np.cumsum(ll)]
    lid = np.repeat(np.arange(nl), ll)
    ut, tix = np.unique(lt, return_inverse=True)
    T = len(ut)
    tok = D.token_status(flat)
    G0 = D.geometry(tok, ll, lt)                                   # tokens as edited
    vs, vl, _, width = D.virtual(F)
    G = D.geometry(vs, vl, lt)                                     # '...' expanded
    J = {"version": version, "split": a.split, "texts": int(T), "lines": int(nl), "signs": int(N)}

    # ---------------- groups of texts
    meta = [texts[t] for t in ut]
    lang = np.array([m["lang"] for m in meta])
    groupings = [
        ("language", lang),
        ("stratum", np.array([D.stratum_of(m) for m in meta])),
        ("genre (coarse)", np.array([D.coarse_genre(m.get("genre")) for m in meta])),
        ("period (coarse)", np.array([m.get("period_bin") or D.coarse_period(m.get("period")) for m in meta])),
    ]
    t_signs = np.bincount(tix, ll, T)
    groups = [("all", "all", np.ones(T, bool))]
    for gname, lab in groupings:
        for v in sorted(set(lab), key=lambda v: -t_signs[lab == v].sum()):
            m = lab == v
            if m.sum() >= MIN_TEXTS and t_signs[m].sum() >= MIN_SIGNS:
                groups.append((gname, v, m))

    # ---------------- 1. composition (tokens)
    isx, isgap = sign == X_SIGN, sign == GAP
    ident = ~isx & ~isgap
    cat = np.where(isgap, 4, np.where(isx, 3, np.where(brk == 2, 2, np.where(brk == 1, 1, 0))))
    tc = np.zeros((T, 5))
    np.add.at(tc, (tix[lid], cat), 1)
    t_dam = tc[:, 1:].sum(1) > 0
    t_vn = np.bincount(tix, vl, T)
    t_vlost = np.bincount(tix, G["nlost"], T)
    sums = {}
    md = []
    rows = []
    for gname, v, m in groups:
        c = tc[m].sum(0)
        n = c.sum()
        lost_tok = c[2] + c[3] + c[4]
        sums[(gname, v)] = D.summarize(D.subset(G, m[tix]))
        rows.append([f"{gname}: {v}" if gname != "all" else "all", int(m.sum()), int(n), pct(c[0] / n), pct(c[1] / n),
                     pct(c[2] / n), pct(c[3] / n), pct(c[4] / n), pct(lost_tok / n), pct(t_vlost[m].sum() / t_vn[m].sum()),
                     pct(c[2] / max(lost_tok, 1)), pct(t_dam[m].mean())])
    J["composition"] = rows
    md += ["## 1. How much is damaged\n",
           "Share of sign tokens. *clean*: identified, no flag. *damaged*: `#` / half brackets on an identified sign "
           "(partly legible; 'abraded' below). *restored*: identified sign inside `[ ]` (the editor supplied it). "
           "*x*: an illegible or lost sign of known width. *...*: a lacuna of unknown width, one token whatever its "
           "size. *lost* = restored + x + `...`. *lost, expanded*: the same with every `...` counted as its estimated "
           "number of signs (the line filled up to the median length of the text's lines without `...`; on average "
           f"{width[isgap].mean():.1f} positions per `...`, median {np.median(width[isgap]):.0f}). *restored / lost*: how "
           "much of the lost tokens the editor restored. Groups with fewer than "
           f"{MIN_TEXTS} texts or {MIN_SIGNS} signs are omitted.\n"]
    md += table(["group", "texts", "signs", "clean %", "damaged # %", "restored [ ] %", "x %", "... %", "lost %",
                 "lost, expanded %", "restored / lost %", "texts with any damage %"], rows)

    # ---------------- 2. restoration by the editor
    rl, rp, rn = G0["run_line"], G0["run_pos"], G0["run_len"]
    rs = off[rl] + rp
    cg = np.r_[0, np.cumsum(isgap)]
    cx = np.r_[0, np.cumsum(isx)]
    r_gap = cg[rs + rn] - cg[rs]
    r_x = cx[rs + rn] - cx[rs]
    ini, fin = G0["run_ini"], G0["run_fin"]
    classes = [("line-initial", ini & ~fin), ("line-final", fin & ~ini), ("whole line", ini & fin), ("medial", ~ini & ~fin),
               ("all", np.ones(len(rn), bool))]
    rows = []
    for name, m in classes:
        rows.append([name, int(m.sum()), pct(((r_gap == 0) & (r_x == 0))[m].mean()), pct(((r_gap == 0) & (r_x > 0))[m].mean()),
                     pct((r_gap > 0)[m].mean()), num(rn[m].mean()), num(rn[m & (r_gap == 0)].mean())])
    md += ["## 2. How much the editor restored\n",
           "Lost runs = maximal runs of lost tokens within a line. A run is *fully restored* when every token has a "
           "reading, *counted* when it contains x but no `...` (its width is known), *open* when it contains `...`.\n"]
    md += table(["lost runs", "n", "fully restored %", "counted (x, no ...) %", "open (with ...) %", "mean tokens",
                 "mean tokens, runs without ..."], rows)
    rows = []
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)):
        m = (rn >= lo) & (rn <= hi)
        rows.append([f"{lo}" + ("" if hi == lo else "+" if hi > 99 else f"-{hi}"), int(m.sum()),
                     pct(((r_gap == 0) & (r_x == 0))[m].mean()), pct(((r_gap == 0) & (r_x > 0))[m].mean()), pct((r_gap > 0)[m].mean())])
    md += ["By number of tokens in the run (a `...` is one token, so open runs are short in tokens whatever their real "
           "width; a long run of tokens is almost always a restoration, because an unrestored long break is written "
           "`...`, not as a row of x):\n"]
    md += table(["tokens in the run", "n", "fully restored %", "counted %", "open %"], rows)

    # ---------------- 3. length of lost runs
    s0, s1 = D.summarize(G0), sums[("all", "all")]
    rows = []
    for k in ("run length 1", "run length 2", "run length 3", "run length 4-5", "run length 6-10", "run length 11+"):
        rows.append([k.replace("run length ", ""), pct(s0[k]), pct(s1[k])])
    md += ["## 3. Length of lost runs\n",
           f"{len(rn)} lost runs; {pct((r_gap > 0).mean())}% contain a `...`, whose real width is unknown: the token "
           "counts are a lower bound, the expanded counts an estimate.\n"]
    md += table(["signs in the run", "share of runs, tokens %", "share of runs, ... expanded %"], rows)
    rows = []
    for k in ("initial", "final", "whole line", "medial", "all"):
        rows.append([k, num(s0[f"run length, {k}: mean"]), num(s0[f"run length, {k}: median"], 0), num(s0[f"run length, {k}: 90th pct"], 0),
                     num(s1[f"run length, {k}: mean"]), num(s1[f"run length, {k}: median"], 0), num(s1[f"run length, {k}: 90th pct"], 0)])
    md += table(["position of the run", "tokens: mean", "median", "90th pct", "expanded: mean", "median", "90th pct"], rows)
    rows = []
    for gname, v, m in groups:
        s = sums[(gname, v)]
        rows.append([f"{gname}: {v}" if gname != "all" else "all", num(s["run length, all: mean"]),
                     num(s["run length, all: 90th pct"], 0), num(s["run length, initial: mean"]),
                     num(s["run length, final: mean"]), num(s["run length, medial: mean"]),
                     num(s["run length, whole line: mean"])])
    md += ["By group (`...` expanded):\n"]
    md += table(["group", "mean", "90th pct", "initial: mean", "final: mean", "medial: mean", "whole line: mean"], rows)

    # ---------------- 4. position in the line
    rows = []
    for gname, v, m in groups:
        s = sums[(gname, v)]
        rows.append([f"{gname}: {v}" if gname != "all" else "all", pct(s["lines with any loss"]),
                     pct(s["lines: initial loss (not whole)"]), pct(s["lines: final loss (not whole)"]),
                     pct(s["lines: both edges lost (not whole)"]), pct(s["lines: wholly lost"]), pct(s["lines: medial loss"]),
                     pct(s["initial loss: mean share of the line"]), pct(s["final loss: mean share of the line"]),
                     pct(s["lost runs: medial (share of lost signs)"])])
    md += ["## 4. Where in the line\n",
           "Share of lines with a lost run that starts at the first sign (initial), ends at the last sign (final), "
           "covers the line (whole), or touches neither end (medial); mean extent of an edge loss as share of the "
           "line (`...` expanded); share of all lost signs that lie in medial runs.\n"]
    md += table(["group", "lines with loss %", "initial %", "final %", "both edges %", "whole line %", "medial %",
                 "initial: mean extent %", "final: mean extent %", "lost signs in medial runs %"], rows)
    rows = []
    for k in ("initial", "final", "whole line", "medial"):
        rows.append([k, pct(s1[f"lost runs: {k} (share of runs)"]), pct(s1[f"lost runs: {k} (share of lost signs)"])])
    md += table(["lost runs by position", "share of runs %", "share of lost signs %"], rows)
    whole = G["whole"]
    I, Fi = (G["a"] > 0) & ~whole, (G["b"] > 0) & ~whole
    fa, fb = G["a"] / vl, G["b"] / vl
    rows = []
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)):
        rows.append([f"{lo}" + ("" if hi == lo else "+" if hi > 99 else f"-{hi}") + " signs",
                     pct(((G["a"][I] >= lo) & (G["a"][I] <= hi)).mean()), pct(((G["b"][Fi] >= lo) & (G["b"][Fi] <= hi)).mean())])
    for lo, hi in ((0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        rows.append([f"{lo:.2f}-{hi:.2f} of the line".replace("0.00-", "up to "),
                     pct(((fa[I] > lo) & (fa[I] <= hi)).mean()), pct(((fb[Fi] > lo) & (fb[Fi] <= hi)).mean())])
    md += ["Extent of an edge loss: where a line-initial loss ends (counted from the line start) and where a "
           "line-final loss starts (counted from the line end), as sign index and as share of the line "
           "(`...` expanded).\n"]
    md += table(["extent", "initial losses %", "final losses %"], rows)
    medr = ~G["run_ini"] & ~G["run_fin"]
    c = (G["run_pos"][medr] + 0.5 * G["run_len"][medr]) / vl[G["run_line"][medr]]
    md += ["Centre of medial runs (share of the line): " + ", ".join(
        f"{lo:.1f}-{lo + 0.2:.1f}: {pct(((c >= lo) & (c < lo + 0.2 + 1e-9)).mean())}%" for lo in (0.0, 0.2, 0.4, 0.6, 0.8)) + ".\n"]
    # abraded signs
    Ga = D.geometry(np.where(tok == D.ABRADED, D.LOST, D.CLEAN).astype(np.int8), ll, lt)
    sa = D.summarize(Ga)
    rows = [[k, pct(sa[f"lost runs: {k} (share of runs)"]), pct(sa[f"lost runs: {k} (share of lost signs)"]),
             num(sa[f"run length, {k}: mean"])] for k in ("initial", "final", "whole line", "medial")]
    md += ["### Damaged (#) signs\n",
           f"Runs of damaged-but-legible signs are short (mean {num(s0['abraded runs: mean length'])} signs; "
           f"{pct(s0['abraded runs: length 1'])}% are single signs) and {pct(s0['abraded runs touching a lost run'])}% of them "
           f"touch a lost run of the same line (the half-preserved signs at the edge of a break). {pct(s0['lines with abraded signs'])}% "
           f"of the lines have at least one; {pct(s0['lines with loss and abraded signs'])}% have both a loss and a damaged sign. "
           "Position of the runs of damaged signs in the line (same definitions as for lost runs):\n"]
    md += table(["runs of # signs by position", "share of runs %", "share of # signs %", "mean length"], rows)

    # ---------------- 5. vertical persistence
    rows = []
    for gname, v, m in groups:
        s = sums[(gname, v)]
        rows.append([f"{gname}: {v}" if gname != "all" else "all"] +
                    [pct(s[f"P({x} next | {y}{x})"]) for x in ("initial", "final", "whole", "medial", "abraded") for y in ("", "not ")] +
                    [num(s["corr of initial extent, consecutive lines (both lost)"]), num(s["corr of final extent, consecutive lines (both lost)"]),
                     num(s["corr of lost share of the line, consecutive lines"])])
    md += ["## 5. Persistence over consecutive lines\n",
           "Probability that line i+1 has a loss of a kind given that line i has one (yes) or has none (no); "
           "correlation of the extent (share of the line, `...` expanded) between consecutive lines that both have "
           "the edge loss; correlation of the lost share of the line between all consecutive lines. A text's lines "
           "are consecutive across surfaces (the arrays do not mark them).\n"]
    md += table(["group", "initial: yes %", "no %", "final: yes %", "no %", "whole: yes %", "no %", "medial: yes %", "no %",
                 "# signs: yes %", "no %", "r extent initial", "r extent final", "r lost share"], rows)
    same = G["same"]
    for name, x in (("line-initial loss", I), ("line-final loss", Fi), ("wholly lost lines", whole)):
        bl = D.block_lengths(x, same)
        md.append(f"Runs of consecutive lines with {name}: {len(bl)} runs, mean {bl.mean():.2f} lines; " + ", ".join(
            f"{lab}: {pct(((bl >= lo) & (bl <= hi)).mean())}%" for lab, lo, hi in
            (("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6-10", 6, 10), ("11+", 11, 10 ** 9))) + ".\n")
    both = same[1:] & I[:-1] & I[1:]
    dlt = (fa[1:] - fa[:-1])[both]
    md.append(f"Between two consecutive lines with an initial loss the extent changes by {np.abs(dlt).mean():.3f} of the line on "
              f"average; it is unchanged (within 0.05) in {pct((np.abs(dlt) <= 0.05).mean())}% of the pairs, grows by more than "
              f"0.2 in {pct((dlt > 0.2).mean())}% and shrinks by more than 0.2 in {pct((dlt < -0.2).mean())}%.\n")

    # ---------------- 6. position in the text
    start = np.flatnonzero(~same)
    cnt = np.diff(np.r_[start, nl])
    idx = np.arange(nl) - np.repeat(start, cnt)
    tl = np.repeat(cnt, cnt)
    ridx = tl - 1 - idx
    lf = G["nlost"] / vl
    af = G["nabr"] / vl
    long_ = tl >= 12
    rows = []

    def prow(name, s):
        rows.append([name, int(s.sum()), pct(lf[s].mean()), pct(whole[s].mean()), pct(I[s].mean()), pct(Fi[s].mean()), pct(af[s].mean())])

    for k in range(4):
        prow(f"line {k + 1}", long_ & (idx == k))
    prow("lines 5-6", long_ & (idx >= 4) & (idx <= 5))
    prow("middle (more than 6 lines from either end)", long_ & (idx >= 6) & (ridx >= 6))
    prow("5th and 6th line from the end", long_ & (ridx >= 4) & (ridx <= 5))
    for k in range(3, -1, -1):
        prow("last line" if k == 0 else f"{k + 1}th line from the end".replace("2th", "2nd").replace("3th", "3rd"), long_ & (ridx == k))
    md += ["## 6. Position in the text\n",
           "The corpus arrays have no line labels: `scripts/build_corpus.py` holds a label per line (`o 1`, `r 3'`) only "
           "in memory while building and does not store it, so obverse / reverse / edge boundaries and the primes that "
           "mark lines counted after a break are not available. Position is therefore measured from the start and the end "
           f"of the text only (texts of at least 12 lines: {int((cnt >= 12).sum())} texts). Mean lost share of the line "
           "(`...` expanded) and share of lines by kind of loss:\n"]
    md += table(["line", "lines", "lost share %", "wholly lost %", "initial loss %", "final loss %", "# share %"], rows)
    rel = (idx + 0.5) / tl
    md += ["By relative position in the text (tenths; same texts): lost share % " + ", ".join(
        pct(lf[long_ & (rel >= d / 10) & (rel < (d + 1) / 10)].mean()) for d in range(10)) +
        ". The rise in the middle is where the end of the obverse and the start of the reverse fall in most tablets.\n"]

    # ---------------- 7. sign classes
    num_read = np.array([bool(NUM.match(r)) for r in read_vocab])
    num_sign = np.array([bool(NUM.match(s)) for s in sign_names])
    is_num = num_read[read] | num_sign[sign]
    is_det = det > 0
    last_tok = np.zeros(N, bool)
    last_tok[off[1:] - 1] = True
    first_tok = np.zeros(N, bool)
    first_tok[off[:-1]] = True
    wend = (join == 0) | last_tok
    wstart = np.r_[True, wend[:-1]] | first_tok
    wid = np.cumsum(wstart) - 1
    w_det = (np.bincount(wid, is_det) > 0)[wid]
    cls = np.where(is_num, 0, np.where(is_det, 1, np.where(w_det, 2, 3)))
    cnames = ["numeral", "determinative", "other sign of a word with a determinative", "other"]
    st3 = np.where(brk == 2, 2, np.where(brk == 1, 1, 0))          # of identified signs: clean / damaged / restored
    md += ["## 7. Which signs are damaged\n",
           "Only identified signs can be classified, so 'lost' is limited here to signs the editor restored; x and `...` "
           "have no class. Numerals: reading or sign name of the form n(...). Determinatives: the det plane. Words with "
           "a determinative (names of persons, gods and places, and classified nouns) are the nearest available proxy "
           "for names; date formulae and colophons are not marked in the corpus. Rate ratio = rate in the class / rate "
           "among 'other' signs. Within-text ratio = observed number of damaged (restored) signs of the class / the number "
           "expected if every sign of a text had that text's own rate: it removes the difference between well and badly "
           "preserved texts (numerals are concentrated in administrative tablets, which are the best preserved). A lost "
           "numeral or name is harder to restore than a formula, so the restored column understates their real loss; "
           "the # column does not depend on restorability.\n"]
    tsign = tix[lid]
    rows = []
    for lg in ("all", "sux", "akk"):
        m = ident if lg == "all" else ident & (lang == lg)[tix][lid]
        tab = np.zeros((4, 3))
        np.add.at(tab, (cls[m], st3[m]), 1)
        rate = tab / tab.sum(1, keepdims=True)
        t_n = np.maximum(np.bincount(tsign[m], minlength=T), 1)
        t_r = [np.bincount(tsign[m & (st3 == j)], minlength=T) / t_n for j in (1, 2)]
        for k in range(4):
            n_ct = np.bincount(tsign[m & (cls == k)], minlength=T)
            wt = [tab[k, j + 1] / max((n_ct * t_r[j]).sum(), 1e-9) for j in (0, 1)]
            rows.append([lg, cnames[k], int(tab[k].sum()), pct(tab[k].sum() / tab.sum()), pct(rate[k, 1]), num(rate[k, 1] / rate[3, 1]),
                         num(wt[0]), pct(rate[k, 2]), num(rate[k, 2] / rate[3, 2]), num(wt[1])])
    md += table(["language", "class", "identified signs", "share of signs %", "damaged # %", "rate ratio", "within-text ratio",
                 "restored [ ] %", "rate ratio", "within-text ratio"], rows)
    # word position
    wpos = np.where(wstart & wend, 0, np.where(wstart, 1, np.where(wend, 3, 2)))
    wn = ["one-sign word", "first sign of a word", "inner sign", "last sign of a word"]
    rows = []
    for lg in ("all", "sux", "akk"):
        m = ident if lg == "all" else ident & (lang == lg)[tix][lid]
        tab = np.zeros((4, 3))
        np.add.at(tab, (wpos[m], st3[m]), 1)
        rate = tab / tab.sum(1, keepdims=True)
        for k in range(4):
            rows.append([lg, wn[k], int(tab[k].sum()), pct(rate[k, 1]), pct(rate[k, 2])])
    md += ["Position in the word (identified signs):\n"]
    md += table(["language", "position", "signs", "damaged # %", "restored [ ] %"], rows)
    re_ = rs + rn - 1
    inner_s = rp > 0
    inner_e = rp + rn < ll[rl]
    base_s = wstart[~first_tok].mean()
    base_e = wend[~last_tok].mean()
    al_, ap_, an_ = G0["abr_line"], G0["abr_pos"], G0["abr_len"]
    as_ = off[al_] + ap_
    md.append("Do breaks follow word boundaries? Of the lost runs that start inside the line, "
              f"{pct(wstart[rs[inner_s]].mean())}% start at the first sign of a word (of all signs inside lines, {pct(base_s)}% are word-initial); "
              f"of those that end inside the line, {pct(wend[re_[inner_e]].mean())}% end at the last sign of a word (baseline {pct(base_e)}%). "
              f"For runs of damaged (#) signs the figures are {pct(wstart[as_[ap_ > 0]].mean())}% and "
              f"{pct(wend[(as_ + an_ - 1)[ap_ + an_ < ll[al_]]].mean())}%. Editors mark and restore word by word (and x and `...` are "
              "written as words of their own), so the alignment of lost runs with words is in part an artefact of the edition, "
              "not of the clay.\n")
    md.insert(0, f"# Geometry of real damage\n\nCorpus {version} (`{os.path.relpath(PROC, ROOT)}`), split `{a.split}`: {T} texts, {nl} lines, "
                 f"{N} sign tokens. Produced by `scripts/damage_report.py`; definitions in `cun/damage.py`. A sign position is "
                 "*lost* if it is inside `[ ]` (brk = 2; restored by the editor or not) or is an x or `...` token, *damaged* "
                 "(abraded) if it is an identified sign flagged `#` (brk = 1). All geometry is per line (the corpus has no "
                 "surface coordinates): where in the line a lost run starts and ends, and how that persists over consecutive "
                 "lines. Sampler and its validation: `VALIDATION.md`.\n")
    md += ["## 8. What the line-level view cannot give\n",
           "- No surface coordinates: sign widths, line indentation and the physical width of a break are unknown; an extent "
           "is a number of signs or a share of the line's signs.\n"
           "- No surface boundaries or line labels in the arrays (section 6): top and bottom edges of obverse and reverse "
           "cannot be told from the middle of a text, and columns of multi-column tablets follow each other as lines.\n"
           "- Lines that are entirely gone are mostly not in the transliteration at all (`$ 3 lines broken` is not a line), so "
           "the share of wholly lost lines and the length of their blocks are lower bounds.\n"
           "- The width of `...` is an estimate; the editor's choice between `[x x]`, `[...]` and a restoration depends on "
           "the edition and on how predictable the text is, so 'restored' is not a random sample of what is lost.\n"
           "- Names, date formulae and colophons are not marked; only numerals and determinatives can be told apart, and only "
           "among identified signs.\n"]
    with open(os.path.join(a.out, "REPORT.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    J["summary_tokens"] = {k: (None if isinstance(v, float) and v != v else float(v)) for k, v in s0.items()}
    J["summary_by_group"] = {f"{g}: {v}": {k: (None if isinstance(x, float) and x != x else float(x)) for k, x in s.items()}
                             for (g, v), s in sums.items()}
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump(J, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
