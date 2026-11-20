"""Validation of the damage sampler (cun.damage): synthetic masks on the line structures of random
chunks of a split against the real damage (PLAN_v3 stage 0, item 2).

  python scripts/damage_validate.py [--split train] [--n 20000] [--stats PATH] [--out results/damage]

Writes <out>/VALIDATION.md (side-by-side statistics, timing, examples as ASCII art) and
<out>/validation.json. The corpus is read from cun.paths.PROC (CUN_PROC)."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import damage as D  # noqa: E402
from cun.devset import CHUNK  # noqa: E402
from cun.paths import PROC  # noqa: E402


def chunk_structures(F, pick_n, rng):
    """Random chunks (lines of a text concatenated, cut every CHUNK signs, as in cun.devset): for each,
    (text, line lengths within the chunk, slice of the flat arrays, starts at text start, ends at text end)."""
    ll, lt = F["ll"], F["lt"]
    off = np.r_[0, np.cumsum(ll)]
    tb = np.r_[0, np.flatnonzero(lt[1:] != lt[:-1]) + 1, len(lt)]       # line index where each text begins
    t_sign0, t_sign1 = off[tb[:-1]], off[tb[1:]]
    nch = -(-(t_sign1 - t_sign0) // CHUNK)
    cum = np.r_[0, np.cumsum(nch)]
    out = []
    for c in rng.permutation(cum[-1]):
        k = int(np.searchsorted(cum, c, "right") - 1)
        s = int(t_sign0[k] + (c - cum[k]) * CHUNK)
        e = int(min(s + CHUNK, t_sign1[k]))
        if e - s < 2:
            continue
        l0 = int(np.searchsorted(off, s, "right") - 1)
        l1 = int(np.searchsorted(off, e, "left"))
        cuts = np.clip(off[l0:l1 + 1], s, e)
        out.append((int(lt[l0]), np.diff(cuts), s, e, s == t_sign0[k], e == t_sign1[k]))
        if len(out) == pick_n:
            break
    return out


def fmt(x):
    if isinstance(x, (int, np.integer)):
        return str(x)
    if x != x:
        return "-"
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "damage"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = D.load_stats(a.stats)
    with gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb") as fh:
        corpus = pickle.load(fh)
    if stats.get("version") != corpus["version"]:
        raise SystemExit(f"damage statistics are of corpus {stats.get('version')}, corpus is {corpus['version']}: refit")
    texts = corpus["texts"]
    F = D.flatten(corpus, a.split)
    del corpus["lines"]
    rng = np.random.default_rng(a.seed)
    chunks = chunk_structures(F, a.n, rng)
    tok = D.token_status(F["flat"])
    lens = [c[1] for c in chunks]
    ll_all = np.concatenate(lens)
    grp = np.repeat(np.arange(len(chunks)), [len(x) for x in lens])
    real = D.summarize(D.geometry(np.concatenate([tok[c[2]:c[3]] for c in chunks]), ll_all, grp))

    def synth(**kw):
        st = []
        req, got = [], []
        for ti, l, s, e, ts, te in chunks:
            k = dict(kw)
            if "target_rate" in k:
                k["target_rate"] = float(np.clip(rng.beta(2, 4), 0.05, 0.8))     # the training noise rate (C8)
                req.append(k["target_rate"])
            m, ab = D.sample_mask(rng, l, stats, stratum=D.stratum_of(texts[ti]), text_start=ts, text_end=te, **k)
            got.append(int(m.sum()))
            if "target_rate" in k:
                req[-1] = max(1, int(round(req[-1] * len(m))))
            st.append(np.where(ab, D.ABRADED, np.where(m, D.LOST, D.CLEAN)).astype(np.int8))
        return D.summarize(D.geometry(np.concatenate(st), ll_all, grp)), np.array(req), np.array(got)

    syn, _, _ = synth(allow_clean=True)
    syn_t, req, got = synth(target_rate=None)
    cols = [("real, all of the split, '...' expanded (fitted to)", stats["summary"]),
            (f"real, the {len(chunks)} chunks, tokens as edited", real),
            ("synthetic, same chunks, natural amount", syn),
            ("synthetic, same chunks, target_rate ~ training t", syn_t)]
    md = [f"# Damage sampler: validation\n",
          f"Corpus {stats['version']}, split `{a.split}`; statistics file `{os.path.relpath(a.stats or D.stats_path(), ROOT)}`. "
          f"{len(chunks)} random chunks (lines of a text concatenated, cut every {CHUNK} signs). Synthetic masks: "
          "`sample_mask` with the stratum of the text and `text_start` / `text_end` set where the chunk begins / ends "
          "the text. Column 3 uses `allow_clean=True` (the natural amount of damage, including undamaged texts) and "
          "is the one to compare with the real columns; column 4 requests a masking rate t ~ Beta(2, 4) clipped to "
          "[0.05, 0.80] per chunk (the training noise rate), so its amounts differ by construction and only its shapes "
          "are of interest. Columns 1 and 2 differ in two ways: column 1 counts every `...` as its estimated number of "
          "signs and takes whole texts; column 2 counts tokens and takes chunks (the first and last line of a chunk "
          "may be cut). The sampler is fitted to column 1 and applied to the line lengths of column 2.\n",
          "| statistic | " + " | ".join(c[0] for c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for k in syn:
        if k in ("signs", "lines"):
            continue
        md.append(f"| {k} | " + " | ".join(fmt(c[1].get(k, float('nan'))) for c in cols) + " |")
    md.append("\n'groups' are texts in column 1 and chunks in the others.\n")
    err = np.abs(got - req)
    md.append("## Requested and realised masking rate\n")
    md.append(f"`target_rate` on the {len(chunks)} chunks (request = max(1, round(t x signs)) masked signs): realised "
              f"minus requested number of masked signs: exact in {np.mean(err == 0):.1%} of the chunks, mean absolute "
              f"deviation {err.mean():.2f} signs, 99th percentile {np.percentile(err, 99):.0f}, maximum {err.max():.0f} "
              "(the tolerance of the scaling is 1% of the chunk, at least one sign).\n")
    # ---------- timing
    md.append("## Speed\n")
    md.append("| call | microseconds per call (mean over the chunks) |\n|---|---:|")
    tim = {}
    for name, kw in (("sample_mask, natural amount", {}), ("sample_mask, target_rate=0.05", {"target_rate": 0.05}),
                     ("sample_mask, target_rate=0.3", {"target_rate": 0.3}),
                     ("sample_mask, target_rate=0.8", {"target_rate": 0.8})):
        sub = chunks[:5000]
        t0 = time.perf_counter()
        for ti, l, s, e, ts, te in sub:
            D.sample_mask(rng, l, stats, **kw)
        tim[name] = (time.perf_counter() - t0) / len(sub) * 1e6
        md.append(f"| `{name}` | {tim[name]:.0f} |")
    sub = chunks[:5000]
    t0 = time.perf_counter()
    gaps = [D.sample_gap(rng, l, stats, 20) for ti, l, s, e, ts, te in sub]
    tim["sample_gap, max_signs=20"] = (time.perf_counter() - t0) / len(sub) * 1e6
    md.append(f"| `sample_gap, max_signs=20` | {tim['sample_gap, max_signs=20']:.0f} |")
    md.append(f"\nMean chunk: {np.mean([x.sum() for x in lens]):.0f} signs in {np.mean([len(x) for x in lens]):.1f} lines. "
              "One CPU core, including the lazy preparation of the tables on the first call.\n")
    # ---------- gap types
    md.append("## `sample_gap`: types\n")
    tc = stats["gap_type_counts"][0]
    md.append("| type | real connected regions (share) | sampled, max_signs=20 (share) | sampled: mean signs | sampled: mean lines |\n|---|---:|---:|---:|---:|")
    cn = collections.Counter(g[1]["type"] for g in gaps if g[0] is not None)
    for i, t in enumerate(D.GAP_TYPES):
        sel = [g for g in gaps if g[0] is not None and g[1]["type"] == t]
        md.append(f"| {t} | {tc[i] / tc.sum():.3f} | {cn[t] / max(1, sum(cn.values())):.3f} | "
                  f"{np.mean([g[0].sum() for g in sel]) if sel else float('nan'):.1f} | "
                  f"{np.mean([len(g[1]['lines']) for g in sel]) if sel else float('nan'):.1f} |")
    md.append("\nReal regions: maximal groups of consecutive lines with a loss at the same edge (corner: the extent "
              "changes by at least 0.4 of the line from the first to the last line, monotonically for at least three "
              "quarters of the steps), blocks of wholly lost lines, medial runs (hole: on two or more consecutive lines "
              "within a quarter of the line of each other).\n")
    # ---------- examples
    md.append("## Examples\n")
    md.append("One row per line; `#` lost, `+` abraded (damaged but partly legible), `.` preserved; rows longer than "
              "60 signs are cut (`>`). Left: a real damaged tablet (tokens as edited, at most its first 256 signs). "
              "Right: a synthetic mask drawn for the same line lengths (`sample_mask`, same stratum, `target_rate` = the "
              "damaged share of the real tablet, so that the two can be compared by eye). The tablets are the first ten "
              "of the random chunks that begin a text, have 8 to 26 lines and 8% to 60% damaged signs.\n")
    ex = []
    seen = set()
    for ti, l, s, e, ts, te in chunks:
        r = (tok[s:e] > 0).mean()
        if ts and ti not in seen and 8 <= len(l) <= 26 and 0.08 <= r <= 0.6:
            seen.add(ti)
            ex.append((ti, l, s, e, te))
        if len(ex) == 10:
            break
    txt = []
    for ti, l, s, e, te in ex:
        t = texts[ti]
        m, ab = D.sample_mask(rng, l, stats, stratum=D.stratum_of(t), text_start=True, text_end=te,
                              target_rate=float((tok[s:e] > 0).mean()))
        A = D.render_ascii(l, tok[s:e] == D.LOST, tok[s:e] == D.ABRADED)
        B = D.render_ascii(l, m, ab)
        w = max(len(x) for x in A) + 4
        txt.append(f"{t['id']}  {t['lang']}, {t.get('period')}, {t.get('genre')}  [{D.stratum_of(t)}]  "
                   f"real {(tok[s:e] > 0).mean():.0%} damaged, synthetic {m.mean():.0%}")
        txt.append("real".ljust(w) + "synthetic")
        txt += [x.ljust(w) + y for x, y in zip(A, B)]
        txt.append("")
    md += ["```"] + txt + ["```\n"]
    md.append("The same line structures at requested rates (`target_rate` 0.15, 0.4, 0.8; first three tablets):\n")
    txt = []
    for ti, l, s, e, te in ex[:3]:
        rows = []
        for tr in (0.15, 0.4, 0.8):
            m, ab = D.sample_mask(rng, l, stats, stratum=D.stratum_of(texts[ti]), target_rate=tr, text_start=True, text_end=te)
            rows.append(D.render_ascii(l, m, ab))
        w = max(len(x) for x in rows[0]) + 4
        txt.append(texts[ti]["id"])
        txt.append("".join(f"t = {tr}".ljust(w) for tr in (0.15, 0.4, 0.8)))
        txt += ["".join(x.ljust(w) for x in r) for r in zip(*rows)]
        txt.append("")
    md += ["```"] + txt + ["```\n"]
    md.append("`sample_gap` (one connected region, max_signs = 20) on the first tablet, one per type:\n")
    ti, l, s, e, te = ex[0]
    rows, heads = [], []
    for t in D.GAP_TYPES:
        m, info = D.sample_gap(rng, l, stats, 20, stratum=D.stratum_of(texts[ti]), kind=t)
        if m is not None:
            rows.append(D.render_ascii(l, m))
            heads.append(f"{t} ({info['side']})" if t in ("corner",) else t)
    w = max(len(x) for x in rows[0]) + 4
    md += ["```", "".join(h.ljust(w) for h in heads)] + ["".join(x.ljust(w) for x in r) for r in zip(*rows)] + ["```\n"]
    with open(os.path.join(a.out, "VALIDATION.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    J = {"version": stats["version"], "split": a.split, "n_chunks": len(chunks),
         "columns": {c[0]: {k: (None if isinstance(v, float) and v != v else float(v)) for k, v in c[1].items()}
                     for c in cols},
         "rate_abs_dev_signs_mean": float(err.mean()), "timing_us": tim}
    with open(os.path.join(a.out, "validation.json"), "w") as fh:
        json.dump(J, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
