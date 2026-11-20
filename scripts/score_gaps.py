"""Score every system on a gap / transliteration evaluation set.

  python scripts/score_gaps.py --split dev

Reads results/gaps/*__<split>.pkl.gz (models from cun.evaluate, ngram from scripts/eval_gaps_ngram.py).
Gap metrics per instance (gold = the clean signs of the gap):
  exact     whole gap right (sign sequence)
  ser       sign error rate: Levenshtein(pred signs, gold signs) / L
  cer       transliteration CER: Levenshtein(rendered pred, rendered gold) / len(rendered gold)
            (rendered with readings, joiners and determinatives; cun.evalset.render)
  read      known width only: per-slot reading accuracy (slots whose gold sign has a reading)
  bound     known width only: word-boundary F1 inside the gap (joiner 'w' or line end), pooled
  det       known width only: per-slot determinative accuracy
  top1/5/10 single-sign gaps, known width
Bins: L = 1, 2-5, 6-10, 11-15, 16-20 and cumulative 1-5, 1-10, 1-15, 1-20; languages sux / akk (L 1-20).
Every table also on 'novel' gaps (gap + 2 signs of context on each side found nowhere verbatim in the
train texts). Paired bootstrap (texts resampled, 2,000 draws) of each model decoder minus the 4-gram on
exact, ser, cer (known width, L 1-20; all and novel gaps).
Transliteration (bare signs): reading accuracy over identified signs with a reading; polyvalent signs
(top reading < 90% of the sign's occurrences in train); by language.
Output: results/gaps/scores__<split>.json and .md
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN  # noqa: E402
from cun.paths import RESULTS  # noqa: E402

BINS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16-20", 16, 20),
        ("1-5", 1, 5), ("1-10", 1, 10), ("1-15", 1, 15), ("1-20", 1, 20)]
BOUND = (0, 6)


def per_instance(ev, inst, pred, cond):
    rows = []
    for x, p in zip(inst, pred):
        gs, gr, gj, gd = evalset.gold(ev, x)
        L = x["L"]
        ps = [int(v) for v in p["sign"]]
        g_str = evalset.render(ev, gs, gr, gj, gd)
        p_str = evalset.render(ev, p["sign"], p["read"], p["join"], p["det"])
        r = {"L": L, "exact": float(ps == gs.tolist()),
             "ser": Levenshtein.distance(ps, gs.tolist()) / L,
             "cer": Levenshtein.distance(p_str, g_str) / max(1, len(g_str))}
        if "cands" in p:                               # ranked whole-gap candidates
            hit = [list(int(v) for v in c) == gs.tolist() for c in p["cands"]]
            for k in (5, 10, 20):
                r[f"ex@{k}"] = float(any(hit[:k]))
        if cond == "known":
            has = gr > 1
            r["read_n"], r["read_ok"] = int(has.sum()), int((p["read"][has] == gr[has]).sum())
            r["det_ok"] = int((p["det"] == gd).sum())
            gb, pb = np.isin(gj[:-1], BOUND), np.isin(p["join"][:-1], BOUND)
            r["tp"], r["fp"], r["fn"] = int((gb & pb).sum()), int((~gb & pb).sum()), int((gb & ~pb).sum())
            if L == 1 and "topk" in p:
                tk = list(p["topk"])
                for k in (1, 5, 10):
                    r[f"top{k}"] = float(int(gs[0]) in tk[:k])
        rows.append(r)
    return rows


def agg(rows):
    out = {"n": len(rows)}
    if not rows:
        return out
    for k in ("exact", "ser", "cer"):
        out[k] = 100 * float(np.mean([r[k] for r in rows]))
    if "read_n" in rows[0]:
        out["read"] = 100 * sum(r["read_ok"] for r in rows) / max(1, sum(r["read_n"] for r in rows))
        out["det"] = 100 * sum(r["det_ok"] for r in rows) / sum(r["L"] for r in rows)
        tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
        out["bound_f1"] = 100 * 2 * tp / max(1, 2 * tp + fp + fn)
    for k in ("top1", "top5", "top10", "ex@5", "ex@10", "ex@20"):
        v = [r[k] for r in rows if k in r]
        if v and (k.startswith("top") or len(v) == len(rows)):
            out[k] = 100 * float(np.mean(v))
    return out


def boot(a, b, cl, n=2000, seed=0):
    """Paired cluster bootstrap of mean(a - b); returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    u, inv = np.unique(cl, return_inverse=True)
    s = np.bincount(inv, d)
    c = np.bincount(inv)
    draws = []
    for _ in range(n):
        k = rng.integers(0, len(u), len(u))
        draws.append(s[k].sum() / c[k].sum())
    return 100 * d.mean(), 100 * np.percentile(draws, 2.5), 100 * np.percentile(draws, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    ev = evalset.load(ROOT, a.split, version=evalset.corpus_version(ROOT))
    inst = ev["instances"]
    novel = np.array([x["novel"] for x in inst])
    lang = np.array([ev["langs"][ev["chunks"][x["chunk"]][1]] for x in inst])
    text = np.array([ev["chunks"][x["chunk"]][0] for x in inst])
    Ls = np.array([x["L"] for x in inst])
    systems = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "gaps", f"*__{a.split}.pkl.gz"))):
        name = os.path.basename(f).split("__")[0]
        systems[name] = pickle.load(gzip.open(f, "rb"))
        if systems[name].get("version") != ev["version"]:
            raise SystemExit(f"{f}: corpus version {systems[name].get('version')} != {ev['version']}")
    res = {"split": a.split, "version": ev["version"], "novel_share": {b: float(novel[(Ls >= lo) & (Ls <= hi)].mean())
           for b, lo, hi in BINS}, "gaps": {}, "boot": {}, "translit": {}}
    per = {}
    for name, out in systems.items():
        for key, pred in out.items():
            if "/" not in key or len(pred) != len(inst):
                continue
            cond = key.split("/")[0]
            sysname = f"{name}/{key.split('/')[1]}"
            rows = per_instance(ev, inst, pred, cond)
            per[(sysname, cond)] = rows
            d = {}
            for b, lo, hi in BINS:
                d[b] = agg([r for r, L in zip(rows, Ls) if lo <= L <= hi])
                d["novel:" + b] = agg([r for r, L, nv in zip(rows, Ls, novel) if lo <= L <= hi and nv])
            for lg in ("sux", "akk"):
                d[lg] = agg([r for r, l in zip(rows, lang) if l == lg])
            res["gaps"].setdefault(cond, {})[sysname] = d
    base = per.get(("ngram/4gram", "known"))
    if base:
        for (sysname, cond), rows in per.items():
            if cond != "known" or sysname.startswith("ngram/"):
                continue
            res["boot"][sysname] = {k: boot([r[k] for r in rows], [r[k] for r in base], text)
                                    for k in ("exact", "ser", "cer")}
            nv = np.flatnonzero(novel)
            res["boot"][sysname + " (novel)"] = {k: boot([rows[i][k] for i in nv], [base[i][k] for i in nv], text[nv])
                                                 for k in ("exact", "ser", "cer")}
    # transliteration
    poly = set(systems.get("ngram", {}).get("polyvalent_signs", []))
    gold_r, sign, tl = [], [], []
    for ti, lg, a2 in ev["chunks"]:
        m = (a2[1] > 1) & (a2[0] >= FIRST_REAL_SIGN)
        gold_r.append(a2[1][m])
        sign.append(a2[0][m])
        tl.append(np.full(int(m.sum()), ev["langs"][lg]))
    gold_r, sign, tl = np.concatenate(gold_r), np.concatenate(sign), np.concatenate(tl)
    is_poly = np.isin(sign, list(poly))
    for name, out in systems.items():
        if "translit" not in out:
            continue
        keys = out["translit"][0].keys()
        for k in keys:
            p = []
            for (ti, lg, a2), d in zip(ev["chunks"], out["translit"]):
                m = (a2[1] > 1) & (a2[0] >= FIRST_REAL_SIGN)
                p.append(np.asarray(d[k])[m])
            p = np.concatenate(p)
            ok = p == gold_r
            res["translit"][f"{name}/{k}"] = {
                "all": 100 * ok.mean(), "n": int(len(ok)), "polyvalent": 100 * ok[is_poly].mean(),
                "n_poly": int(is_poly.sum()), **{lg: 100 * ok[tl == lg].mean() for lg in ("sux", "akk")}}
    json.dump(res, open(os.path.join(RESULTS, "gaps", f"scores__{a.split}.json"), "w"), indent=1)
    write_md(res, a.split)


def write_md(res, split):
    L = [f"# Sign-level evaluation, split {split} (corpus {res['version']})\n",
         "Novel gaps: the gap plus 2 signs of context on each side occurs nowhere verbatim in the train texts. "
         "Share novel by L bin: " + ", ".join(f"{b} {100 * v:.0f}%" for b, v in res["novel_share"].items()) + "\n"]
    f = lambda v: "" if v is None else f"{v:.1f}"
    for cond, title in (("known", "Gap restoration, known width"), ("unknown", "Gap restoration, unknown width")):
        if cond not in res["gaps"]:
            continue
        S = res["gaps"][cond]
        for metric, mt in (("cer", "Transliteration CER (%; lower is better)"), ("ser", "Sign error rate (%)"),
                           ("exact", "Whole gap exactly right (%)")):
            L.append(f"\n## {title}: {mt}\n")
            cols = [b for b, _, _ in BINS] + ["sux", "akk"]
            L.append("| system | " + " | ".join(cols) + " |")
            L.append("|---" * (len(cols) + 1) + "|")
            for s, d in S.items():
                L.append(f"| {s} | " + " | ".join(f(d[c].get(metric)) for c in cols) + " |")
            for s, d in S.items():
                L.append(f"| {s}, novel gaps | " + " | ".join(f(d.get("novel:" + c, {}).get(metric)) for c in cols) + " |")
        if cond == "known":
            L.append(f"\n## {title}: other planes (L 1-20) and single-sign top-k\n")
            L.append("| system | reading acc | word-boundary F1 | det acc | top1 | top5 | top10 |")
            L.append("|---|---|---|---|---|---|---|")
            for s, d in S.items():
                a1, a2 = d["1-20"], d["1"]
                L.append(f"| {s} | {f(a1.get('read'))} | {f(a1.get('bound_f1'))} | {f(a1.get('det'))} | "
                         f"{f(a2.get('top1'))} | {f(a2.get('top5'))} | {f(a2.get('top10'))} |")
    if res["boot"]:
        L.append("\n## Model minus 4-gram (known width, L 1-20; 95% CI, texts resampled)\n")
        L.append("| system | exact | SER | CER |")
        L.append("|---|---|---|---|")
        for s, d in res["boot"].items():
            L.append(f"| {s} | " + " | ".join(f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]" for v in
                                              (d["exact"], d["ser"], d["cer"])) + " |")
    if res["translit"]:
        L.append("\n## Transliteration of bare signs: reading accuracy (%)\n")
        L.append("| system | all | polyvalent | sux | akk | n |")
        L.append("|---|---|---|---|---|---|")
        for s, d in res["translit"].items():
            L.append(f"| {s} | {d['all']:.1f} | {d['polyvalent']:.1f} | {d['sux']:.1f} | {d['akk']:.1f} | {d['n']} |")
    open(os.path.join(RESULTS, "gaps", f"scores__{split}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
