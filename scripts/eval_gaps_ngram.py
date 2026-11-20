"""Non-neural systems on the gap and transliteration evaluation sets (same output format as cun.evaluate).

  python scripts/eval_gaps_ngram.py --split dev [--workers 12]

Training data: the pretraining train texts (split 'train'; dev, test and excluded texts never), as for
the neural model.
Every model is language-specific (separate Sumerian and Akkadian models; pooled for other languages),
as the neural model is told the language; '-pooled' variants ignore language.
Gap restoration, known width only (an n-gram has no model of gap width):
  4gram   Witten-Bell sign 4-gram, beam 8, right context scored (cun.ngram.fill_runs); readings by the
          context lookup over the filled sign sequence; joiner and determinative: most frequent for the
          (sign, reading) pair.
Transliteration (bare signs): mf, ctx, hmm (cun.ngram.Translit).
Output: results/gaps/ngram__<split>.pkl.gz
"""
from __future__ import annotations

import argparse
import gzip
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN  # noqa: E402
from cun.devset import load_chunks  # noqa: E402
from cun.paths import PROC, RESULTS  # noqa: E402
from cun.ngram import WB, Translit, fill_runs  # noqa: E402

G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def group(lang_idx):
    l = G["langs"][lang_idx]
    return l if l in G["lm"] else "all"


def gap_one(x, pooled=False):
    ev = G["ev"]
    gk = "all" if pooled else group(ev["chunks"][x["chunk"]][1])
    lm, T = G["lm"][gk], G["tr"][gk]
    r = evalset.gap_input(ev, x, "known")
    g, L = r["gap"], x["L"]
    filled, ranked = fill_runs(lm, G["cands"], r["sign"], beam=8, topk=10 if L == 1 else 0)
    sg = filled[g:g + L]
    rd = T.ctx(filled)[g:g + L]
    d = {"sign": sg, "read": rd, "join": np.array([T.join(s, q) for s, q in zip(sg, rd)]),
         "det": np.array([T.det(s, q) for s, q in zip(sg, rd)])}
    if L == 1:
        d["topk"] = ranked[g]
    return d


def gap_pooled(x):
    return gap_one(x, pooled=True)


def translit_one(ch):
    T, P = G["tr"][group(ch[1])], G["tr"]["all"]
    a = ch[2]
    sign = a[0].astype(np.int64)
    tgt = (a[1] > 1) & (sign >= FIRST_REAL_SIGN)
    return {"mf": np.array([T.mf(s) for s in sign]), "ctx": T.ctx(sign), "hmm": T.hmm(sign, tgt),
            "hmm-pooled": P.hmm(sign, tgt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    V, NR = len(corpus["sign_vocab"]), len(corpus["read_vocab"])
    train = load_chunks(corpus, "train")
    excl = [t for t in corpus["texts"] if t["split"] == "excluded"]
    G["ev"] = ev = evalset.load(ROOT, a.split, version=corpus["version"])
    G["langs"] = corpus["langs"]
    corpus_v = corpus["version"]
    G["cands"] = np.arange(FIRST_REAL_SIGN, V, dtype=np.int64)
    G["lm"], G["tr"] = {}, {}
    for gk in ("all", "sux", "akk"):
        sub = train if gk == "all" else [c for c in train if corpus["langs"][c[4]] == gk]
        G["lm"][gk] = WB([c[0] for c in sub], 4, 4096, V)
        G["tr"][gk] = Translit(sub, V, NR, corpus["sign_reads"])
        log("built", gk, len(sub))
    T = G["tr"]["all"]
    del corpus
    log("models built; train chunks", len(train), "excluded texts", len(excl))
    with mp.get_context("fork").Pool(a.workers) as pool:
        gaps = pool.map(gap_one, ev["instances"], chunksize=4)
        gaps_pooled = pool.map(gap_pooled, ev["instances"], chunksize=4)
        log("gaps done")
        tr = pool.map(translit_one, ev["chunks"], chunksize=4)
        log("translit done")
    poly = sorted(int(s) for s in T.c1 if T.polyvalent(s))
    out = {"split": a.split, "version": corpus_v, "n_inst": len(gaps), "known/4gram": gaps, "known/4gram-pooled": gaps_pooled, "translit": tr, "polyvalent_signs": poly}
    os.makedirs(os.path.join(RESULTS, "gaps"), exist_ok=True)
    with gzip.open(os.path.join(RESULTS, "gaps", f"ngram__{a.split}.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
