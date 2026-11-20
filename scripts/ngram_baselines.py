"""Non-neural baselines for the sign-level tasks, on the exact dev noisings the neural models are scored on.

Restoration (dev pattern 'span', t = 0.15): Witten-Bell interpolated 4-gram over sign ids, trained on the
pretraining train chunks. Each maximal run of masked signs is filled left to right with beam search
(beam 8); at the last sign of a run the known signs to its right (up to 3) are scored too, so the fill
must also fit what follows. Candidates: every real sign.

Transliteration (dev pattern 'translit': all readings hidden, signs visible):
  mf       most frequent reading of the sign in train
  ctx      most frequent reading given (previous sign, sign, next sign), backing off to the left or right
           sign pair (whichever context is more frequent), then to the sign
  hmm      trigram HMM: hidden states = readings, Witten-Bell trigram over readings, emission
           P(sign | reading); Viterbi. States per sign: its readings seen >= 2 times (top 20).
The transliteration baselines are told which positions carry a reading (the targets); positions without
one enter the reading stream as a sign token.

  python scripts/ngram_baselines.py [--workers 32] [--max-chunks N]
Output: results/baselines_dev.json (accuracy overall, by language, polyvalent subset) and
results/baselines_dev_pred.npz (per-position predictions for paired comparison with the models).
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun.devset import dev_noisings, load_chunks  # noqa: E402
from cun.paths import PROC, RESULTS  # noqa: E402
from cun.ngram import FIRST_REAL, WB, Translit, fill_runs  # noqa: E402
from cun.noise import IGN  # noqa: E402

BEAM = 8
G = {}                  # globals shared with forked workers


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def restore_chunk(d):
    masked = d["s_tg"] != IGN
    seq, _ = fill_runs(G["sign_lm"], G["cands"], d["sign"], beam=BEAM)
    return seq[masked], d["s_tg"][masked]


def translit_chunk(d):
    sign, r_tg = d["sign"].astype(np.int64), d["r_tg"]
    tgt = r_tg != IGN
    T = G["tr"]
    out = {"mf": np.array([T.mf(s) for s in sign]), "ctx": T.ctx(sign), "hmm": T.hmm(sign, tgt)}
    return {k: v[tgt] for k, v in out.items()}, r_tg[tgt], sign[tgt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-chunks", type=int, default=None, help="smoke test: dev chunks per pattern")
    a = ap.parse_args()
    corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    n_lang = len(corpus["langs"])
    V, NR = len(corpus["sign_vocab"]), len(corpus["read_vocab"])
    train = load_chunks(corpus, "train")
    dev_chunks = load_chunks(corpus, "dev")
    dev = dev_noisings(dev_chunks, n_lang)
    log("train chunks", len(train), "dev noisings", len(dev))

    # sign 4-gram
    G["sign_lm"] = WB([c[0] for c in train], 4, 4096, V)
    G["cands"] = np.arange(FIRST_REAL, V, dtype=np.int64)
    log("sign 4-gram built", {n: len(k) for n, k in G["sign_lm"].keys.items()})

    G["tr"] = Translit(train, V, NR, corpus["sign_reads"])
    log("transliteration statistics built")

    span = [d for p, _, d in dev if p == 0][:a.max_chunks]
    trl = [d for p, _, d in dev if p == 4][:a.max_chunks]
    langs_span = [corpus["langs"][int(d["lang"][0])] for d in span]
    langs_trl = [corpus["langs"][int(d["lang"][0])] for d in trl]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        t0 = time.time()
        rs = pool.map(restore_chunk, span, chunksize=1)
        log("restoration done", round(time.time() - t0))
        t0 = time.time()
        ts = pool.map(translit_chunk, trl, chunksize=1)
        log("transliteration done", round(time.time() - t0))

    res = {"restoration": {}, "transliteration": {}}
    pr = np.concatenate([p for p, _ in rs]); gr = np.concatenate([g for _, g in rs])
    lr = np.concatenate([[l] * len(g) for l, (_, g) in zip(langs_span, rs)])
    res["restoration"]["4gram"] = {"all": float((pr == gr).mean()), "n": int(len(gr)),
                                   **{l: float((pr[lr == l] == gr[lr == l]).mean()) for l in ("sux", "akk")}}
    gt = np.concatenate([g for _, g, _ in ts]); st = np.concatenate([s for _, _, s in ts])
    lt = np.concatenate([[l] * len(g) for l, (_, g, _) in zip(langs_trl, ts)])
    poly = np.array([G["tr"].polyvalent(s) for s in st])
    preds = {}
    for k in ("mf", "ctx", "hmm"):
        p = np.concatenate([o[k] for o, _, _ in ts])
        preds[k] = p
        res["transliteration"][k] = {"all": float((p == gt).mean()), "n": int(len(gt)),
                                     "polyvalent": float((p[poly] == gt[poly]).mean()), "n_poly": int(poly.sum()),
                                     **{l: float((p[lt == l] == gt[lt == l]).mean()) for l in ("sux", "akk")}}
    os.makedirs(RESULTS, exist_ok=True)
    tag = "" if a.max_chunks is None else f"_smoke{a.max_chunks}"
    json.dump(res, open(os.path.join(RESULTS, f"baselines_dev{tag}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(RESULTS, f"baselines_dev_pred{tag}.npz"),
                        restore_pred=pr, restore_gold=gr, restore_lang=lr,
                        translit_gold=gt, translit_sign=st, translit_lang=lt, translit_poly=poly, **preds)
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
