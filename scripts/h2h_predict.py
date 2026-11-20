"""Head-to-head, step 2 (GPU node): the sign model and the sign 4-gram on the matched positions.

  python scripts/h2h_predict.py [--ckpt ckpt/base-long/best.pt] [--max N]

Sign model, one masked slot per input (sign MASK, reading READ_MASK, joiner and determinative unknown),
language 'sux':
  line     the line alone, as system B sees it (other signs with readings, joiners, determinatives; the
           line's last joiner is the line-end mark, as in pretraining)
  tablet   up to 256 signs of the tablet around the slot (lines concatenated, line ends on the join plane)
Signs ranked among real signs; the reading of each of the 5 best signs is the best reading attested for
that sign in the train texts (cun.evaluate.Decoder.read_choice).
4-gram: Witten-Bell sign 4-gram of the 'sux' train chunks (cun.ngram, as scripts/eval_gaps_ngram.py), beam 8,
right context scored; reading by Translit.ctx over the filled sequence; same two contexts.
Output: results/headtohead/predictions.pkl.gz
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
from cun.consts import FIRST_REAL_SIGN, MASK, READ_MASK  # noqa: E402
from cun.devset import CHUNK, load_chunks  # noqa: E402
from cun.ngram import WB, Translit, fill_runs  # noqa: E402
from cun.noise import D_UNK, J_UNK  # noqa: E402

from cun.paths import PROC, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "headtohead")
LINE_END = 6
G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def row(planes, g, lang):
    a = planes.astype(np.int64).copy()
    a[0, g], a[1, g], a[2, g], a[3, g] = MASK, READ_MASK, J_UNK, D_UNK
    return {"sign": a[0], "read": a[1], "join": a[2], "det": a[3], "lang": lang, "gap": g, "M": 1}


def inputs(P):
    """-> {'line': rows, 'tablet': rows} aligned with P['positions']."""
    out = {"line": [], "tablet": []}
    for x in P["positions"]:
        a = P["lines"][x["lid"]]["planes"].copy()
        a[2, -1] = LINE_END
        assert a[0, x["slot"]] == x["gold_sign"]
        out["line"].append(row(a, x["slot"], P["lang"]))
        t = P["tablets"][x["pid"]]
        q, n = x["tab_slot"], t.shape[1]
        assert t[0, q] == x["gold_sign"]
        s = 0 if n <= CHUNK else min(max(0, q - CHUNK // 2), n - CHUNK)
        out["tablet"].append(row(t[:, s:s + CHUNK], q - s, P["lang"]))
    return out


def run_model(a, P, rows, sign_reads):
    import torch
    from cun.evaluate import Decoder, forward
    from cun.model import CunConfig, CunEncoder
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cuda", weights_only=False)
    assert ck.get("corpus_version") == P["version"], (ck.get("corpus_version"), P["version"])
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    dec = Decoder(model, {"sign_reads": sign_reads}, "cuda")
    res = {"step": ck.get("step")}
    for cond, rs in rows.items():
        order = np.argsort([len(r["sign"]) for r in rs], kind="stable")
        top = np.zeros((len(rs), 10), np.int32)
        rd = np.zeros((len(rs), 5), np.int32)
        pr = np.zeros((len(rs), 10), np.float32)
        for s in range(0, len(order), 96):
            idx = order[s:s + 96]
            out = forward(model, [rs[i] for i in idx], "cuda")
            for j, i in enumerate(idx):
                g = rs[i]["gap"]
                p = torch.softmax(out["sign"][j, g].masked_fill(~dec.allow_known, float("-inf")), -1)
                tk = p.topk(10)
                top[i] = tk.indices.cpu().numpy()
                pr[i] = tk.values.cpu().numpy()
                rd[i] = dec.read_choice(out["read"][j, g].unsqueeze(0).expand(5, -1), tk.indices[:5]).cpu().numpy()
        res[cond] = {"top": top, "read": rd, "prob": pr}
        log("model", cond, "done")
    return res


def ngram_one(k):
    sg, g = G["rows"][k]
    filled, ranked = fill_runs(G["lm"], G["cands"], sg, beam=8, topk=10)
    return ranked[g].astype(np.int32), int(G["tr"].ctx(filled)[g])


def run_ngram(a, rows, corpus):
    V, NR = len(corpus["sign_vocab"]), len(corpus["read_vocab"])
    train = [c for c in load_chunks(corpus, "train") if corpus["langs"][c[4]] == "sux"]
    G["lm"] = WB([c[0] for c in train], 4, 4096, V)
    G["tr"] = Translit(train, V, NR, corpus["sign_reads"])
    G["cands"] = np.arange(FIRST_REAL_SIGN, V, dtype=np.int64)
    log("4-gram built on sux train chunks:", len(train))
    res = {}
    for cond, rs in rows.items():
        G["rows"] = [(r["sign"], r["gap"]) for r in rs]
        with mp.get_context("fork").Pool(a.workers) as pool:
            o = pool.map(ngram_one, range(len(rs)), chunksize=16)
        res[cond] = {"top": np.stack([t for t, _ in o]), "read": np.array([r for _, r in o], np.int32)}
        log("4-gram", cond, "done")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/base-long/best.pt")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max", type=int, default=None, help="smoke test")
    a = ap.parse_args()
    P = pickle.load(gzip.open(os.path.join(OUT, "positions.pkl.gz"), "rb"))
    if a.max:
        P["positions"] = P["positions"][:: max(1, len(P["positions"]) // a.max)]
    rows = inputs(P)
    log("positions", len(P["positions"]))
    corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    assert corpus["version"] == P["version"]
    out = {"version": P["version"], "ckpt": a.ckpt, "n": len(P["positions"]),
           "key": [(x["set"], x["lid"], x["pos"]) for x in P["positions"]]}
    out["ngram"] = run_ngram(a, rows, corpus)          # before torch: fork-safe
    out["model"] = run_model(a, P, rows, corpus["sign_reads"])
    tag = "_smoke" if a.max else ""
    with gzip.open(os.path.join(OUT, f"predictions{tag}.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
