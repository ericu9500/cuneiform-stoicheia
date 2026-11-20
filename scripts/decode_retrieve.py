"""Retrieval of parallels in the TRAIN texts for every dev gap (CPU; PLAN_v3 Stage 1, item 6).

  python scripts/decode_retrieve.py            -> results/decode/retrieval__dev.pkl.gz

Query: the (up to) 6 signs before and 6 signs after the gap. For a gap width G a train position p is a
candidate when the signs next to the gap match exactly (anchor), tried in this order:
  '22'  2 signs before and 2 after, G signs apart     '11'  1 before and 1 after
  '30'  3 signs before only                           '03'  3 signs after only
The first tier with a candidate is used. Candidates are ranked by fuzzy agreement of the wider context:
lm / rm = number of the 6 aligned positions on the left / right that carry the same sign (substitutions
allowed, no insertions). Regime of a gap:
  parallel   a '22' candidate with lm >= 4 and rm >= 4      formulaic   any other candidate      none
Retrieved fills = the distinct sign sequences between the anchors (real signs only, never across texts),
ranked by (occurs in a parallel, best lm + rm, occurrences at that best score, occurrences in all).
Only texts with corpus split == 'train' are indexed (hashed n-grams, numpy sort / searchsorted as in
cun.evalset.novelty). Output per instance: {'known': record for G = L, 'byG': {G: record}} with
record = {'tier', 'cat', 'n': candidates, 'fills': [(signs, n_all, best, n_best, n_par)]}.
"""
from __future__ import annotations

import gzip
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN  # noqa: E402
from cun.decode import K_MAX  # noqa: E402

CTX = 6
TIERS = (("22", 2, 2), ("11", 1, 1), ("30", 3, 0), ("03", 0, 3))
CAP = 20000
PM = 1000003


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def train_signs(corpus):
    by_text = {}
    for ti, a in corpus["lines"]:
        if corpus["texts"][ti]["split"] == "train":
            by_text.setdefault(ti, []).append(a[0])
    return np.concatenate([np.r_[np.concatenate(v), -1] for v in by_text.values()]).astype(np.int64)


class Index:
    """Gap starts p of the train signs keyed by the hash of l signs before p and r signs from p + G."""

    def __init__(self, big, bad_cum, G, l, r):
        n = len(big)
        m = n - G - r - l + 1
        self.offs = list(range(-l, 0)) + list(range(G, G + r))
        h = np.zeros(m, np.uint64)
        ok = np.ones(m, bool)
        for off in self.offs:
            seg = big[l + off:l + off + m]
            h = h * np.uint64(PM) + (seg + 2).astype(np.uint64)
            ok &= seg >= FIRST_REAL_SIGN
        P = np.arange(l, l + m)
        ok &= (bad_cum[P + G] - bad_cum[P]) == 0          # the fill: real signs of one text
        P, h = P[ok], h[ok]
        o = np.argsort(h, kind="stable")
        self.h, self.P = h[o], P[o].astype(np.int32)

    def find(self, big, anchor):
        hw = 0
        for v in anchor:
            hw = (hw * PM + int(v) + 2) % (1 << 64)
        hw = np.uint64(hw)
        i, j = np.searchsorted(self.h, hw, "left"), np.searchsorted(self.h, hw, "right")
        pos = self.P[i:j].astype(np.int64)
        if len(pos):                                       # hash collisions
            good = np.ones(len(pos), bool)
            for off, v in zip(self.offs, anchor):
                good &= big[pos + off] == v
            pos = pos[good]
        return pos


def record(big, pos, tier, left, right, G, topn):
    n = len(pos)
    if n > CAP:
        pos = pos[np.linspace(0, n - 1, CAP).astype(np.int64)]
    lm = np.zeros(len(pos), np.int64)
    rm = np.zeros(len(pos), np.int64)
    N = len(big)
    for j in range(1, len(left) + 1):
        if left[-j] >= FIRST_REAL_SIGN:
            lm += big[np.maximum(pos - j, 0)] == left[-j]
    for j in range(len(right)):
        if right[j] >= FIRST_REAL_SIGN:
            rm += big[np.minimum(pos + G + j, N - 1)] == right[j]
    par = (lm >= 4) & (rm >= 4) if tier == "22" else np.zeros(len(pos), bool)
    sc = lm + rm
    fills = big[pos[:, None] + np.arange(G)[None]]
    u, inv = np.unique(fills, axis=0, return_inverse=True)
    inv = inv.ravel()
    n_all = np.bincount(inv, minlength=len(u))
    best = np.zeros(len(u), np.int64)
    np.maximum.at(best, inv, sc)
    n_best = np.bincount(inv, weights=(sc == best[inv]), minlength=len(u)).astype(np.int64)
    n_par = np.bincount(inv, weights=par, minlength=len(u)).astype(np.int64)
    o = np.lexsort((-n_all, -n_best, -best, -(n_par > 0).astype(int)))[:topn]
    return {"tier": tier, "cat": "parallel" if par.any() else "formulaic", "n": int(n),
            "fills": [(u[i].astype(np.int16), int(n_all[i]), int(best[i]), int(n_best[i]), int(n_par[i])) for i in o]}


def main():
    corpus = pickle.load(gzip.open(os.path.join(ROOT, "data", "proc", "corpus.pkl.gz"), "rb"))
    ev = evalset.load(ROOT, "dev", version=corpus["version"])
    big = train_signs(corpus)
    del corpus
    bad_cum = np.r_[0, np.cumsum(big < FIRST_REAL_SIGN)]
    log("train signs", len(big))
    inst = ev["instances"]
    ctx = []
    for x in inst:
        a = ev["chunks"][x["chunk"]][2][0].astype(np.int64)
        s, L = x["start"], x["L"]
        ctx.append((a[max(0, s - CTX):s], a[s + L:s + L + CTX]))
    out = [{"known": None, "byG": {}} for _ in inst]
    for G in range(1, K_MAX + 1):
        todo = list(range(len(inst)))
        for tier, l, r in TIERS:
            if not todo:
                break
            ix = Index(big, bad_cum, G, l, r)
            rest = []
            for i in todo:
                left, right = ctx[i]
                if len(left) < l or len(right) < r:
                    rest.append(i)
                    continue
                anchor = [int(v) for v in left[len(left) - l:]] + [int(v) for v in right[:r]]
                if min(anchor) < FIRST_REAL_SIGN:
                    rest.append(i)
                    continue
                pos = ix.find(big, anchor)
                if not len(pos):
                    rest.append(i)
                    continue
                known = G == inst[i]["L"]
                rec = record(big, pos, tier, left, right, G, 20 if known else 5)
                out[i]["byG"][G] = rec
                if known:
                    out[i]["known"] = rec
            todo = rest
            del ix
        for i in todo:
            out[i]["byG"][G] = {"tier": None, "cat": "none", "n": 0, "fills": []}
            if G == inst[i]["L"]:
                out[i]["known"] = out[i]["byG"][G]
        log("G", G, "no match", len(todo))
    os.makedirs(os.path.join(ROOT, "results", "decode"), exist_ok=True)
    with gzip.open(os.path.join(ROOT, "results", "decode", "retrieval__dev.pkl.gz"), "wb") as fh:
        pickle.dump({"version": ev["version"], "split": "dev", "inst": out}, fh, protocol=5)
    cats = [o["known"]["cat"] for o in out]
    log("saved; known-width regimes:", {c: cats.count(c) for c in ("parallel", "formulaic", "none")})


if __name__ == "__main__":
    main()
