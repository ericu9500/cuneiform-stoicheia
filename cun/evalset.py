"""Gap-restoration evaluation instances (torch-free; shared by the model evaluation, the n-gram baseline
and the scorer).

An instance is one chunk (<= 256 signs of a text, lines concatenated) with ONE contiguous gap of L signs,
L = 1..20. The gap may cross word and line boundaries. Gold signs must be clean: identified signs (no
<x>/<gap>/<unk-sign>) that are neither damaged nor restored by the editor in the source. The rest of the
chunk is given as it is in the edition, including its real damage, with readings, joiners and
determinatives visible (restoring a break in a transliterated text).

Conditions
  known     the gap has exactly L slots (the width of the break is known, like [x x x])
  unknown   the gap has L + extra slots, extra ~ U{0 .. ceil(0.3 L) + 2} as in pretraining; systems may
            leave slots empty (BLANK) (width uncertain, like [...])

Splits: 'dev' (the 2,000 pretraining dev texts; for development only) and 'test' (2,000 texts never used for
a choice), both labelled in the corpus. Each instance carries `novel`: its gap plus 2 signs of context on
each side occurs verbatim nowhere in the train texts (scores are reported on all and on novel gaps).
"""
from __future__ import annotations

import gzip
import json
import os
import pickle

import numpy as np

from .consts import FIRST_REAL_SIGN, MASK, READ_MASK
from .devset import CHUNK, LINE_END, check_version, in_split
from .paths import PROC
from .noise import D_UNK, J_UNK

MAX_L = 20
# corpus v3: test1 = the v2 test texts (same seed and size as 'test', so the same gaps); devA / devB / test2 new
SEEDS = {"dev": 1, "test": 2, "test1": 2, "devA": 3, "devB": 4, "test2": 5}
PER_L = {"dev": 300, "test": 500, "test1": 500, "devA": 300, "devB": 300, "test2": 500}
JOIN_STR = {0: " ", 1: "-", 2: ".", 3: ":", 4: "+", 5: "", 6: " "}


def split_texts(corpus, split):
    assert split in SEEDS, split
    return {i for i, t in enumerate(corpus["texts"]) if in_split(t, split)}


def novelty(corpus, chunks, inst, ctx=2):
    """Per instance: True if the gap with `ctx` signs of context on each side (as far as the chunk allows)
    occurs verbatim nowhere in the train texts (lines concatenated per text, as in pretraining)."""
    by_text = {}
    for ti, a in corpus["lines"]:
        if corpus["texts"][ti]["split"] == "train":
            by_text.setdefault(ti, []).append(a[0])
    big = np.concatenate([np.r_[np.concatenate(v), -1] for v in by_text.values()]).astype(np.int64)
    out = np.zeros(len(inst), bool)
    wins = {}
    for k, x in enumerate(inst):
        a = chunks[x["chunk"]][2][0].astype(np.int64)
        lo, hi = max(0, x["start"] - ctx), min(len(a), x["start"] + x["L"] + ctx)
        wins.setdefault(hi - lo, []).append((k, a[lo:hi]))
    P = np.uint64(1000003)
    for W, items in wins.items():
        h = np.zeros(len(big) - W + 1, np.uint64)
        bad = np.zeros(len(big) - W + 1, np.int64)
        for j in range(W):
            seg = big[j:len(big) - W + 1 + j]
            h = h * P + (seg + 2).astype(np.uint64)
            bad += seg < 0
        hs = np.unique(h[bad == 0])
        for k, w in items:
            hw = 0
            for v in w:                                # same wrap-around arithmetic as the array hash
                hw = (hw * 1000003 + int(v) + 2) % (1 << 64)
            hw = np.uint64(hw)
            i = np.searchsorted(hs, hw)
            out[k] = not (i < len(hs) and hs[i] == hw)
    return out


def eval_chunks(corpus, texts):
    """[(text idx, lang idx, (5, n) array: sign, read, join, det, brk)] chunked like pretraining."""
    langs = {l: i for i, l in enumerate(corpus["langs"])}
    by_text = {}
    for ti, a in corpus["lines"]:
        if ti in texts:
            a = a.copy()
            a[2, -1] = LINE_END
            by_text.setdefault(ti, []).append(a)
    out = []
    for ti, arrs in by_text.items():
        a = np.concatenate(arrs, 1)
        lg = langs[corpus["texts"][ti]["lang"]]
        for s in range(0, a.shape[1], CHUNK):
            c = a[:, s:s + CHUNK]
            if c.shape[1] >= 2:
                out.append((ti, lg, c))
    return out


def make_instances(chunks, split):
    rng = np.random.default_rng(SEEDS[split])
    inst = []
    for L in range(1, MAX_L + 1):
        order = rng.permutation(len(chunks))
        k = 0
        for ci in order:
            if k >= PER_L[split]:
                break
            a = chunks[ci][2]
            n = a.shape[1]
            if n < L + 4:                                  # some context besides the gap
                continue
            ok = (a[0] >= FIRST_REAL_SIGN) & (a[4] == 0)
            win = np.convolve(ok.astype(int), np.ones(L, int), "valid") == L
            starts = np.flatnonzero(win)
            if not len(starts):
                continue
            s = int(rng.choice(starts))
            extra = int(rng.integers(0, int(np.ceil(0.3 * L)) + 3))
            inst.append({"chunk": int(ci), "start": s, "L": L, "extra": extra})
            k += 1
    return inst


def build(corpus, root, split):
    """`root` is kept for the callers' signature; the files live in cun.paths.PROC (CUN_PROC)."""
    chunks = eval_chunks(corpus, split_texts(corpus, split))
    inst = make_instances(chunks, split)
    nov = novelty(corpus, chunks, inst)
    for x, v in zip(inst, nov):
        x["novel"] = bool(v)
    obj = {"version": corpus["version"], "split": split, "chunks": chunks, "instances": inst, "langs": corpus["langs"],
           "sign_names": corpus["sign_names"], "read_vocab": corpus["read_vocab"],
           "sign_reads": corpus["sign_reads"]}
    with gzip.open(os.path.join(PROC, f"evalset_{split}.pkl.gz"), "wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    return obj


def load(root, split, version=None):
    ev = pickle.load(gzip.open(os.path.join(PROC, f"evalset_{split}.pkl.gz"), "rb"))
    if version is not None:
        check_version(version, ev.get("version"), f"evalset_{split}")
    return ev


def corpus_version(root):
    """Version of the current corpus without loading it (corpus_stats.json is written with it)."""
    return json.load(open(os.path.join(PROC, "corpus_stats.json")))["version"]


def gap_input(ev, inst, cond):
    """Input planes for one instance: dict sign/read/join/det/status/slot (int64 arrays), lang, gap start,
    n slots. status: the source's damage flag of the context signs (3 on the gap and on unknown signs);
    slot: 1 on the slots of an unknown-width gap (used by models trained with the elastic flag)."""
    ti, lg, a = ev["chunks"][inst["chunk"]]
    s, L = inst["start"], inst["L"]
    M = L + (inst["extra"] if cond == "unknown" else 0)
    planes = {}
    for k, row, fill in (("sign", 0, MASK), ("read", 1, READ_MASK), ("join", 2, J_UNK), ("det", 3, D_UNK)):
        x = a[row].astype(np.int64)
        planes[k] = np.r_[x[:s], np.full(M, fill, np.int64), x[s + L:]]
    st = a[4].astype(np.int64).copy()
    st[a[0] < 6] = 3
    planes["status"] = np.r_[st[:s], np.full(M, 3, np.int64), st[s + L:]]
    planes["slot"] = np.r_[np.zeros(s, np.int64), np.full(M, int(cond == "unknown"), np.int64),
                           np.zeros(len(st) - s - L, np.int64)]
    return {**planes, "lang": lg, "gap": s, "M": M, "text": ti}


def gold(ev, inst):
    """Gold planes of the gap: (sign, read, join, det) arrays of length L."""
    a = ev["chunks"][inst["chunk"]][2]
    s, L = inst["start"], inst["L"]
    return tuple(a[r, s:s + L].astype(np.int64) for r in range(4))


def render(ev, sign, read, join, det):
    """Transliteration string of a gap (inner separators only); reading, else sign name; dets in braces."""
    out = []
    n = len(sign)
    for i in range(n):
        r = int(read[i])
        tok = ev["read_vocab"][r] if r > 1 else ev["sign_names"][int(sign[i])]
        if int(det[i]) in (1, 2):
            tok = "{" + tok + "}"
        out.append(tok)
        if i < n - 1:
            out.append(JOIN_STR.get(int(join[i]), " "))
    return "".join(out)
