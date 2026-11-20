"""Duplicate-witness evaluation set (PLAN_v3 stage 0, item 5): REAL lacunae with attested answers.

Where one exemplar (witness A) of a composition is broken and another exemplar (witness B) preserves the
same passage, B supplies the answer. Nothing is simulated: the break is A's, the answer is B's.

Sets (never a text with split == 'train', so the model has seen neither witness):
  witness-dev    both witnesses in {devA, devB, excluded}, the excluded ones not linked to a test2 text
  witness-test   both witnesses in {test2, excluded linked to a test2 text}; a pair without a test2 text
                 (two excluded exemplars of a test2 composition) is kept but flagged has_test2 = False
                 (`instances(..., strict=True)` leaves it out). To be touched once, at the very end.
An excluded text is 'linked to test2' if it shares a composite (Q-number) or a museum number with a test2
text or is its sign-8-gram near-duplicate; such a text is never used in witness-dev. Excluded exemplars of
test1 compositions are used in witness-dev (test1 texts themselves are not used at all).

Pair sources
  composite   the two texts share a composite id (text['composites'], cun.composites)
  8gram       not linked by a composite: >= MIN_SHARED rare sign 8-grams in common and these are >= CONTAIN
              of the smaller text's rare 8-grams (rare: in at most DF_MAX texts of the whole corpus; C14)

Alignment (`align`): whole-text sign streams, line ends ignored. A sign takes part if it is identified and
not restored by the editor (brk != 2); <x>, <gap>, <unk-sign> and restored signs match nothing. Sign
4-grams that are unique in both texts give anchor points, the longest increasing chain of them is kept
(patience), every anchor is extended to a block of identical signs, and the stretches between blocks are
aligned with difflib.

Lacuna (`extract`): a maximal run of lost tokens of A (brk == 2, <x> or <gap>; at most MAX_A tokens) that
is flanked on both sides by >= FLANK consecutive aligned identical signs, and for which B has between the corresponding anchors 1..MAX_L signs, all identified and clean.
The anchors (up to ANCHOR_GRAM signs of each flank) must be unique in both witnesses within +-WINDOW signs,
and the two signs next to the lacuna on either side must not recur inside B's answer (a shifted alignment
would then be as good).
`clean_flanks`: the FLANK signs on each side are also undamaged (brk == 0) in both witnesses: the strict
set, which `select` / `instances` return by default (clean_only=False adds lacunae whose flanking signs are
identified but marked damaged in A or B, as the signs at the edge of a break usually are).
If several witnesses B answer the same lacuna, the majority reading is the answer (ties: longest flanks);
n_wit / n_agree record how many witnesses answered and how many give exactly this reading.

File format: PROC/witness_dev.pkl.gz, PROC/witness_test.pkl.gz (pickle, gzip), a dict:
  version      corpus version;   name 'witness-dev' | 'witness-test';   params (the constants below)
  langs, sign_names, read_vocab, sign_reads     as in the corpus (and in eval3 files)
  texts        {text idx: corpus metadata dict} of every witness used
  arrays       {text idx: int32 (5, n): sign, read, join, det, brk; lines concatenated, join = LINE_END (6)
               on the last sign of a line}
  pairs        [{a, b (text idx), source, composites, shared (aligned signs), n_lac}] pairs with a lacuna
  stats        candidate pairs per source / split combination, rejection counts, ...
  lacunae      list of dicts, one per lacuna:
    id, a, b (text idx), a_id, b_id, a_split, b_split, has_test2, source, composites, same_object,
    a_is_composite / b_is_composite (the witness is an Oracc composite text, Q-number),
    a_s, a_e          the lost run in arrays[a] (token positions; the lacuna as edited is arrays[a][:, a_s:a_e])
    b_s, b_e          the answer in arrays[b];  answer: int32 (4, L) sign, read, join, det of B
    L                 number of signs of the answer
    kind              'restored' (only signs restored by the editor) | 'x' (only x) | 'ellipsis' (only '...')
                      | 'mixed';  n_restored, n_x, n_gap, n_unk (tokens of each sort);  has_gap
    a_slots           number of explicit slots in A (restored signs + x), None if A has a '...'
    width_conflict    A gives explicit slots and their number differs from L by more than max(1, 34%): B
                      probably has material that A never had (an extra entry or line, a gloss, another
                      recension); left out by `select` unless width_conflicts=True
    b_line_ends       line ends of B inside the answer
    editor            int32 signs restored by A's editor (kind == 'restored'), else None
    editor_exact      editor == answer (sign level), None if not restored;  editor_dist (Levenshtein)
    left, right       length of the aligned identical flank on each side (capped at 20)
    clean_flanks      the 3 signs on each side are undamaged in both witnesses (strict set)
    n_wit, n_agree    witnesses answering this lacuna / giving exactly this answer;  alts [(b, signs)]
    line_pos          'initial' | 'final' | 'medial' | 'whole' (a whole line) | 'multi' (crosses a line end)
    lo, hi            the chunk of A (window of <= 256 signs around the lacuna, both variants fit)
    stratum, period_bin, lang, genre_class   of A
    novel             the answer with 2 signs of context occurs nowhere in the train texts;
    regime            'parallel' (found with 4 signs of context) | 'formulaic' (with 2) | 'novel'
    cls               per answer sign: 'numeral' | 'name' | 'other' (as in cun.eval3)

Instances for the model (`instances(obj, kind)`): an eval3-style dict (version, chunks, meta, instances,
langs, sign_reads, sign_names, read_vocab) that cun.eval3.masked_input / gold / run_rows accept.
  kind = 'known'    known width: A's chunk with the lacuna replaced by exactly L slots; the stored chunk
                    carries B's signs in the slots (the gold), instance['pos'] are the slots, so
                    eval3.masked_input masks them (sign MASK, reading READ_MASK, join / det unknown) and
                    eval3.gold returns B's signs. Everything else in the chunk is A as edited.
  kind = 'edited'   for a later unknown-width evaluation: A's chunk as edited; instance['pos'] are the
                    tokens of the lacuna, 'gold' B's signs, 'a_slots', 'has_gap'; `gap_input(ev, x, M)`
                    gives the input with the lacuna replaced by M slots.

  python -m cun.witness build            (needs the corpus: ~6 GB; run on a compute node)
"""
from __future__ import annotations

import bisect
import collections
import difflib
import gzip
import os
import pickle
import re
import time

import numpy as np

from .consts import FIRST_REAL_SIGN, GAP, MASK, READ_MASK, UNK_SIGN, X_SIGN
from .devset import CHUNK, LINE_END
from .noise import D_UNK, J_UNK
from .paths import PROC

MAX_L = 20            # longest answer
MAX_A = 40            # longest lost run of A (tokens)
FLANK = 3             # aligned identical clean signs required on each side
ANCHOR_GRAM = 6       # up to this many signs of a flank must be unique ...
WINDOW = 40           # ... within this many signs on either side, in both witnesses
KGRAM = 4             # anchor points of the alignment
FILL_MAX = 20000      # difflib between blocks only if len(a) * len(b) is at most this
EDGE = 80             # stretch aligned by difflib before the first / after the last block
NGRAM, DF_MAX, CONTAIN, MIN_SHARED = 8, 20, 0.5, 8
PARAMS = dict(MAX_L=MAX_L, MAX_A=MAX_A, FLANK=FLANK, ANCHOR_GRAM=ANCHOR_GRAM, WINDOW=WINDOW, KGRAM=KGRAM,
              FILL_MAX=FILL_MAX, EDGE=EDGE, NGRAM=NGRAM, DF_MAX=DF_MAX, CONTAIN=CONTAIN, MIN_SHARED=MIN_SHARED)
HELD = ("devA", "devB", "test1", "test2")
STAGES = ("flank not aligned", "flank shorter than 3 aligned identical signs", "A's lost run longer than 40 tokens",
          "B has nothing between the anchors", "B has more than 20 signs", "B damaged or unidentified there",
          "anchor not unique in the window", "anchor recurs inside B's answer", "accepted")
NAME_DETS = {"d", "m", "f", "ki", "diš", "dingir", "munus", "uru", "kur", "id₂", "i₇", "lu₂"}
NUM = re.compile(r"^\d+(/\d+)?\(")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def path(name):
    return os.path.join(PROC, f"witness_{name}.pkl.gz")


# ------------------------------------------------------------------ texts

def nontrain(corpus):
    """{text idx: (5, n) array with line ends on the join plane} of every text that is not in train."""
    keep = {i for i, t in enumerate(corpus["texts"]) if t["split"] != "train"}
    by = {}
    for ti, a in corpus["lines"]:
        if ti in keep and a.shape[1]:
            a = a.copy()
            a[2, -1] = LINE_END
            by.setdefault(ti, []).append(a)
    return {ti: np.concatenate(v, 1) for ti, v in by.items()}


def keys(a):
    """Alignment keys of a text array: sign id, or -1 for tokens that match nothing."""
    k = a[0].astype(np.int64)
    k[(a[0] < FIRST_REAL_SIGN) | (a[4] == 2)] = -1
    return k


def gram_hashes(k, n=NGRAM):
    """uint64 hashes of the n-grams of a key sequence that contain no wildcard."""
    if len(k) < n:
        return np.zeros(0, np.uint64)
    h = np.zeros(len(k) - n + 1, np.uint64)
    bad = np.zeros(len(k) - n + 1, bool)
    for j in range(n):
        seg = k[j:len(k) - n + 1 + j]
        h = h * np.uint64(1000003) + (seg + 2).astype(np.uint64)
        bad |= seg < 0
    return np.unique(h[~bad])


def _in(h, ref):
    """h in ref (sorted unique array), by binary search."""
    if not len(ref):
        return np.zeros(len(h), bool)
    i = np.minimum(np.searchsorted(ref, h), len(ref) - 1)
    return ref[i] == h


def common_grams(corpus, cand):
    """Hashes of sign 8-grams that occur in more than DF_MAX texts of the corpus (restricted to `cand`,
    the sorted hashes that matter)."""
    by = {}
    for ti, a in corpus["lines"]:
        by.setdefault(ti, []).append(a)
    out = []
    for ti, v in by.items():
        h = gram_hashes(keys(np.concatenate(v, 1)))
        if len(h):
            out.append(h[_in(h, cand)])
    u, c = np.unique(np.concatenate(out), return_counts=True)
    return u[c > DF_MAX]


# ------------------------------------------------------------------ alignment

def _lis(js):
    """Indices of a longest strictly increasing subsequence of js."""
    tails, tail_idx, prev = [], [], [-1] * len(js)
    for n, j in enumerate(js):
        p = bisect.bisect_left(tails, j)
        if p == len(tails):
            tails.append(j)
            tail_idx.append(n)
        else:
            tails[p] = j
            tail_idx[p] = n
        prev[n] = tail_idx[p - 1] if p else -1
    out = []
    n = tail_idx[-1] if tail_idx else -1
    while n >= 0:
        out.append(n)
        n = prev[n]
    return out[::-1]


def _difflib(la, lb, i0, j0, amap):
    if not la or not lb or len(la) * len(lb) > FILL_MAX:
        return
    xa = [v if v >= 0 else -(2 + n) for n, v in enumerate(la)]
    xb = [v if v >= 0 else -(10 ** 7 + n) for n, v in enumerate(lb)]
    for i, j, l in difflib.SequenceMatcher(None, xa, xb, autojunk=False).get_matching_blocks():
        for d in range(l):
            amap[i0 + i + d] = j0 + j + d


def align(ka, kb):
    """Monotone alignment of two key sequences: amap[i] = index in kb of the sign aligned (and identical)
    to ka[i], or -1."""
    na, nb = len(ka), len(kb)
    amap = [-1] * na
    la, lb = ka.tolist(), kb.tolist()
    chain = []
    if na >= KGRAM and nb >= KGRAM:
        def uniq(k):
            g = np.zeros(len(k) - KGRAM + 1, np.int64)
            bad = np.zeros(len(g), bool)
            for j in range(KGRAM):
                seg = k[j:len(k) - KGRAM + 1 + j]
                g = g * 4096 + seg
                bad |= seg < 0
            pos = np.flatnonzero(~bad)
            u, first, c = np.unique(g[pos], return_index=True, return_counts=True)
            return u[c == 1], pos[first[c == 1]]
        ua, pa = uniq(ka)
        ub, pb = uniq(kb)
        _, xa, xb = np.intersect1d(ua, ub, assume_unique=True, return_indices=True)
        if len(xa):
            ii, jj = pa[xa], pb[xb]
            o = np.argsort(ii)
            ii, jj = ii[o].tolist(), jj[o].tolist()
            chain = [(ii[n], jj[n]) for n in _lis(jj)]
    if not chain:
        _difflib(la, lb, 0, 0, amap)
        return amap
    blocks = []
    ei = ej = 0
    for i, j in chain:
        if i < ei or j < ej:
            continue
        while i > ei and j > ej and la[i - 1] == lb[j - 1] and la[i - 1] >= 0:
            i -= 1
            j -= 1
        l = KGRAM
        while i + l < na and j + l < nb and la[i + l] == lb[j + l] and la[i + l] >= 0:
            l += 1
        blocks.append((i, j, l))
        ei, ej = i + l, j + l
    pi = pj = 0
    for n, (i, j, l) in enumerate(blocks):
        if n == 0:
            s, t = max(0, i - EDGE), max(0, j - EDGE)
            _difflib(la[s:i], lb[t:j], s, t, amap)
        else:
            _difflib(la[pi:i], lb[pj:j], pi, pj, amap)
        for d in range(l):
            amap[i + d] = j + d
        pi, pj = i + l, j + l
    _difflib(la[pi:pi + EDGE], lb[pj:pj + EDGE], pi, pj, amap)
    return amap


# ------------------------------------------------------------------ lacunae

def lost_runs(a):
    lost = (a[4] == 2) | (a[0] == X_SIGN) | (a[0] == GAP)
    d = np.diff(np.r_[0, lost.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _count(seq, lo, hi, gram):
    """Occurrences of gram (list) in seq[lo:hi]."""
    g = len(gram)
    return sum(seq[p:p + g] == gram for p in range(max(0, lo), min(len(seq), hi) - g + 1))


def extract(A, B, amap, runs=None):
    """Lacunae of A answered by B. Returns ([(s, e, bs, be, left, right, left clean, right clean)],
    {(s, e): index into STAGES}). left / right: aligned identical signs next to the run; clean: of these, the
    number directly at the run that are undamaged in both witnesses."""
    ka, kb = keys(A).tolist(), keys(B).tolist()
    ca, cb = (A[4] == 0).tolist(), (B[4] == 0).tolist()
    na = len(ka)
    ok, rej = [], {}
    for s, e in (lost_runs(A) if runs is None else runs):
        if s == 0 or e >= na or amap[s - 1] < 0 or amap[e] < 0:
            continue                                               # stage 0 is implicit
        jl, jr = amap[s - 1], amap[e]
        left = 0
        while left < 20 and s - 1 - left >= 0 and amap[s - 1 - left] == jl - left:
            left += 1
        right = 0
        while right < 20 and e + right < na and amap[e + right] == jr + right:
            right += 1
        lc = 0
        while lc < left and ca[s - 1 - lc] and cb[jl - lc]:
            lc += 1
        rc = 0
        while rc < right and ca[e + rc] and cb[jr + rc]:
            rc += 1
        if left < FLANK or right < FLANK:
            rej[(s, e)] = 1
            continue
        if e - s > MAX_A:
            rej[(s, e)] = 2
            continue
        bs, be = jl + 1, jr
        if be <= bs:
            rej[(s, e)] = 3
            continue
        if be - bs > MAX_L:
            rej[(s, e)] = 4
            continue
        if not all(kb[j] >= 0 and cb[j] for j in range(bs, be)):
            rej[(s, e)] = 5
            continue
        gl, gr = min(left, ANCHOR_GRAM), min(right, ANCHOR_GRAM)
        Lg, Rg = ka[s - gl:s], ka[e:e + gr]
        if (_count(ka, s - WINDOW, e + WINDOW, Lg) != 1 or _count(ka, s - WINDOW, e + WINDOW, Rg) != 1
                or _count(kb, bs - WINDOW, be + WINDOW, Lg) != 1 or _count(kb, bs - WINDOW, be + WINDOW, Rg) != 1):
            rej[(s, e)] = 6
            continue
        span = kb[bs - 2:be + 2]
        if _count(span, 0, len(span), ka[s - 2:s]) != 1 or _count(span, 0, len(span), ka[e:e + 2]) != 1:
            rej[(s, e)] = 7                                        # e.g. ME ME [...] against ME ME E ME ME I
            continue
        ok.append((s, e, bs, be, left, right, lc, rc))
    return ok, rej


G = {}


def _pair(job):
    ia, ib = job
    A, B = G["arr"][ia], G["arr"][ib]
    ka, kb = keys(A), keys(B)
    amap = align(ka, kb)
    shared = sum(v >= 0 for v in amap)
    if shared < 2 * FLANK:
        return ia, ib, shared, [], {}, [], {}
    bmap = [-1] * len(kb)
    for i, j in enumerate(amap):
        if j >= 0:
            bmap[j] = i
    oa, ra = extract(A, B, amap)
    ob, rb = extract(B, A, bmap)
    return ia, ib, shared, oa, ra, ob, rb


def line_pos(A, s, e):
    ini = s == 0 or A[2, s - 1] == LINE_END
    fin = A[2, e - 1] == LINE_END
    if (A[2, s:e - 1] == LINE_END).any():
        return "multi"
    return "whole" if ini and fin else "initial" if ini else "final" if fin else "medial"


def chunk_window(A, s, e, L):
    """Window [lo, hi) of A of at most CHUNK signs around the lacuna, such that it fits both as edited and
    with L slots; starts at a line start where one lies at least 16 signs before the lacuna."""
    n = A.shape[1]
    budget = CHUNK - max(L, e - s)
    lo = max(0, s - budget // 2)
    hi = min(n, e + budget - (s - lo))
    lo = max(0, s - (budget - (hi - e)))
    if lo > 0:
        st = np.flatnonzero(A[2, lo - 1:max(lo, s - 16)] == LINE_END)
        if len(st):
            lo = lo + int(st[0])
    return lo, hi


def known_chunk(A, r):
    """A's chunk with the lacuna replaced by the answer of B (the slots); returns (array, slot start)."""
    s, e, lo, hi = r["a_s"], r["a_e"], r["lo"], r["hi"]
    L = r["L"]
    slot = np.zeros((5, L), np.int32)
    slot[:4] = r["answer"]
    slot[2, slot[2] == LINE_END] = 0
    if A[2, e - 1] == LINE_END:
        slot[2, -1] = LINE_END
    elif slot[2, -1] == LINE_END:
        slot[2, -1] = 0
    return np.concatenate([A[:, lo:s], slot, A[:, e:hi]], 1), s - lo


def slot_classes(a, pos, RV):
    """'numeral' | 'name' | 'other' per position, as in cun.eval3.build."""
    n = a.shape[1]
    wend = np.flatnonzero(np.isin(a[2], (0, LINE_END)))
    if not len(wend) or wend[-1] != n - 1:
        wend = np.r_[wend, n - 1]
    wid = np.searchsorted(wend, np.arange(n))
    named = np.zeros(len(wend), bool)
    for i in np.flatnonzero(a[3] != 0):
        if RV[a[1, i]] in NAME_DETS:
            named[wid[i]] = True
    return np.array(["numeral" if NUM.match(RV[a[1, p]]) else "name" if named[wid[p]] else "other" for p in pos])


# ------------------------------------------------------------------ build

def sides(texts, arr, common):
    """Which set an excluded text may serve ('dev' | 'test'), the composite pairs and the 8-gram pairs."""
    ids = sorted(arr)
    byq, bym = collections.defaultdict(list), collections.defaultdict(list)
    for i in ids:
        for q in texts[i]["composites"]:
            byq[q].append(i)
        for m in texts[i]["museum"]:
            bym[m].append(i)
    link = collections.defaultdict(set)            # text -> splits of held-out texts it is linked to
    for grp in list(byq.values()) + [v for v in bym.values() if len(v) <= 10]:
        sp = {texts[i]["split"] for i in grp} & set(HELD)
        for i in grp:
            link[i] |= sp
    # rare 8-grams
    hs, ts, nrare = [], [], {}
    for n, i in enumerate(ids):
        h = gram_hashes(keys(arr[i]))
        h = h[~_in(h, common)]
        nrare[i] = len(h)
        hs.append(h)
        ts.append(np.full(len(h), n, np.int64))
    hs, ts = np.concatenate(hs), np.concatenate(ts)
    o = np.argsort(hs, kind="stable")
    hs, ts = hs[o], ts[o]
    cut = np.flatnonzero(np.diff(hs) != 0) + 1
    cnt = collections.Counter()
    for g in np.split(ts, cut):
        if 2 <= len(g) <= DF_MAX:
            g = g.tolist()
            for x in range(len(g)):
                for y in range(x + 1, len(g)):
                    cnt[(g[x], g[y])] += 1
    near = {}
    for (x, y), c in cnt.items():
        a, b = ids[x], ids[y]
        if c >= MIN_SHARED and c >= CONTAIN * min(nrare[a], nrare[b]):
            near[(a, b)] = c / min(nrare[a], nrare[b])
            for u, v in ((a, b), (b, a)):
                if texts[v]["split"] in HELD:
                    link[u].add(texts[v]["split"])
    side = {}
    for i in ids:
        sp = texts[i]["split"]
        side[i] = {"devA": "dev", "devB": "dev", "test2": "test", "test1": None}.get(sp) if sp != "excluded" else \
            ("test" if "test2" in link[i] else "dev")
    comp_pairs = collections.defaultdict(list)
    for q, grp in byq.items():
        for x in range(len(grp)):
            for y in range(x + 1, len(grp)):
                comp_pairs[(grp[x], grp[y])].append(q)
    return side, link, comp_pairs, near


def build(corpus, arr=None, common=None, workers=8, novelty=True):
    import multiprocessing as mp
    from rapidfuzz.distance import Levenshtein
    texts = corpus["texts"]
    RV = corpus["read_vocab"]
    if arr is None:
        arr = nontrain(corpus)
    if common is None:
        cand = np.unique(np.concatenate([gram_hashes(keys(a)) for a in arr.values()]))
        common = common_grams(corpus, cand)
        log("8-grams of non-train texts", len(cand), "common (in more than", DF_MAX, "texts)", len(common))
    side, link, comp_pairs, near = sides(texts, arr, common)
    excl = [i for i in arr if texts[i]["split"] == "excluded"]
    st = {"excluded_side": dict(collections.Counter(side[i] for i in excl)),
          "excluded_links": dict(collections.Counter("+".join(sorted(link[i])) or "none" for i in excl)),
          "candidate_pairs": {}, "dropped_pairs": collections.Counter()}
    log("excluded texts by side", st["excluded_side"], "links", st["excluded_links"])
    jobs = {}
    for src, pairs in (("composite", comp_pairs), ("8gram", near)):
        for (a, b) in pairs:
            if (a, b) in jobs:
                continue
            if side[a] is None or side[b] is None:
                st["dropped_pairs"][src + ": involves a test1 text"] += 1
            elif side[a] != side[b]:
                st["dropped_pairs"][src + ": dev-side with test-side text"] += 1
            else:
                jobs[(a, b)] = src
    log("candidate pairs", len(jobs), dict(collections.Counter(jobs.values())), "dropped", dict(st["dropped_pairs"]))
    G["arr"] = arr
    order = sorted(jobs, key=lambda p: -arr[p[0]].shape[1] * arr[p[1]].shape[1])
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_pair, order, chunksize=64)
    log("aligned")

    def cls(a, b):
        return "+".join(sorted((texts[a]["split"], texts[b]["split"])))
    out = {}
    for name in ("dev", "test"):
        cands = collections.defaultdict(list)             # (a, s, e) -> [(b, bs, be, left, right)]
        stage = collections.defaultdict(int)              # (a, s, e) -> furthest stage
        pstat = collections.defaultdict(lambda: collections.Counter())
        shared = {}
        used = set()
        for ia, ib, sh, oa, ra, ob, rb in res:
            if side[ia] != name:
                continue
            src = jobs[(ia, ib)]
            k = (src, cls(ia, ib))
            pstat[k]["pairs"] += 1
            pstat[k]["aligned (>= 6 signs)"] += sh >= 2 * FLANK
            pstat[k]["with lacuna"] += bool(oa or ob)
            shared[(ia, ib)] = shared[(ib, ia)] = sh
            used |= {ia, ib}
            for a, b, o, r in ((ia, ib, oa, ra), (ib, ia, ob, rb)):
                for s, e, bs, be, l, rr, lc, rc in o:
                    cands[(a, s, e)].append((b, bs, be, l, rr, min(lc, rc) >= FLANK))
                for (s, e), v in r.items():
                    stage[(a, s, e)] = max(stage[(a, s, e)], v)
        n_runs = sum(len(lost_runs(arr[i])) for i in used)
        rejc = collections.Counter(STAGES[v] for k, v in stage.items() if k not in cands)
        rejc[STAGES[0]] = n_runs - len(cands) - sum(rejc.values())
        rejc[STAGES[-1]] = len(cands)
        lac = []
        for (a, s, e), cs in sorted(cands.items()):
            A = arr[a]
            ans = collections.defaultdict(list)
            for c in cs:
                ans[tuple(arr[c[0]][0, c[1]:c[2]].tolist())].append(c)
            best = max(ans.values(), key=lambda v: (len(v), max(c[5] for c in v), max(c[3] + c[4] for c in v)))
            b, bs, be, left, right, clean = max(best, key=lambda c: (c[5], c[3] + c[4], texts[c[0]]["split"] != "excluded", -c[0]))
            B = arr[b]
            L = be - bs
            tok = A[:, s:e]
            n_gap = int((tok[0] == GAP).sum())
            n_x = int((tok[0] == X_SIGN).sum())
            n_unk = int((tok[0] == UNK_SIGN).sum())
            n_res = int((tok[0] >= FIRST_REAL_SIGN).sum())
            kind = "restored" if n_res == e - s else "x" if n_x == e - s else "ellipsis" if n_gap == e - s else "mixed"
            ed = tok[0].copy() if kind == "restored" else None
            ta, tb = texts[a], texts[b]
            lo, hi = chunk_window(A, s, e, L)
            r = {"a": a, "b": b, "a_id": ta["id"], "b_id": tb["id"], "a_split": ta["split"], "b_split": tb["split"],
                 "has_test2": "test2" in (ta["split"], tb["split"]), "source": jobs.get((a, b)) or jobs[(b, a)],
                 "composites": sorted(set(ta["composites"]) & set(tb["composites"])),
                 "same_object": bool(set(ta["museum"]) & set(tb["museum"])) or (ta["cdli"] is not None and ta["cdli"] == tb["cdli"]),
                 "a_is_composite": ta["id"].startswith("Q"), "b_is_composite": tb["id"].startswith("Q"),
                 "a_s": s, "a_e": e, "b_s": bs, "b_e": be, "answer": B[:4, bs:be].copy(), "L": L,
                 "kind": kind, "n_restored": n_res, "n_x": n_x, "n_gap": n_gap, "n_unk": n_unk, "has_gap": n_gap > 0,
                 "a_slots": None if n_gap else e - s,
                 "width_conflict": bool(not n_gap and abs(e - s - L) > max(1, 0.34 * max(e - s, L))),
                 "b_line_ends": int((B[2, bs:be - 1] == LINE_END).sum()), "editor": ed,
                 "editor_exact": None if ed is None else bool(ed.tolist() == B[0, bs:be].tolist()),
                 "editor_dist": None if ed is None else int(Levenshtein.distance(ed.tolist(), B[0, bs:be].tolist())),
                 "left": left, "right": right, "clean_flanks": bool(clean), "n_wit": len(cs), "n_agree": len(best),
                 "alts": [(c[0], arr[c[0]][0, c[1]:c[2]].copy()) for v in ans.values() if v is not best for c in v[:1]],
                 "line_pos": line_pos(A, s, e), "lo": lo, "hi": hi,
                 "stratum": ta.get("stratum"), "period_bin": ta.get("period_bin"), "lang": ta["lang"],
                 "genre_class": ta.get("genre_class")}
            ch, g = known_chunk(A, r)
            r["cls"] = slot_classes(ch, np.arange(g, g + L), RV)
            lac.append(r)
        for n, r in enumerate(lac):
            r["id"] = n
        if novelty and lac:
            from . import evalset
            chunks = {n: (r["a"], 0, known_chunk(arr[r["a"]], r)[0]) for n, r in enumerate(lac)}
            inst = [{"chunk": n, "start": r["a_s"] - r["lo"], "L": r["L"]} for n, r in enumerate(lac)]
            n2 = evalset.novelty(corpus, chunks, inst, ctx=2)
            n4 = evalset.novelty(corpus, chunks, inst, ctx=4)
            for r, v2, v4 in zip(lac, n2, n4):
                r["novel"] = bool(v2)
                r["regime"] = "novel" if v2 else "formulaic" if v4 else "parallel"
        pl = collections.Counter((r["a"], r["b"]) for r in lac)
        tx = sorted({r["a"] for r in lac} | {r["b"] for r in lac})
        stats = {**st, "candidate_pairs": {f"{k[0]}: {k[1]}": dict(v) for k, v in sorted(pstat.items())},
                 "lost_runs_in_paired_texts": n_runs, "runs": dict(rejc), "dropped_pairs": dict(st["dropped_pairs"])}
        obj = {"version": corpus["version"], "name": f"witness-{name}", "params": PARAMS, "langs": corpus["langs"],
               "sign_names": corpus["sign_names"], "read_vocab": RV, "sign_reads": corpus["sign_reads"],
               "texts": {i: texts[i] for i in tx}, "arrays": {i: arr[i] for i in tx},
               "pairs": [{"a": a, "b": b, "source": jobs.get((a, b)) or jobs[(b, a)],
                          "composites": sorted(set(texts[a]["composites"]) & set(texts[b]["composites"])),
                          "shared": shared[(a, b)], "n_lac": n} for (a, b), n in sorted(pl.items())],
               "stats": stats, "lacunae": lac}
        with gzip.open(path(name), "wb") as fh:
            pickle.dump(obj, fh, protocol=5)
        log(obj["name"], "lacunae", len(lac), "pairs", len(pl), "runs", dict(rejc))
        for k, v in stats["candidate_pairs"].items():
            log("   ", k, v)
        out[name] = obj
    return out


# ------------------------------------------------------------------ use

def load(name, version=None):
    """name: 'dev' | 'test'."""
    obj = pickle.load(gzip.open(path(name), "rb"))
    if version is not None:
        from .devset import check_version
        check_version(version, obj["version"], f"witness_{name}")
    return obj


def select(obj, strict=True, clean_only=True, width_conflicts=False, where=None, min_flank=FLANK, agree_only=False, max_per_composite=None, seed=0):
    """Lacunae of a set: strict = only pairs with a held-out (dev: any; test: a test2) text as documented
    (drops the has_test2 == False items of witness-test); clean_only: the strict set (clean flanks);
    width_conflicts: keep lacunae whose explicit width in A contradicts B; where: a predicate on the
    lacuna record; min_flank: both flanks at least this long;
    agree_only: drop lacunae on which the witnesses disagree; max_per_composite: at most this many lacunae
    per composite (or per pair, for pairs without a composite), drawn at random with `seed`."""
    lac = [r for r in obj["lacunae"] if min(r["left"], r["right"]) >= min_flank and (r["clean_flanks"] or not clean_only)
           and (width_conflicts or not r["width_conflict"]) and (where is None or where(r))
           and not (strict and obj["name"] == "witness-test" and not r["has_test2"])
           and not (agree_only and r["n_agree"] < r["n_wit"])]
    if max_per_composite:
        rng = np.random.default_rng(seed)
        grp = collections.defaultdict(list)
        for r in lac:
            grp[r["composites"][0] if r["composites"] else (min(r["a"], r["b"]), max(r["a"], r["b"]))].append(r)
        keep = set()
        for v in grp.values():
            keep |= {v[i]["id"] for i in rng.permutation(len(v))[:max_per_composite]}
        lac = [r for r in lac if r["id"] in keep]
    return lac


def instances(obj, kind="known", **sel):
    """eval3-style evaluation dict of a witness set (see the module docstring). kind: 'known' | 'edited'.
    Keyword arguments as in `select`."""
    langs = {l: i for i, l in enumerate(obj["langs"])}
    chunks, meta, inst = {}, {}, []
    for r in select(obj, **sel):
        A = obj["arrays"][r["a"]]
        ci = r["id"]
        base = {"chunk": ci, "text": r["a"], "type": "witness", "lac": r["id"], "b_text": r["b"], "kind": r["kind"],
                "a_slots": r["a_slots"], "has_gap": r["has_gap"], "novel": r.get("novel"), "L": r["L"],
                "regime": [r.get("regime", "novel")], "cls": r["cls"],
                "alts": [np.asarray(v, np.int64) for _, v in r["alts"]]}
        if kind == "known":
            ch, g = known_chunk(A, r)
            x = {**base, "pos": np.arange(g, g + r["L"], dtype=np.int32), "runs": [(g, r["L"])]}
        else:
            ch, g = A[:, r["lo"]:r["hi"]], r["a_s"] - r["lo"]
            n = r["a_e"] - r["a_s"]
            x = {**base, "pos": np.arange(g, g + n, dtype=np.int32), "runs": [(g, n)], "unknown_width": True,
                 "gold": r["answer"][0].astype(np.int64)}
        chunks[ci] = (r["a"], langs[r["lang"]], ch)
        meta[ci] = {"stratum": r["stratum"], "period_bin": r["period_bin"], "lang": r["lang"],
                    "src": obj["texts"][r["a"]]["src"], "id": r["a_id"]}
        inst.append(x)
    return {"version": obj["version"], "split": obj["name"], "kind": kind, "chunks": chunks, "meta": meta,
            "instances": inst, "langs": obj["langs"], "sign_reads": obj["sign_reads"], "sign_names": obj["sign_names"],
            "read_vocab": obj["read_vocab"]}


def gap_input(ev, x, M):
    """Input planes of an instance with its gap replaced by M masked slots (unknown-width decoding of an
    'edited' instance; with M = L on a 'known' instance it equals cun.eval3.masked_input)."""
    ti, lg, a = ev["chunks"][x["chunk"]]
    s, e = int(x["pos"][0]), int(x["pos"][-1]) + 1
    planes = {}
    for k, row, fill in (("sign", 0, MASK), ("read", 1, READ_MASK), ("join", 2, J_UNK), ("det", 3, D_UNK)):
        v = a[row].astype(np.int64)
        planes[k] = np.r_[v[:s], np.full(M, fill, np.int64), v[e:]]
    st = a[4].astype(np.int64).copy()
    st[a[0] < FIRST_REAL_SIGN] = 3
    planes["status"] = np.r_[st[:s], np.full(M, 3, np.int64), st[e:]]
    return {**planes, "lang": lg, "gap": s, "M": M}


def render(obj, a, mark=None):
    """Transliteration of a (>= 4, n) array: reading, else sign name; [ ] around restored signs, x, ...;
    '#' after damaged signs; ' / ' at line ends; mark = (s, e) puts <<< >>> around that stretch."""
    JS = {0: " ", 1: "-", 2: ".", 3: ":", 4: "+", 5: "", 6: " / "}
    out = []
    n = a.shape[1]
    for i in range(n):
        s, r = int(a[0, i]), int(a[1, i])
        tok = "x" if s == X_SIGN else "..." if s == GAP else obj["read_vocab"][r] if r > 1 else obj["sign_names"][s]
        if a[3, i] in (1, 2):
            tok = "{" + tok + "}"
        if a.shape[0] > 4:
            if a[4, i] == 2:
                tok = "[" + tok + "]"
            elif a[4, i] == 1:
                tok += "#"
        if mark and i == mark[0]:
            out.append("<<< ")
        out.append(tok)
        if mark and i == mark[1] - 1:
            out.append(" >>>")
        if i < n - 1 or int(a[2, i]) == LINE_END:
            out.append(JS.get(int(a[2, i]), " "))
    return "".join(out).strip()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build",))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    log("corpus", corpus["version"])
    build(corpus, workers=a.workers)


if __name__ == "__main__":
    main()
