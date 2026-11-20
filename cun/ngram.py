"""Non-neural sign-level models (torch-free), shared by the dev baselines and the gap evaluation.

WB          Witten-Bell interpolated n-gram over int tokens
fill_runs   fill every maximal run of MASK in a sign sequence with beam search under a sign n-gram;
            the last sign of a run is also scored against the known signs to its right
Translit    reading statistics from the train chunks:
              mf(sign)            most frequent reading
              ctx(signs)          most frequent reading given (prev, sign, next) with backoff
              hmm(signs, tgt)     trigram HMM over readings, Viterbi
              join/det of a (sign, reading): most frequent, for rendering n-gram restorations
"""
from __future__ import annotations

import collections

import numpy as np

from .consts import MASK, READ_NONE

FIRST_REAL = 6


class WB:
    """Witten-Bell interpolated n-gram over int tokens < base; key(w1..wn) = (..(w1*base+w2)..)*base+wn."""

    def __init__(self, seqs, order, base, bos):
        assert bos < base and base ** order < 2 ** 63, (base, order)
        assert all((len(x) == 0) or int(np.max(x)) < base for x in seqs), "token id >= base"
        self.order, self.base, self.bos = order, base, bos
        pad = [np.r_[np.full(order - 1, bos, np.int64), s.astype(np.int64)] for s in seqs]
        self.keys, self.cnt, self.hkeys, self.hc, self.ht = {}, {}, {}, {}, {}
        for n in range(1, order + 1):
            ks = []
            for s in pad:
                if len(s) < n:
                    continue
                k = np.zeros(len(s) - n + 1, np.int64)
                for j in range(n):
                    k = k * base + s[j:len(s) - n + 1 + j]
                ks.append(k)
            u, c = np.unique(np.concatenate(ks), return_counts=True)
            self.keys[n], self.cnt[n] = u, c
            if n >= 2:
                h = u // base
                hu, first, T = np.unique(h, return_index=True, return_counts=True)
                self.hkeys[n], self.ht[n] = hu, T
                self.hc[n] = np.add.reduceat(c, first)
        c1 = np.zeros(base, np.float64)
        c1[self.keys[1]] = self.cnt[1]
        self.p1 = (c1 + 1.0) / (c1.sum() + base)

    @staticmethod
    def _look(keys, vals, q):
        i = np.searchsorted(keys, q)
        i = np.minimum(i, len(keys) - 1)
        ok = keys[i] == q
        return np.where(ok, vals[i], 0)

    def logp(self, ctx, w):
        """ctx: (C, order-1) int array of preceding tokens; w: (C,) -> log P(w | ctx)."""
        p = self.p1[w]
        for n in range(2, self.order + 1):
            h = np.zeros(len(w), np.int64)
            for j in range(self.order - n, self.order - 1):
                h = h * self.base + ctx[:, j]
            ch = self._look(self.hkeys[n], self.hc[n], h).astype(np.float64)
            th = self._look(self.hkeys[n], self.ht[n], h).astype(np.float64)
            chw = self._look(self.keys[n], self.cnt[n], h * self.base + w).astype(np.float64)
            p = np.where(ch > 0, (chw + th * p) / np.maximum(ch + th, 1), p)
        return np.log(p)


def fill_runs(lm, cands, seq, beam=8, topk=0):
    """Fill MASK runs of seq (left to right). Returns (filled seq, {run start: top-k candidates} for
    single-sign runs when topk > 0)."""
    o = lm.order
    seq = np.asarray(seq, np.int64).copy()
    n = len(seq)
    ranked = {}
    i = 0
    while i < n:
        if seq[i] != MASK:
            i += 1
            continue
        j = i
        while j < n and seq[j] == MASK:
            j += 1
        left = np.r_[np.full(o - 1, lm.bos, np.int64), seq[:i]][-(o - 1):]
        right = []
        for k in range(j, min(n, j + o - 1)):
            if seq[k] == MASK:
                break
            right.append(seq[k])
        beams = [(0.0, list(left))]
        for pos in range(i, j):
            allc = []
            for sc, h in beams:
                ctx = np.tile(np.array(h[-(o - 1):], np.int64), (len(cands), 1))
                s = sc + lm.logp(ctx, cands)
                if pos == j - 1 and right:
                    hist = np.tile(np.array(h[-(o - 2):], np.int64), (len(cands), 1))
                    hist = np.c_[hist, cands]
                    for r in right:
                        s = s + lm.logp(hist[:, -(o - 1):], np.full(len(cands), r, np.int64))
                        hist = np.c_[hist, np.full(len(cands), r, np.int64)]
                if topk and j - i == 1:
                    ranked[i] = cands[np.argsort(-s)[:topk]]
                top = np.argpartition(-s, beam)[:beam]
                for t in top:
                    allc.append((float(s[t]), h + [int(cands[t])]))
            allc.sort(key=lambda x: -x[0])
            beams = allc[:beam]
        seq[i:j] = beams[0][1][len(left):]
        i = j
    return seq, ranked


class Translit:
    def __init__(self, train_chunks, n_sign, n_read, sign_reads, max_states=20):
        S, NR = n_sign, n_read
        self.S, self.NR = S, NR
        c1 = collections.defaultdict(collections.Counter)
        c3 = collections.defaultdict(collections.Counter)
        cl = collections.defaultdict(collections.Counter)
        cr = collections.defaultdict(collections.Counter)
        cj = collections.defaultdict(collections.Counter)
        cd = collections.defaultdict(collections.Counter)
        toks = []
        for sg, rd, jn, dt, *_ in train_chunks:
            sg, rd, jn, dt = sg.tolist(), rd.tolist(), jn.tolist(), dt.tolist()
            n = len(sg)
            t = []
            for i in range(n):
                s, r = sg[i], rd[i]
                cj[(s, r)][jn[i]] += 1
                cd[(s, r)][dt[i]] += 1
                if r != READ_NONE:
                    p = sg[i - 1] if i else S
                    q = sg[i + 1] if i + 1 < n else S
                    c1[s][r] += 1
                    c3[(p, s, q)][r] += 1
                    cl[(p, s)][r] += 1
                    cr[(s, q)][r] += 1
                    t.append(r)
                else:
                    t.append(NR + s)
            toks.append(np.array(t, np.int64))

        def best(cn):
            return {k: (v.most_common(1)[0][0], sum(v.values())) for k, v in cn.items()}
        self.c1 = c1
        self._mf = {s: v.most_common(1)[0][0] for s, v in c1.items()}
        self.ctx3, self.ctxl, self.ctxr = best(c3), best(cl), best(cr)
        self._join = {k: v.most_common(1)[0][0] for k, v in cj.items()}
        self._det = {k: v.most_common(1)[0][0] for k, v in cd.items()}
        self.states = {int(s): [r for r, _ in c1[int(s)].most_common() if r in set(rs)][:max_states]
                       for s, rs in sign_reads.items()}
        rc = collections.Counter()
        for s, v in c1.items():
            rc.update(v)
        self.emis = {(r, s): float(np.log(c / rc[r])) for s, v in c1.items() for r, c in v.items()}
        self.lm = WB(toks, 3, 16384, NR + S + 1)

    def polyvalent(self, s):
        v = self.c1.get(int(s))
        return bool(v) and len(v) > 1 and v.most_common(1)[0][1] / sum(v.values()) < 0.9

    def mf(self, s):
        return self._mf.get(int(s), READ_NONE)

    def join(self, s, r):
        return self._join.get((int(s), int(r)), 0)

    def det(self, s, r):
        return self._det.get((int(s), int(r)), 0)

    def ctx(self, sign):
        S = self.S
        pc = np.r_[S, sign[:-1]]
        nc = np.r_[sign[1:], S]
        out = []
        for p, s, q in zip(pc, sign, nc):
            k3 = (int(p), int(s), int(q))
            if k3 in self.ctx3:
                out.append(self.ctx3[k3][0])
                continue
            l, r = self.ctxl.get((int(p), int(s))), self.ctxr.get((int(s), int(q)))
            if l and r:
                out.append(l[0] if l[1] >= r[1] else r[0])
            elif l or r:
                out.append((l or r)[0])
            else:
                out.append(self.mf(s))
        return np.array(out, np.int64)

    def hmm(self, sign, tgt):
        """Viterbi readings for every position; positions with tgt False enter as sign tokens."""
        lm, NR = self.lm, self.NR
        n = len(sign)
        cand = []
        for i in range(n):
            s = int(sign[i])
            if tgt[i]:
                c = self.states.get(s)
                cand.append(np.array(c if c else [self._mf.get(s, NR + s)], np.int64))
            else:
                cand.append(np.array([NR + s], np.int64))
        delta = np.zeros((1, 1))
        hist = [np.array([lm.bos]), np.array([lm.bos])]
        back = []
        for i in range(n):
            A, B, C = hist[-2], hist[-1], cand[i]
            aa, bb, cc = np.meshgrid(np.arange(len(A)), np.arange(len(B)), np.arange(len(C)), indexing="ij")
            ctx = np.c_[A[aa.ravel()], B[bb.ravel()]]
            lp = lm.logp(ctx, C[cc.ravel()]).reshape(len(A), len(B), len(C))
            e = np.array([self.emis.get((int(c), int(sign[i])), 0.0) if c < NR else 0.0 for c in C])
            tot = delta[:, :, None] + lp + e[None, None, :]
            back.append(tot.argmax(0))
            delta = tot.max(0)
            hist.append(C)
        b, c = np.unravel_index(int(delta.argmax()), delta.shape)
        path = [c]
        for i in range(n - 1, 0, -1):
            a = back[i][b, c]
            path.append(b)
            b, c = a, b
        path = path[::-1]
        out = np.array([cand[i][path[i]] for i in range(n)], np.int64)
        return np.where(out >= NR, READ_NONE, out)
