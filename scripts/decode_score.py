"""CPU side of the decoding experiments: scores the raw outputs of scripts/decode_run.py on the dev gaps,
tunes every free choice on half A, reports half A and half B, writes results/decode/REPORT.md
(+ scores.json), and (--write-plan) the plan of the final GPU run (results/decode/final_plan.pkl.gz).

  python scripts/decode_score.py [--write-plan]

Halves: cun.decode.half(text id) ('A' tuning, 'B' confirmation). Metrics as scripts/score_gaps.py: exact
(whole gap), SER (sign Levenshtein / L), CER of the rendered transliteration; exact@k = gold among the k
best whole-gap candidates.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import pickle
import sys

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import decode as D  # noqa: E402
from cun import evalset  # noqa: E402

OUT = os.path.join(ROOT, "results", "decode")
BINS = [("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-15", 11, 15), ("16-20", 16, 20)]
COV = (100, 90, 80, 70, 60, 50, 40, 30, 20, 10)


def load(name):
    p = os.path.join(OUT, name)
    return pickle.load(gzip.open(p, "rb")) if os.path.exists(p) else None


class Ctx:
    def __init__(self):
        self.ev = ev = evalset.load(ROOT, "dev", version=evalset.corpus_version(ROOT))
        assert ev["split"] == "dev"
        self.inst = inst = ev["instances"]
        self.N = len(inst)
        self.L = np.array([x["L"] for x in inst])
        self.text = np.array([ev["chunks"][x["chunk"]][0] for x in inst])
        self.half = np.array([D.half(t) for t in self.text])
        self.A, self.B = self.half == "A", self.half == "B"
        self.gold = [evalset.gold(ev, x) for x in inst]
        self.gkey = [D.key(g[0]) for g in self.gold]
        self.gstr = [evalset.render(ev, *g) for g in self.gold]
        self.lines, self.js, self.cache = [], {}, {}

    def metric(self, i, sign, planes):
        """(exact, ser, cer) of one prediction for instance i."""
        ck = (i, D.key(sign), planes[0].tobytes(), planes[1].tobytes(), planes[2].tobytes())
        if ck not in self.cache:
            self.cache[ck] = self._metric(i, sign, planes)
        return self.cache[ck]

    def _metric(self, i, sign, planes):
        gs = self.gold[i][0].tolist()
        ps = [int(v) for v in sign]
        p_str = evalset.render(self.ev, sign, *planes)
        return (float(ps == gs), Levenshtein.distance(ps, gs) / len(gs),
                Levenshtein.distance(p_str, self.gstr[i]) / max(1, len(self.gstr[i])))

    def metric_pred(self, i, p):
        return self.metric(i, p["sign"], (p["read"], p["join"], p["det"]))

    def add(self, *l):
        self.lines.extend(l)

    def table(self, head, rows):
        self.add("", "| " + " | ".join(head) + " |", "|" + "---|" * len(head))
        for r in rows:
            self.add("| " + " | ".join(r if isinstance(r, str) else (f"{r:.1f}" if isinstance(r, float) else str(r))
                                       for r in r) + " |")
        self.add("")


def mean(v, m):
    return 100 * float(np.mean(np.asarray(v, float)[m])) if m.any() else float("nan")


def boot(d, cl, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    s, c = np.bincount(inv, d), np.bincount(inv)
    k = rng.integers(0, len(u), (n, len(u)))
    dr = s[k].sum(1) / c[k].sum(1)
    return 100 * d.mean(), 100 * np.percentile(dr, 2.5), 100 * np.percentile(dr, 97.5)


def auroc(conf, ok):
    conf, ok = np.asarray(conf, float), np.asarray(ok, bool)
    if ok.all() or not ok.any():
        return float("nan")
    u, inv, cnt = np.unique(conf, return_inverse=True, return_counts=True)
    rk = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = ok.sum(), (~ok).sum()
    return 100 * float((rk[ok].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.sum() * abs(conf[m].mean() - ok[m].mean())
    return 100 * e / len(conf)


def risk_cov(conf, val):
    """Mean of val over the most confident c% (c in COV)."""
    o = np.argsort(-np.asarray(conf, float), kind="stable")
    v = np.asarray(val, float)[o]
    return [100 * float(v[:max(1, int(round(len(v) * c / 100)))].mean()) for c in COV]


# ----------------------------------------------------------------------------- systems as per-instance arrays
def sys_from_preds(C, preds, cands=None):
    """dict of per-instance arrays ex1 / ser / cer (+ ex5, ex20 when candidate lists are given)."""
    m = np.array([C.metric_pred(i, p) for i, p in enumerate(preds)])
    d = {"ex1": m[:, 0], "ser": m[:, 1], "cer": m[:, 2]}
    if cands is not None:
        for k in (5, 20):
            d[f"ex{k}"] = np.array([float(C.gkey[i] in c[:k]) for i, c in enumerate(cands)])
    return d


def sys_from_ranked(C, K, ranked):
    """ranked: per instance the candidate keys best first (planes from the known-stage file K)."""
    m = np.array([C.metric(i, D.unkey(r[0]), K["planes"][i][r[0]]) for i, r in enumerate(ranked)])
    d = {"ex1": m[:, 0], "ser": m[:, 1], "cer": m[:, 2]}
    for k in (5, 20):
        d[f"ex{k}"] = np.array([float(C.gkey[i] in r[:k]) for i, r in enumerate(ranked)])
    return d


def row(C, name, d, cols=("ex1", "ex5", "ex20", "ser", "cer"), extra=()):
    r = [name]
    for hm in (C.A, C.B):
        r += [mean(d[c], hm) if c in d else "" for c in cols]
    return r + list(extra)


HEAD = ["exact", "exact@5", "exact@20", "SER", "CER"]


def head2(first, cols=HEAD, extra=()):
    return [first] + [f"A {c}" for c in cols] + [f"B {c}" for c in cols] + list(extra)


# ----------------------------------------------------------------------------- 1 beam
def sec_beam(C, K, cur):
    C.add("## 1. Beam search over the iterative fill (known width)", "",
          f"Half A n = {int(C.A.sum())}, half B n = {int(C.B.sum())} gaps (L = 1..20, about 150 per L and half). "
          "`iter` = the current decoder (cun.evaluate, file results/gaps/base-long__dev); `beam B` = "
          "cun.decode.beam_search, B hypotheses, each expanded at its most confident open slot with that "
          "slot's B best signs; `s2` / `s3` = expanded at its 2 / 3 most confident slots. Score = sum of "
          "log-probabilities of the choices. All hypotheses of a gap have the same number of choices, so the "
          "length-normalised score (sum / L) gives the same ranking as the unnormalised one: no difference by "
          "construction within a known width (normalisation matters only between widths, section 4, and for "
          "abstention, section 5). exact@k: the true fill is among the k best candidates. "
          "Cost = forward rows per gap (iter = L).")
    S = {}
    rows = []
    if cur is not None:
        for nm in ("indep", "iter"):
            S["cur_" + nm] = sys_from_preds(C, cur["known/" + nm])
            rows.append(row(C, f"current {nm}", S["cur_" + nm], extra=[""]))
    for nm in K["beams"]:
        ranked = [[h[0] for h in H] for H in K["beams"][nm]]
        S[nm] = sys_from_ranked(C, K, ranked)
        if int(nm[1:].split("s")[0]) < 20:
            S[nm].pop("ex20")
        if nm == "b1":
            S[nm].pop("ex5")
        rows.append(row(C, {"b1": "beam 1 (= iter, re-implemented)"}.get(nm, "beam " + nm[1:].replace("s", " s")), S[nm],
                        extra=[f"{K['cost'][nm]['rows'] / len(K['ids']):.0f}"]))
    C.table(head2("decoder", extra=["rows/gap"]), rows)
    C.add("By gap length, half B (exact@1 / exact@5 / exact@20, SER):")
    rows = []
    for nm in ("b1", "b5", "b20"):
        r = ["beam " + nm[1:]]
        for b, lo, hi in BINS:
            m = C.B & (C.L >= lo) & (C.L <= hi)
            r.append(" / ".join(f"{mean(S[nm][c], m):.1f}" if c in S[nm] else "-" for c in ("ex1", "ex5", "ex20"))
                     + f"; {mean(S[nm]['ser'], m):.1f}")
        rows.append(r)
    C.table(["decoder"] + [f"L {b} (n={int((C.B & (C.L >= lo) & (C.L <= hi)).sum())})" for b, lo, hi in BINS], rows)
    return S


# ----------------------------------------------------------------------------- 2 PLL
def rerank(H, sc):
    o = np.argsort(-np.asarray(sc), kind="stable")
    return [H[j] for j in o]


def sec_pll(C, K, S):
    C.add("## 2. Pseudo-likelihood rescoring of the beam candidates (known width)", "",
          "The best n candidates of a beam are re-ranked; the others keep their order behind them. "
          "PLL = sum over slots of log P(sign | context, all other slots of the candidate), each slot masked "
          "in turn (L rows per candidate); PLL-2 = two passes per candidate (odd slots masked, then even "
          "slots). `chain + a PLL`: a tuned on half A (lowest SER; grid 0.1 .. 4). Oracle = best candidate "
          "among the n by SER.")
    rows = []
    best = {}
    for nm, n in (("b5", 5), ("b20", 5), ("b20", 10)):
        Hs = [[h for h in H] for H in K["beams"][nm]]
        chain = [np.array([h[1] for h in H[:n]]) for H in Hs]
        feats = {}
        for pn in ("pll", "pll2"):
            feats[pn] = [np.array([float(K[pn][i][h[0]].sum()) for h in H[:n]]) for i, H in enumerate(Hs)]

        def system(sc):
            ranked = [[h[0] for h in rerank(H[:n], s)] + [h[0] for h in H[n:]] for H, s in zip(Hs, sc)]
            return sys_from_ranked(C, K, ranked)
        base = system(chain)
        rows.append(row(C, f"beam {nm[1:]}, top {n}: beam score", base, extra=["0"]))
        for pn, lab in (("pll", "PLL"), ("pll2", "PLL-2")):
            d = system(feats[pn])
            rows.append(row(C, f"beam {nm[1:]}, top {n}: {lab} only", d, extra=["L" if pn == "pll" else "2"]))
            res = {}
            for a in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                res[a] = system([c + a * f for c, f in zip(chain, feats[pn])])
            a = min(res, key=lambda a: (mean(res[a]["ser"], C.A), a))
            rows.append(row(C, f"beam {nm[1:]}, top {n}: chain + {a:g} {lab}", res[a], extra=["L" if pn == "pll" else "2"]))
            best[(nm, n, pn)] = (a, res[a])
        # oracle
        orc = []
        for i, H in enumerate(Hs):
            ms = [C.metric(i, D.unkey(h[0]), K["planes"][i][h[0]]) for h in H[:n]]
            orc.append(min(ms, key=lambda t: (t[1], t[2])))
        orc = np.array(orc)
        rows.append(row(C, f"beam {nm[1:]}, top {n}: oracle", {"ex1": orc[:, 0], "ser": orc[:, 1], "cer": orc[:, 2]}, extra=[""]))
    C.table(head2("candidates: ranking", extra=["extra rows / candidate"]), rows)
    return best


# ----------------------------------------------------------------------------- 3 remask
def sec_remask(C, K, S):
    C.add("## 3. Remasking after the iterative fill (ReMDM-style, known width)", "",
          "Start: the iterative fill (beam 1). Each round the k least confident slots are re-masked together "
          "and re-predicted most-confident-first given the rest. `fill`: confidence = probability of the sign "
          "when it was filled in; `pll`: confidence = pseudo-likelihood given the complete fill. rN = N "
          "rounds; f25 / f50: k = 25% / 50% of the slots (at least 1); one: k = 1; acc = the new fill is kept "
          "only if its summed PLL is higher. Single-sign gaps cannot change.")
    rows = [row(C, "beam 1 (iterative), no remasking", S["b1"], cols=("ex1", "ser", "cer"), extra=["", f"{K['cost']['b1']['rows'] / len(K['ids']):.0f}"])]
    out = {}
    for nm, v in K["remask"].items():
        d = sys_from_ranked(C, K, [[k] for k in v])
        ch = 100 * np.mean([k != H[0][0] for k, H in zip(v, K["beams"]["b1"])])
        out[nm] = d
        rows.append(row(C, nm, d, cols=("ex1", "ser", "cer"),
                        extra=[f"{ch:.1f}", f"+{K['cost']['remask_' + nm]['rows'] / len(K['ids']):.0f}"]))
    C.table(head2("variant", cols=["exact", "SER", "CER"], extra=["gaps changed %", "rows/gap"]), rows)
    return out


# ----------------------------------------------------------------------------- 4 widths
WGRID = [dict(blank=b, chain=c, pll=p, norm=n, prior=q) for b, c, p, n, q in
         itertools.product((0.0, 1.0), (0.0, 0.1, 0.25, 0.5, 1.0, 2.0), (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
                           (0.0, 0.5, 1.0), (0.0, 1.0)) if (b or c or p) and (n == 0 or c or p)]
WNAMED = [("BLANK score (as width_first)", dict(blank=1.0)),
          ("chain log-prob, sum", dict(chain=1.0)), ("chain log-prob, per sign", dict(chain=1.0, norm=1.0)),
          ("PLL, sum", dict(pll=1.0)), ("PLL, per sign", dict(pll=1.0, norm=1.0)),
          ("PLL per sign + width prior", dict(pll=1.0, norm=1.0, prior=1.0)),
          ("chain sum + width prior", dict(chain=1.0, prior=1.0)),
          ("BLANK + width prior", dict(blank=1.0, prior=1.0))]


def wfmt(w):
    return ", ".join(f"{k} {v:g}" for k, v in w.items() if v)


class Widths:
    def __init__(self, C, W):
        self.C, self.W = C, W
        N, KM = C.N, D.K_MAX
        self.met = np.full((N, KM, 3), np.nan)
        self.chain = np.zeros((N, KM))
        self.pll = np.zeros((N, KM))
        self.pll2 = np.zeros((N, KM))
        self.have = np.zeros((N, KM), bool)
        for i, d in enumerate(W["w"]):
            for k, r in d.items():
                self.have[i, k - 1] = True
                self.met[i, k - 1] = C.metric(i, D.unkey(r["key"]), r["planes"])
                self.chain[i, k - 1], self.pll[i, k - 1], self.pll2[i, k - 1] = r["chain"], r["pll"].sum(), r["pll2"].sum()
        self.cond = {}
        for c in ("unknown", "r25", "r50"):
            cand = np.zeros((N, KM), bool)
            blank = np.zeros((N, KM))
            prior = np.zeros((N, KM))
            for i, cd in enumerate(W["conds"]):
                M, ks = cd[c]
                cand[i, np.array(ks) - 1] = True
                blank[i, :M] = W["blank"][c][i]
                prior[i, :M] = np.maximum(D.width_prior(M), -30.0)
            assert not (cand & ~self.have).any()
            self.cond[c] = (cand, blank, prior)
        self.kk = np.arange(1, KM + 1)[None, :].astype(float)

    def choose(self, c, w, n_widths=None, pl="pll"):
        cand, blank, prior = self.cond[c]
        if n_widths:
            b = np.where(cand, blank, -np.inf)
            thr = -np.sort(-b, 1)[:, min(n_widths, b.shape[1]) - 1][:, None]
            cand = cand & (b >= thr)
        nk = self.kk ** w.get("norm", 0.0)
        s = (w.get("blank", 0.0) * blank + w.get("chain", 0.0) * self.chain / nk
             + w.get("pll", 0.0) * getattr(self, pl) / nk + w.get("prior", 0.0) * prior)
        s = np.where(cand, s, -np.inf)
        return s.argmax(1) + 1

    def system(self, k):
        ar = np.arange(self.C.N)
        m = self.met[ar, k - 1]
        return {"ex1": m[:, 0], "ser": m[:, 1], "cer": m[:, 2], "wacc": (k == self.C.L).astype(float),
                "wmae": np.abs(k - self.C.L) / 100.0, "k": k}

    def rows_per_gap(self, c, w, n_widths=None):
        cand = self.cond[c][0]
        tot = 1.0
        if w.get("chain") or w.get("pll"):
            ks = [np.flatnonzero(r) + 1 for r in cand]
            if n_widths:
                ks = [k[:n_widths] if len(k) > n_widths else k for k in ks]   # approximate: count only
            per = np.mean([k.sum() if len(k) > 1 else 0 for k in ks])
            tot += per * (1 + (1 if w.get("pll") else 0))
        return tot


def sec_widths(C, W, cur, curw, S):
    C.add("## 4. Unknown width by marginalisation over candidate widths", "",
          "Every admissible width k is decoded as a known-width gap of k slots (iterative fill) and scored; "
          "the best width wins. Components of the width score (cun.decode.combine_width_score): "
          "BLANK = the score of the current width_first decoder (one pass over the M slots: slots < k real, "
          "slots >= k BLANK); chain = sum of log-probabilities of the iterative fill of k slots; PLL = "
          "pseudo-log-likelihood of that fill; `per sign` = divided by k (norm 1; norm 0.5 = divided by "
          "sqrt k); prior = log P(L = k | M slots) of the elastic-gap construction of pretraining "
          "(cun.noise; computed exactly from the construction, cun.decode.width_prior). "
          "`tuned` = the weights with the lowest SER on half A (grid of "
          f"{len(WGRID)} combinations). Width conditions: **unknown** = the L + extra slots of the evaluation "
          "set, any width 1..M; **+-50% / +-25%** = the editor's estimate w is drawn uniformly among the "
          "integers with the true width inside w +- 50% / 25% (seeded; the range is therefore not centred on "
          "the truth), admissible widths = that range (at most 30 slots), the BLANK pass sees max(range) "
          "slots; **exact** = known width. width acc = chosen width is the true one; |dk| = mean absolute "
          "width error in signs. SER / CER of the iterative fill of the chosen width. Cost in rows per gap "
          "for the width decision (the final fill of the chosen width comes on top).")
    Wd = Widths(C, W)
    cols = ("wacc", "wmae", "ex1", "ser", "cer")
    hd = ["width acc", "|dk|", "exact", "SER", "CER"]
    tuned = {}
    for c, title in (("unknown", "Unknown width (L + extra slots)"), ("r50", "Width known to +-50%"), ("r25", "Width known to +-25%")):
        rows = []
        if c == "unknown":
            for nm, src, k_ in (("current indep", cur, "unknown/indep"), ("current iter", cur, "unknown/iter"),
                                ("current width_first (iter-w)", curw, "unknown/iter-w")):
                if src is None:
                    continue
                d = sys_from_preds(C, src[k_])
                kk = np.array([len(p["sign"]) for p in src[k_]])
                d["wacc"], d["wmae"] = (kk == C.L).astype(float), np.abs(kk - C.L) / 100.0
                S["cur_" + k_] = d
                rows.append(row(C, nm, d, cols=cols, extra=["", ""]))
        for nm, w in WNAMED:
            rows.append(row(C, nm, Wd.system(Wd.choose(c, w)), cols=cols, extra=[wfmt(w), f"{Wd.rows_per_gap(c, w):.0f}"]))
        best = min(WGRID, key=lambda w: (mean(Wd.system(Wd.choose(c, w))["ser"], C.A), w["pll"] > 0, w["chain"] > 0))
        tuned[c] = best
        rows.append(row(C, "tuned on A", Wd.system(Wd.choose(c, best)), cols=cols, extra=[wfmt(best), f"{Wd.rows_per_gap(c, best):.0f}"]))
        if c != "unknown":
            rows.append(row(C, "weights tuned on 'unknown'", Wd.system(Wd.choose(c, tuned["unknown"])), cols=cols,
                            extra=[wfmt(tuned["unknown"]), f"{Wd.rows_per_gap(c, tuned['unknown']):.0f}"]))
        nop = [w for w in WGRID if not w["pll"]]
        b2 = min(nop, key=lambda w: (mean(Wd.system(Wd.choose(c, w))["ser"], C.A), w["chain"] > 0))
        tuned[c + ":nopll"] = b2
        rows.append(row(C, "tuned on A, no PLL", Wd.system(Wd.choose(c, b2)), cols=cols, extra=[wfmt(b2), f"{Wd.rows_per_gap(c, b2):.0f}"]))
        b3 = min(WGRID, key=lambda w: (mean(Wd.system(Wd.choose(c, w, pl="pll2"))["ser"], C.A), w["pll"] > 0))
        rows.append(row(C, "tuned on A, PLL-2 in place of PLL", Wd.system(Wd.choose(c, b3, pl="pll2")), cols=cols, extra=[wfmt(b3), ""]))
        for nw in (3, 5):
            rows.append(row(C, f"tuned weights, only the {nw} widths with the best BLANK score",
                            Wd.system(Wd.choose(c, best, n_widths=nw)), cols=cols, extra=[wfmt(best), ""]))
        orc = Wd.system(C.L.copy())
        rows.append(row(C, "oracle width (= known width, iterative)", orc, cols=cols, extra=["", ""]))
        C.add(f"### {title}")
        C.table(head2("width score", cols=hd, extra=["weights", "rows/gap"]), rows)
    # summary by condition and by L
    C.add("### The four width conditions, iterative fill of the chosen width (tuned width score per condition)")
    rows = []
    for c, lab in (("exact", "exact width"), ("r25", "+-25%"), ("r50", "+-50%"), ("unknown", "unknown (L + extra slots)")):
        d = Wd.system(C.L.copy()) if c == "exact" else Wd.system(Wd.choose(c, tuned[c]))
        rows.append(row(C, lab, d, cols=cols))
        if c != "exact":
            rows.append(row(C, lab + ", BLANK score only", Wd.system(Wd.choose(c, dict(blank=1.0))), cols=cols))
    C.table(head2("condition", cols=hd), rows)
    C.add("Single-sign gaps (L = 1) and the other length bins, half B, CER (SER):")
    rows = []
    systems = []
    if "cur_unknown/iter-w" in S:
        systems.append(("unknown: current width_first", S["cur_unknown/iter-w"]))
    for c, lab in (("unknown", "unknown"), ("r50", "+-50%"), ("r25", "+-25%")):
        systems.append((f"{lab}: BLANK score", Wd.system(Wd.choose(c, dict(blank=1.0)))))
        systems.append((f"{lab}: tuned", Wd.system(Wd.choose(c, tuned[c]))))
    systems.append(("exact width", Wd.system(C.L.copy())))
    for nm, d in systems:
        r = [nm]
        for b, lo, hi in BINS:
            m = C.B & (C.L >= lo) & (C.L <= hi)
            r.append(f"{mean(d['cer'], m):.1f} ({mean(d['ser'], m):.1f})")
        rows.append(r)
    C.table(["system"] + [f"L {b}" for b, _, _ in BINS], rows)
    C.js["width_weights"] = tuned
    return Wd, tuned


# ----------------------------------------------------------------------------- 5 calibration
def sec_calib(C, K, S):
    C.add("## 5. Temperature scaling of the sign head, abstention", "",
          "Single-sign gaps (known width): p = softmax(logits / T) over the real signs; T fitted on half A by "
          "NLL of the true sign. ECE: 10 equal-width confidence bins; AUROC: confidence of the top sign "
          "separating right from wrong predictions. Because half A has only about 150 single-sign gaps, the "
          "same is repeated on every slot of the first pass of all gaps (all slots masked: the `indep` "
          "situation; probabilities renormalised over the 128 best signs + the true sign).")
    cal = K["calib"]
    ids = np.arange(C.N)                       # C is aligned with the instances of K
    Ts = np.exp(np.linspace(-1.2, 1.2, 241))
    res = {}

    def analyse(v, gold_logit, top_ok, half, label):
        v = v.astype(np.float64)

        def stats(T):
            z = v / T
            mx = z.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(z - mx).sum(1))
            return gold_logit / T - lse, np.exp(z[:, 0] - lse)
        a = half == "A"
        T = Ts[int(np.argmin([-stats(t)[0][a].mean() for t in Ts]))]
        rows = []
        for hn in ("A", "B"):
            m = half == hn
            l1, c1 = stats(1.0)
            lt, ct = stats(T)
            rows.append([f"{label}, half {hn}", int(m.sum()), 100 * top_ok[m].mean(), -l1[m].mean(), -lt[m].mean(),
                         ece(c1[m], top_ok[m]), ece(ct[m], top_ok[m]), auroc(c1[m], top_ok[m]), auroc(ct[m], top_ok[m])])
        res[label] = {"T": float(T), "rows": rows}
        return T, stats(1.0)[1], stats(T)[1], rows

    one = np.array(cal["one"])
    gold1 = np.array([C.gold[ids[i]][0][0] for i in one])
    v1 = cal["one_v"]
    i1 = cal["one_i"].astype(np.int64)
    gl = np.array([v1[r, np.flatnonzero(i1[r] == gold1[r])[0]] for r in range(len(one))], float)
    ok1 = i1[:, 0] == gold1
    h1 = C.half[ids[one]]
    T1, c1, ct1, rows1 = analyse(v1, gl, ok1, h1, "single-sign gaps")
    # all slots of the first pass
    vs, gls, oks, hs = [], [], [], []
    for r, i in enumerate(ids):
        Lr = C.L[i]
        g = C.gold[i][0]
        for j in range(Lr):
            vv = cal["rawv"][r, j].astype(np.float64)
            ii = cal["rawi"][r, j].astype(np.int64)
            gt = float(cal["rawt"][r, j])
            if g[j] not in ii:
                vv = np.r_[vv[:-1], gt]
            vs.append(vv)
            gls.append(gt)
            oks.append(ii[0] == g[j])
            hs.append(C.half[i])
    T2, c2, ct2, rows2 = analyse(np.array(vs), np.array(gls), np.array(oks), np.array(hs), "all first-pass slots")
    C.add(f"Fitted on half A: T = {T1:.2f} (single-sign gaps), T = {T2:.2f} (all first-pass slots).")
    C.table(["set", "n", "top-1 acc", "NLL T=1", "NLL fitted T", "ECE T=1", "ECE fitted T", "AUROC T=1", "AUROC fitted T"],
            [[r[0], r[1]] + [float(x) if k != 1 and k != 2 else f"{x:.3f}" for k, x in enumerate(r[2:])] for r in rows1 + rows2])
    C.add("Reliability, half B (confidence bin: n, mean confidence, accuracy):")
    edges = [0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0001]
    rows = []
    for lab, c_, ct_, ok_, h_ in (("single-sign gaps", c1, ct1, ok1, h1), ("all first-pass slots", c2, ct2, np.array(oks), np.array(hs))):
        for tn, cc in (("T=1", c_), ("fitted T", ct_)):
            r = [f"{lab}, {tn}"]
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (h_ == "B") & (cc >= lo) & (cc < hi)
                r.append(f"{int(m.sum())}: {100 * cc[m].mean():.0f} / {100 * ok_[m].mean():.0f}" if m.any() else "0")
            rows.append(r)
    C.table(["set"] + [f"[{lo:g}, {min(hi, 1):g}{']' if hi > 1 else ')'}" for lo, hi in zip(edges[:-1], edges[1:])], rows)
    C.add("### Abstention: risk-coverage", "",
          "Predictions are kept from the most confident down; accuracy of those kept at each coverage. "
          "Single signs: confidence = top probability (the ranking does not depend on T up to ties; the "
          "threshold does). Whole gaps (known width, beam 20, L = 1..20 pooled, and L >= 2): `beam score` = "
          "chain log-probability of the best candidate; `per sign` = divided by L; `margin` = best minus "
          "second-best score; `beam share` = exp(best) / sum of exp over the 20 candidates.")
    rows = []
    for hn in ("A", "B"):
        m = h1 == hn
        rows.append([f"single-sign gaps, half {hn} (n={int(m.sum())}): accuracy"] + risk_cov(c1[m], ok1[m]) + [auroc(c1[m], ok1[m])])
    H = K["beams"]["b20"]
    s1 = np.array([h[0][1] for h in H])
    conf = {"beam score": s1, "per sign": s1 / C.L[ids],
            "margin": np.array([h[0][1] - h[1][1] if len(h) > 1 else 50.0 for h in H]),
            "beam share": np.array([1.0 / np.exp(np.array([x[1] for x in h]) - h[0][1]).sum() for h in H])}
    ex, ser = S["b20"]["ex1"], S["b20"]["ser"]
    for sub, sm in (("all gaps", np.ones(C.N, bool)), ("L >= 2", C.L >= 2)):
        for cn, cv in conf.items():
            for hn in ("A", "B"):
                m = (C.half == hn) & sm
                rows.append([f"whole gaps, {sub}, {cn}, half {hn} (n={int(m.sum())}): exact"] + risk_cov(cv[m], ex[m]) + [auroc(cv[m], ex[m] > 0)])
        for hn in ("A", "B"):
            m = (C.half == hn) & sm
            rows.append([f"whole gaps, {sub}, per sign, half {hn}: SER of the kept"] + risk_cov(conf["per sign"][m], ser[m]) + [""])
    C.table(["confidence"] + [f"{c}%" for c in COV] + ["AUROC"], rows)
    # threshold transfer
    C.add("Thresholds chosen on half A for a target accuracy of the kept predictions, applied to half B:")
    rows = []
    for lab, cv, okv, hv in (("single signs, calibrated probability", ct1, ok1.astype(float), h1),
                             ("whole gaps, beam share", conf["beam share"], ex, C.half),
                             ("whole gaps, beam score per sign", conf["per sign"], ex, C.half)):
        for target in (0.9, 0.8):
            a = hv == "A"
            o = np.argsort(-cv[a], kind="stable")
            acc = np.cumsum(okv[a][o]) / np.arange(1, a.sum() + 1)
            good = np.flatnonzero(acc >= target)
            if not len(good):
                continue
            thr = cv[a][o][good[-1]]
            b = (hv == "B") & (cv >= thr)
            rows.append([lab, f"{100 * target:.0f}", f"{thr:.3f}", 100 * (good[-1] + 1) / a.sum(),
                         100 * b.sum() / (hv == "B").sum(), 100 * okv[b].mean() if b.any() else float("nan")])
    C.table(["confidence", "target acc", "threshold", "A coverage", "B coverage", "B accuracy of kept"], rows)
    C.js["calibration"] = res


# ----------------------------------------------------------------------------- 6 retrieval
N_RETR = 3                                      # retrieved fills per gap that get a bonus / are injected


def bonus_of(rec, wq, wp, wf):
    """Bonus of the retrieved fills of one gap: wq * (agreeing context signs - 7, parallels only)
    + wp * log(1 + occurrences in parallels) + wf * log(1 + occurrences between the anchors)."""
    return {D.key(f[0]): float(wq * max(0, f[2] - 7) * (f[4] > 0) + wp * np.log1p(f[4]) + wf * np.log1p(f[1]))
            for f in rec["fills"][:N_RETR]}


def width_bonus_of(byG, wn):
    """Width evidence from retrieval: wn * log(1 + parallels found for a gap of k signs)."""
    out = {}
    for G, r in byG.items():
        n = sum(f[4] for f in r["fills"])
        if n:
            out[G] = float(wn * np.log1p(n))
    return out


def sec_retrieval(C, K, R, S):
    C.add("## 6. Retrieval-copy baseline and interpolation (known width)", "",
          "scripts/decode_retrieve.py searches the train texts (15.8M signs) for the context of the gap: "
          "anchors next to the gap must match exactly (2 signs before and 2 after, exactly L signs apart; "
          "fallbacks 1 + 1, then 3 on one side), candidates are ranked by fuzzy agreement of the 6 signs "
          "before and 6 after (aligned, substitutions allowed). Regimes: **parallel** = a candidate with >= 4 "
          "of the 6 signs matching on each side; **formulaic** = only a shorter match; **none**. The "
          "retrieval-copy baseline predicts the most frequent fill of the best-matching candidates "
          "(readings etc. from the model's final pass for the CER).")
    recs = [R["inst"][i]["known"] for i in range(C.N)]
    cat = np.array([r["cat"] for r in recs])
    cats = ("parallel", "formulaic", "none")
    C.table(["regime", "A n", "A %", "B n", "B %", "novel gaps % (A+B)"],
            [[c, int((C.A & (cat == c)).sum()), 100 * (C.A & (cat == c)).sum() / C.A.sum(), int((C.B & (cat == c)).sum()),
              100 * (C.B & (cat == c)).sum() / C.B.sum(), 100 * np.mean([C.inst[i]["novel"] for i in np.flatnonzero(cat == c)])] for c in cats])
    # copy baseline
    cp = np.zeros((C.N, 3))
    cov = np.zeros(C.N, bool)
    cp5 = np.zeros(C.N)
    for i, r in enumerate(recs):
        if r["fills"]:
            k = D.key(r["fills"][0][0])
            cov[i] = True
            cp[i] = C.metric(i, D.unkey(k), K["planes"][i][k]) if k in K["planes"][i] else (0, 1, 1)
            cp5[i] = float(C.gkey[i] in [D.key(f[0]) for f in r["fills"][:5]])
        else:
            cp[i] = (0, 1, 1)
    rows = []
    for c in ("parallel", "formulaic", "all"):
        for hn, hm in (("A", C.A), ("B", C.B)):
            m = hm & ((cat == c) if c != "all" else np.ones(C.N, bool))
            mc = m & cov
            rows.append([f"{c}, half {hn}", int(m.sum()), 100 * mc.sum() / m.sum(), mean(cp[:, 0], mc), mean(cp5, mc), mean(cp[:, 1], mc),
                         mean(cp[:, 2], mc), mean(cp[:, 0], m), mean(cp[:, 1], m),
                         mean(S["b1"]["ex1"], m), mean(S["b1"]["ser"], m), mean(S["b20"]["ex1"], m), mean(S["b20"]["ser"], m)])
    C.add("Retrieval-copy baseline (covered = a fill was retrieved; `all gaps` counts an uncovered gap as wrong, SER 100) "
          "next to the model on the same gaps:")
    C.table(["regime", "n", "coverage %", "copy exact (covered)", "copy exact@5 (covered)", "copy SER (covered)", "copy CER (covered)",
             "copy exact (all gaps)", "copy SER (all gaps)", "iter exact", "iter SER", "beam 20 exact", "beam 20 SER"], rows)
    C.add("Model by regime (beam 20; exact@1 / @5 / @20, SER, CER):")
    rows = []
    for c in cats:
        for hn, hm in (("A", C.A), ("B", C.B)):
            m = hm & (cat == c)
            rows.append([f"{c}, half {hn}", int(m.sum())] + [mean(S["b20"][k], m) for k in ("ex1", "ex5", "ex20", "ser", "cer")]
                        + [mean(S["b1"][k], m) for k in ("ex1", "ser", "cer")])
    C.table(["regime", "n", "exact", "exact@5", "exact@20", "SER", "CER", "iter exact", "iter SER", "iter CER"], rows)
    # interpolation
    C.add("### Interpolation", "",
          f"Candidates = the 20 beam hypotheses, plus (`+ inject`) the {N_RETR} best retrieved fills scored by the "
          "model along the same path (cun.decode.score_fill). Score = chain log-probability + bonus of a "
          "retrieved fill, bonus = wq * (q - 7) [fills found in a parallel; q = agreeing context signs of its "
          "best occurrence, 8..12] + wp * log(1 + n_par) + wf * log(1 + n_all); n_par = occurrences of the "
          "fill in parallels, n_all = occurrences between the matched anchors. Weights tuned on half A "
          "(lowest SER; wq in 0..32, wp in 0..32, wf in 0..1). The tuned weights are large: where a parallel "
          "exists the copy is in effect preferred to the model, ordered by how well its context agrees.")
    H = K["beams"]["b20"]

    def system(g, inject=True, beam="b20"):
        ranked = []
        for i, r in enumerate(recs):
            bon = bonus_of(r, *g) if any(g) else {}
            cd = {h[0]: h[1] for h in K["beams"][beam][i]}
            if inject:
                for f in r["fills"][:N_RETR]:
                    k = D.key(f[0])
                    if k not in cd:
                        cd[k] = K["forced"][i][k]
            ranked.append([k for k, _ in sorted(cd.items(), key=lambda t: -(t[1] + bon.get(t[0], 0.0)))])
        return sys_from_ranked(C, K, ranked)
    grid = [(wq, wp, wf) for wq in (0.0, 2.0, 4.0, 8.0, 16.0, 32.0) for wp in (0.0, 2.0, 8.0, 32.0) for wf in (0.0, 0.25, 1.0)]
    out = {}
    rows = [row(C, "beam 20", S["b20"], extra=["", "", ""])]
    for inj in (False, True):
        res = {g: system(g, inject=inj) for g in grid}
        g = min(res, key=lambda g: (mean(res[g]["ser"], C.A), g))
        out[inj] = (g, res[g])
        rows.append(row(C, "beam 20 + bonus" + (" + inject" if inj else ""), res[g], extra=[f"{v:g}" for v in g]))
        if inj:
            for lab, sub in (("log-count bonuses only (wq = 0)", [g_ for g_ in grid if g_[0] == 0]),
                             ("parallels only (wf = 0)", [g_ for g_ in grid if g_[2] == 0])):
                g2 = min(sub, key=lambda g: (mean(res[g]["ser"], C.A), g))
                rows.append(row(C, "beam 20 + bonus + inject, " + lab, res[g2], extra=[f"{v:g}" for v in g2]))
    g, best = out[True]
    b5 = system(g, beam="b5")
    b5.pop("ex20")
    rows.append(row(C, "beam 5 + bonus + inject (same weights)", b5, extra=[f"{v:g}" for v in g]))
    C.table(head2("system", extra=["wq", "wp", "wf"]), rows)
    C.add(f"By regime, half B (beam 20 -> beam 20 + bonus + inject, wq = {g[0]:g}, wp = {g[1]:g}, wf = {g[2]:g}):")
    rows = []
    for c in cats:
        m = C.B & (cat == c)
        rows.append([c, int(m.sum())] + [f"{mean(S['b20'][k], m):.1f} -> {mean(best[k], m):.1f}" for k in ("ex1", "ex5", "ex20", "ser", "cer")])
    C.table(["regime", "n", "exact", "exact@5", "exact@20", "SER", "CER"], rows)
    C.js["retrieval_weights"] = {"wq": g[0], "wp": g[1], "wf": g[2]}
    return g, best, cat


def sec_retr_width(C, Wd, R):
    C.add("### Retrieval as evidence for the width (unknown width)", "",
          "The search is repeated for every candidate width k (anchors k signs apart). Width score = chain "
          "log-probability (+ prior in the `unknown` condition) + wn * log(1 + parallels found at width k); "
          "iterative fill of the chosen width by the model (no copy bonus on the fill here; section 7 has "
          "both).")
    npar = np.zeros((C.N, D.K_MAX))
    for i in range(C.N):
        for G, v in width_bonus_of(R["inst"][i]["byG"], 1.0).items():
            npar[i, G - 1] = v
    rows = []
    for c, w in (("unknown", dict(chain=1.0, prior=1.0)), ("r50", dict(chain=1.0)), ("r25", dict(chain=1.0))):
        cand, blank, prior = Wd.cond[c]
        for wn in (0.0, 0.5, 1.0, 2.0, 4.0):
            s = w.get("chain", 0) * Wd.chain + w.get("prior", 0) * prior + wn * npar
            d = Wd.system(np.where(cand, s, -np.inf).argmax(1) + 1)
            rows.append(row(C, f"{c}: {wfmt(w)}, wn = {wn:g}", d, cols=("wacc", "wmae", "ex1", "ser", "cer")))
    C.table(head2("condition: width score", cols=["width acc", "|dk|", "exact", "SER", "CER"]), rows)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-plan", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    C = Ctx()
    K, W, R, F = load(f"raw_known{a.tag}.pkl.gz"), load(f"raw_widths{a.tag}.pkl.gz"), load("retrieval__dev.pkl.gz"), load(f"raw_final{a.tag}.pkl.gz")
    gp = os.path.join(ROOT, "results", "gaps")
    cur = pickle.load(gzip.open(os.path.join(gp, "base-long__dev.pkl.gz"), "rb"))
    curw = pickle.load(gzip.open(os.path.join(gp, "base-long-w__dev.pkl.gz"), "rb"))
    for f in (K, W, R, cur, curw):
        assert f is None or (f["version"] == C.ev["version"] and f.get("split", "dev") == "dev")
    if a.tag:                                   # smoke test: restrict everything to the instances decoded
        sub_ctx(C, K["ids"])
        cur = {k: ([v[i] for i in K["ids"]] if "/" in k else v) for k, v in cur.items()}
        curw = {k: ([v[i] for i in K["ids"]] if "/" in k else v) for k, v in curw.items()}
        if R:
            R = {"inst": [R["inst"][i] for i in K["ids"]]}
    C.add(f"# Decoding without retraining: model base-long, dev gaps (corpus {C.ev['version']})", "")
    S = sec_beam(C, K, cur)
    best_pll = sec_pll(C, K, S)
    sec_remask(C, K, S)
    Wd = tuned = None
    if W:
        Wd, tuned = sec_widths(C, W, cur, curw, S)
    sec_calib(C, K, S)
    if R:
        g, bestR, cat = sec_retrieval(C, K, R, S)
        if Wd is not None:
            sec_retr_width(C, Wd, R)
    if a.write_plan:
        WN = 1.0
        allb = []
        for i in range(C.N):
            d = {}
            for G, r in R["inst"][i]["byG"].items():
                d.update(bonus_of(r, *g))
            allb.append(d)
        plan = {"known": {"beam20": dict(B=20, topk=20),
                          "beam5": dict(B=5, topk=5),
                          "beam20+retr": dict(B=20, topk=20, bonus="known"),
                          "beam5+retr": dict(B=5, topk=5, bonus="known")},
                "bonus": {"known": [bonus_of(R["inst"][i]["known"], *g) for i in range(C.N)], "all": allb},
                "width_bonus": [width_bonus_of(R["inst"][i]["byG"], WN) for i in range(C.N)],
                "unknown": {"width_first+beam20": dict(cond="unknown", weights=dict(blank=1.0)),
                            "marg+beam20": dict(cond="unknown", weights=dict(chain=1.0, prior=1.0)),
                            "marg+beam20+retr": dict(cond="unknown", weights=dict(chain=1.0, prior=1.0), bonus="all", width_bonus=True),
                            "r50:marg+beam20": dict(cond="r50", weights=dict(chain=1.0)),
                            "r50:marg+beam20+retr": dict(cond="r50", weights=dict(chain=1.0), bonus="all", width_bonus=True),
                            "r25:marg+beam20": dict(cond="r25", weights=dict(chain=1.0)),
                            "r25:marg+beam20+retr": dict(cond="r25", weights=dict(chain=1.0), bonus="all", width_bonus=True)}}
        with gzip.open(os.path.join(OUT, "final_plan.pkl.gz"), "wb") as fh:
            pickle.dump(plan, fh, protocol=5)
        print("plan written", plan["known"], plan["unknown"])
    if F:
        from decode_final import sec_final
        sec_final(C, F, S, K, W, Wd, tuned)
    name = "REPORT.md" if not a.tag else f"REPORT{a.tag}.md"
    open(os.path.join(OUT, name), "w").write("\n".join(C.lines) + "\n")
    json.dump(C.js, open(os.path.join(OUT, f"scores{a.tag}.json"), "w"), indent=1, default=float)
    print("\n".join(C.lines))


def sub_ctx(C, ids):
    ids = np.array(ids)
    C.inst = [C.inst[i] for i in ids]
    C.N = len(ids)
    for k in ("L", "text", "half", "A", "B"):
        setattr(C, k, getattr(C, k)[ids])
    for k in ("gold", "gkey", "gstr"):
        setattr(C, k, [getattr(C, k)[i] for i in ids])


if __name__ == "__main__":
    main()
