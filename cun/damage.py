"""Geometry of real damage and a sampler that transplants realistic break shapes onto preserved text
(PLAN_v3 stage 0, item 2).

The corpus has no surface coordinates, only lines and a per-sign status, so all geometry is per line:
where in a line a lost run starts and ends, and how that persists over consecutive lines.

Status of a sign position (STATUS_*):
  lost      brk == 2 (inside [ ]: restored by the editor, or gone), or an <x> / <gap> token
  abraded   brk == 1 ('#', half brackets) on an identified sign: damaged but partly legible
  clean     everything else
A <gap> token ('...') stands for an unknown number of signs. For the geometry it is expanded to an
estimated number of sign positions (`virtual`): the line is filled up to the median length of the text's
lines without '...' (at least one position per token). Raw token counts are reported next to it.

Fitted model (`fit`, on texts of the given split only; saved next to the corpus, `stats_path`):
  * every text gets a loss class (share of lost signs: 0, <=5%, <=15%, <=30%, <=50%, <=70%, more) and an abrasion class
    (share of abraded signs: 0, <=5%, <=20%, more); the joint class distribution is stored per stratum;
  * edge chain: a Markov chain over the lines of a text on the joint state (share of the line lost at its
    beginning, share lost at its end; tenths; or 'whole line lost'), one transition matrix per loss class.
    It produces left / right edge loss, corners (extents that grow or shrink) and blocks of lost lines
    (top / bottom of a surface) with the real persistence;
  * interior chain: number of medial lost runs per line given the previous line; a run under a run of the
    previous line keeps its horizontal position (fitted displacement) and a related length: interior
    holes over several lines and isolated medial runs;
  * abrasion chain: number of abraded signs of the line (0, 1, 2, more, more than half) given the line
    above and whether this line has a loss; abraded signs sit at the edge of a lost run with the fitted
    probability, otherwise anywhere in the preserved part of the line;
  * a library of real connected lost regions (edge, corner, lost lines, hole, medial run) for `sample_gap`.

API
  stats = fit(corpus, split='train');  save_stats(stats);  stats = load_stats()
  mask, abraded = sample_mask(rng, line_lengths, stats, stratum=None, target_rate=None)
  mask, info    = sample_gap(rng, line_lengths, stats, max_signs)
Both return boolean arrays over the signs of the chunk (lines concatenated in order). `abraded` marks the
masked signs that are damaged but partly legible; masked and not abraded = lost. A requested rate scales
the extents of the drawn pattern (lost edges widen until little of a line is left; if that is not enough,
what is left of every line is cut back at one edge, as in a fragment, or further lines are lost).
Scripts: scripts/damage_fit.py, damage_report.py, damage_validate.py, damage_test.py.
"""
from __future__ import annotations

import os
import pickle
import re
from bisect import bisect_right

import numpy as np

from .consts import GAP, X_SIGN
from .paths import PROC

CLEAN, ABRADED, LOST = 0, 1, 2
NB = 10                                   # an edge extent is binned in tenths of the line
NE = (NB + 1) * (NB + 1) + 1              # edge states: (left bin, right bin) and 'whole line'
WHOLE = NE - 1
LOST_EDGES = (0.05, 0.15, 0.30, 0.50, 0.70)   # loss classes of a text: 0 | (0,.05] | ... | (.50,.70] | more
ABR_EDGES = (0.05, 0.20)                  # abrasion classes: 0 | (0,.05] | (.05,.20] | more
N_LC, N_AC = len(LOST_EDGES) + 2, len(ABR_EDGES) + 2
N_AL = 5                                  # abraded signs of a line: 0 | 1 | 2 | 3 or more, at most half | more than half
N_CEN = 20                                # bins of the centre of a medial run (fraction of the line)
N_DCEN = 41                               # bins of the displacement of the centre between lines, [-1, 1]
MAX_MED = 30                              # longest medial run kept apart (signs)
LEN_BINS = (1, 2, 3, 5)                   # previous-run length classes: 1 | 2 | 3 | 4-5 | 6+
GAP_TYPES = ("left", "right", "corner", "lines", "hole", "medial")
STRATA = ("sux_admin", "sux_lit", "akk_doc", "akk_lit", "other")
SMOOTH = 20.0                             # pseudo-counts of the pooled row added to a stratum's row


# ------------------------------------------------------------------ metadata
def coarse_genre(g):
    g = (g or "").lower()
    if not g or "uncertain" in g:
        return "unknown"
    if "letter" in g:
        return "letter"
    if "admin" in g:
        return "administrative"
    if any(k in g for k in ("legal", "grant", "decree", "treaty", "appointment", "votive")):
        return "legal"
    if any(k in g for k in ("ritual", "liter", "liturgy", "hymn", "prayer", "incantation", "prophecy")):
        return "literary"
    if any(k in g for k in ("royal", "monument", "inscription")):
        return "royal"
    if any(k in g for k in ("lexical", "school")):
        return "lexical"
    return "scholarly"


def coarse_period(p):
    p = (p or "").lower()
    for key, name in (("ur iii", "Ur III"), ("old assyrian", "Old Assyrian"), ("old babylonian", "Old Babylonian"),
                      ("old akkadian", "Old Akkadian / Lagash II"), ("laga", "Old Akkadian / Lagash II"),
                      ("ed i", "Early Dynastic"), ("early dynastic", "Early Dynastic"), ("ebla", "Early Dynastic"),
                      ("middle", "Middle Bab. / Ass."), ("neo-assyrian", "Neo-Assyrian"), ("century", "Neo-Assyrian"),
                      ("babylonian", "Neo- / Late Babylonian"), ("achaemenid", "Neo- / Late Babylonian"),
                      ("hellenistic", "Neo- / Late Babylonian"), ("seleucid", "Neo- / Late Babylonian"),
                      ("parthian", "Neo- / Late Babylonian"), ("first millennium", "Neo- / Late Babylonian")):
        if key in p:
            return name
    return "other / unknown"


def stratum_of(t):
    """Stratum of a text: its own 'stratum' field when the corpus build has one (corpus v3), otherwise
    derived from language and coarse genre (the four strata of PLAN_v3 and 'other')."""
    if t.get("stratum"):
        return t["stratum"]
    g = coarse_genre(t.get("genre"))
    doc = g in ("administrative", "legal", "letter")
    lit = g in ("literary", "royal", "lexical", "scholarly")
    if t.get("lang") == "sux" and (doc or lit):
        return "sux_admin" if doc else "sux_lit"
    if t.get("lang") == "akk" and (doc or lit):
        return "akk_doc" if doc else "akk_lit"
    return "other"


# ------------------------------------------------------------------ corpus -> flat arrays
def flatten(corpus, split="train"):
    """Lines of the texts of `split` (a name or a collection of names) as flat arrays:
    planes (5, N), ll = signs per line, lt = text index per line."""
    names = {split} if isinstance(split, str) else set(split)
    keep = np.array([t["split"] in names for t in corpus["texts"]])
    arrs, lt = [], []
    for ti, a in corpus["lines"]:
        if keep[ti] and a.shape[1]:
            arrs.append(a)
            lt.append(ti)
    if not arrs:
        raise ValueError(f"no text with split in {sorted(names)}")
    return {"flat": np.concatenate(arrs, 1), "ll": np.array([a.shape[1] for a in arrs], np.int64),
            "lt": np.array(lt, np.int64)}


def token_status(flat):
    sign, brk = flat[0], flat[4]
    st = np.zeros(flat.shape[1], np.int8)
    st[brk == 1] = ABRADED
    st[(brk == 2) | (sign == X_SIGN) | (sign == GAP)] = LOST
    return st


def virtual(F):
    """Status per sign position with every <gap> token expanded to its estimated width.
    Returns status, ll (positions per line), lt, width (positions per token, 1 except for <gap>)."""
    flat, ll, lt = F["flat"], F["ll"], F["lt"]
    st = token_status(flat)
    off = np.r_[0, np.cumsum(ll)]
    isgap = flat[0] == GAP
    ngap = np.add.reduceat(isgap.astype(np.int64), off[:-1])
    # median length of the text's lines without '...' (fallback: of all such lines)
    ok = ngap == 0
    med_all = float(np.median(ll[ok])) if ok.any() else float(np.median(ll))
    order = np.lexsort((ll[ok], lt[ok]))
    t_s, l_s = lt[ok][order], ll[ok][order]
    ut, first, cnt = np.unique(t_s, return_index=True, return_counts=True)
    med = np.full(int(lt.max()) + 1, med_all)
    m = 0.5 * (l_s[first + (cnt - 1) // 2] + l_s[first + cnt // 2])
    med[ut[cnt >= 3]] = m[cnt >= 3]
    lines = np.flatnonzero(ngap > 0)
    total = np.maximum(ngap[lines], np.rint(med[lt[lines]] - (ll[lines] - ngap[lines])).astype(np.int64))
    per, rem = total // ngap[lines], total % ngap[lines]
    width = np.ones(flat.shape[1], np.int64)
    lid = np.repeat(np.arange(len(ll)), ll)
    gi = np.flatnonzero(isgap)
    pos = np.searchsorted(lines, lid[gi])
    rank = np.cumsum(isgap)[gi] - np.cumsum(isgap)[off[lid[gi]]] + isgap[off[lid[gi]]] - 1   # index of the gap in its line
    width[gi] = per[pos] + (rank < rem[pos])
    return np.repeat(st, width), np.add.reduceat(width, off[:-1]), lt, width


# ------------------------------------------------------------------ geometry of a status array
def _runs(flag, lid, off, ll):
    """Maximal runs of True within lines: line, start position in the line, length."""
    if not len(flag):
        z = np.zeros(0, np.int64)
        return z, z, z
    first = np.zeros(len(flag), bool)
    first[off[:-1][ll > 0]] = True
    s = flag.copy()
    s[1:] &= ~flag[:-1] | first[1:]
    e = flag.copy()
    e[:-1] &= ~flag[1:] | first[1:]
    rs, re_ = np.flatnonzero(s), np.flatnonzero(e) + 1
    line = lid[rs]
    return line, rs - off[line], re_ - rs


def geometry(status, ll, grp):
    """Per-line and per-run description of a status array. grp = id of the text (or chunk) of each line;
    consecutive lines with the same id are vertical neighbours."""
    ll = np.asarray(ll, np.int64)
    nl = len(ll)
    off = np.r_[0, np.cumsum(ll)]
    lid = np.repeat(np.arange(nl), ll)
    lost = status == LOST
    abr = status == ABRADED
    rl, rp, rn = _runs(lost, lid, off, ll)
    n_r = ll[rl]
    ini, fin = rp == 0, rp + rn == n_r
    a = np.zeros(nl, np.int64)
    b = np.zeros(nl, np.int64)
    a[rl[ini]] = rn[ini]
    b[rl[fin]] = rn[fin]
    whole = (a == ll) & (ll > 0)
    med = ~ini & ~fin
    nmed = np.bincount(rl[med], minlength=nl)
    nlost = np.bincount(lid[lost], minlength=nl)
    nabr = np.bincount(lid[abr], minlength=nl)
    al, ap, an = _runs(abr, lid, off, ll)
    # abraded run touching a lost sign of the same line
    lpad = np.r_[False, lost, False]
    left_l = lpad[off[al] + ap] & (ap > 0)
    right_l = lpad[off[al] + ap + an + 1] & (ap + an < ll[al])
    same = np.zeros(nl, bool)
    same[1:] = np.asarray(grp)[1:] == np.asarray(grp)[:-1]
    return dict(ll=ll, grp=np.asarray(grp), same=same, a=a, b=b, whole=whole, nmed=nmed, nlost=nlost, nabr=nabr,
                run_line=rl, run_pos=rp, run_len=rn, run_ini=ini, run_fin=fin,
                abr_line=al, abr_pos=ap, abr_len=an, abr_adj=left_l | right_l)


def _cond(x, y, same):
    """P(y on line i+1 | x on line i), P(y | not x) over vertical neighbours."""
    x0, y1 = x[:-1][same[1:]], y[1:][same[1:]]
    return (float(y1[x0].mean()) if x0.any() else float("nan"),
            float(y1[~x0].mean()) if (~x0).any() else float("nan"))


def _corr(x, y):
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(G):
    """Comparable statistics of a damage pattern (real or synthetic)."""
    ll, a, b, whole, same = G["ll"], G["a"], G["b"], G["whole"], G["same"]
    N = max(1, int(ll.sum()))
    nl = max(1, len(ll))
    I, Fi, M = (a > 0) & ~whole, (b > 0) & ~whole, G["nmed"] > 0
    out = {"signs": int(ll.sum()), "lines": len(ll)}
    out["lost share of signs"] = G["nlost"].sum() / N
    out["abraded share of signs"] = G["nabr"].sum() / N
    out["lines with any loss"] = float(((G["nlost"] > 0)).mean())
    out["lines: initial loss (not whole)"] = float(I.mean())
    out["lines: final loss (not whole)"] = float(Fi.mean())
    out["lines: both edges lost (not whole)"] = float((I & Fi).mean())
    out["lines: wholly lost"] = float(whole.mean())
    out["lines: medial loss"] = float(M.mean())
    out["lines with abraded signs"] = float((G["nabr"] > 0).mean())
    out["lines with loss and abraded signs"] = float(((G["nabr"] > 0) & (G["nlost"] > 0)).mean())
    rn, ini, fin = G["run_len"], G["run_ini"], G["run_fin"]
    cls = {"initial": ini & ~fin, "final": fin & ~ini, "whole line": ini & fin, "medial": ~ini & ~fin}
    nr = max(1, len(rn))
    for k, m in cls.items():
        out[f"lost runs: {k} (share of runs)"] = float(m.sum() / nr)
    for k, m in cls.items():
        out[f"lost runs: {k} (share of lost signs)"] = float(rn[m].sum() / max(1, rn.sum()))
    for k, m in list(cls.items()) + [("all", np.ones(len(rn), bool))]:
        x = rn[m]
        out[f"run length, {k}: mean"] = float(x.mean()) if len(x) else float("nan")
        out[f"run length, {k}: median"] = float(np.median(x)) if len(x) else float("nan")
        out[f"run length, {k}: 90th pct"] = float(np.percentile(x, 90)) if len(x) else float("nan")
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)):
        out[f"run length {lo}" + ("" if hi == lo else "+" if hi > 99 else f"-{hi}")] = \
            float(((rn >= lo) & (rn <= hi)).sum() / nr)
    fa, fb = a / np.maximum(ll, 1), b / np.maximum(ll, 1)
    out["initial loss: mean share of the line"] = float(fa[I].mean()) if I.any() else float("nan")
    out["final loss: mean share of the line"] = float(fb[Fi].mean()) if Fi.any() else float("nan")
    out["initial loss: mean signs"] = float(a[I].mean()) if I.any() else float("nan")
    out["final loss: mean signs"] = float(b[Fi].mean()) if Fi.any() else float("nan")
    for name, x in (("initial", I), ("final", Fi), ("whole", whole), ("medial", M), ("abraded", G["nabr"] > 0)):
        p1, p0 = _cond(x, x, same)
        out[f"P({name} next | {name})"] = p1
        out[f"P({name} next | not {name})"] = p0
    s1 = same[1:]
    for name, f, x in (("initial", fa, I), ("final", fb, Fi)):
        both = s1 & x[:-1] & x[1:]
        out[f"corr of {name} extent, consecutive lines (both lost)"] = _corr(f[:-1][both], f[1:][both])
        out[f"|diff| of {name} extent, consecutive lines (both lost)"] = \
            float(np.abs(f[:-1][both] - f[1:][both]).mean()) if both.any() else float("nan")
        out[f"corr of {name} extent, consecutive lines (all)"] = _corr(f[:-1][s1], f[1:][s1])
    lf = G["nlost"] / np.maximum(ll, 1)
    out["corr of lost share of the line, consecutive lines"] = _corr(lf[:-1][s1], lf[1:][s1])
    bl = block_lengths(whole, same)
    out["blocks of lost lines: mean length"] = float(bl.mean()) if len(bl) else float("nan")
    for lo, hi in ((1, 1), (2, 2), (3, 5), (6, 10 ** 9)):
        out[f"blocks of lost lines: length {lo}" + ("" if hi == lo else "+" if hi > 99 else f"-{hi}")] = \
            float(((bl >= lo) & (bl <= hi)).mean()) if len(bl) else float("nan")
    an = G["abr_len"]
    out["abraded runs: mean length"] = float(an.mean()) if len(an) else float("nan")
    out["abraded runs: length 1"] = float((an == 1).mean()) if len(an) else float("nan")
    out["abraded runs touching a lost run"] = float(G["abr_adj"].mean()) if len(an) else float("nan")
    # per group (text or chunk)
    g = G["grp"]
    gs = np.r_[0, np.flatnonzero(~same[1:]) + 1] if len(g) else np.zeros(0, np.int64)
    if len(gs):
        dam = np.add.reduceat(G["nlost"] + G["nabr"], gs)
        tot = np.add.reduceat(ll, gs)
        out["groups without damage"] = float((dam == 0).mean())
        r = dam / np.maximum(tot, 1)
        out["damaged share of a damaged group: median"] = float(np.median(r[dam > 0])) if (dam > 0).any() else float("nan")
        out["damaged share of a damaged group: 90th pct"] = float(np.percentile(r[dam > 0], 90)) if (dam > 0).any() else float("nan")
    return out


def block_lengths(flag, same):
    """Lengths of maximal blocks of consecutive flagged lines of one text."""
    if not len(flag):
        return np.zeros(0, np.int64)
    start = flag.copy()
    start[1:] &= ~(flag[:-1] & same[1:])
    end = flag.copy()
    end[:-1] &= ~(flag[1:] & same[1:])
    return np.flatnonzero(end) - np.flatnonzero(start) + 1


def _blocks(flag, same):
    """Maximal blocks of consecutive flagged lines of one text: first line, number of lines."""
    start = flag.copy()
    start[1:] &= ~(flag[:-1] & same[1:])
    end = flag.copy()
    end[:-1] &= ~(flag[1:] & same[1:])
    s = np.flatnonzero(start)
    return s, np.flatnonzero(end) - s + 1


def _block_sum(x, s, k):
    c = np.r_[0, np.cumsum(x)]
    return c[s + k] - c[s]


# ------------------------------------------------------------------ fitting
def stats_path(proc=None):
    return os.path.join(proc or PROC, "damage_stats.pkl")


def save_stats(stats, path=None):
    path = path or stats_path()
    with open(path, "wb") as fh:
        pickle.dump({k: v for k, v in stats.items() if not k.startswith("_")}, fh, protocol=4)
    return path


def load_stats(path=None):
    with open(path or stats_path(), "rb") as fh:
        return pickle.load(fh)


def edge_state(a, b, whole, ll):
    lb = (a * NB + ll - 1) // np.maximum(ll, 1)
    rb = (b * NB + ll - 1) // np.maximum(ll, 1)
    return np.where(whole, WHOLE, lb * (NB + 1) + rb)


def _len_bin(x):
    return np.searchsorted(np.array(LEN_BINS), x, "left") + 1      # 1..5 ; 0 is 'no run above'


def _norm_cum(c, prior=None, k=0.0):
    """Cumulative distributions along the last axis of counts c (+ k pseudo-counts of `prior`)."""
    c = c.astype(np.float64)
    if prior is not None:
        pr = prior.astype(np.float64)
        pr = pr / np.maximum(pr.sum(-1, keepdims=True), 1e-300)
        c = c + k * pr
    tot = c.sum(-1, keepdims=True)
    uni = np.full(c.shape[-1], 1.0 / c.shape[-1])
    p = np.where(tot > 0, c / np.maximum(tot, 1e-300), uni)
    cum = np.cumsum(p, -1)
    cum[..., -1] = 1.0
    return np.minimum(cum, 1.0).astype(np.float32), p


def fit(corpus, split="train", max_lib=6000, seed=0):
    """Fit the damage statistics on the texts of `split` (a split name or a collection of names)."""
    st = fit_flat(flatten(corpus, split), corpus["texts"], max_lib=max_lib, seed=seed)
    st["version"] = corpus.get("version")
    st["split"] = split if isinstance(split, str) else sorted(split)
    return st


def fit_flat(F, texts, max_lib=6000, seed=0):
    rng = np.random.default_rng(seed)
    vs, vl, lt, width = virtual(F)
    G = geometry(vs, vl, lt)
    nl = len(vl)
    same, a, b, whole = G["same"], G["a"], G["b"], G["whole"]
    ut, tix = np.unique(lt, return_inverse=True)
    names = sorted({stratum_of(texts[t]) for t in ut})
    strata = ["all"] + names
    nS = len(strata)
    s_text = np.array([names.index(stratum_of(texts[t])) + 1 for t in ut])
    sl = s_text[tix]                                             # stratum of the line (1..)
    # ---- classes of texts
    T = len(ut)
    t_n = np.bincount(tix, vl, T)
    t_lost = np.bincount(tix, G["nlost"], T)
    t_abr = np.bincount(tix, G["nabr"], T)
    cl_t = np.where(t_lost > 0, 1 + np.searchsorted(np.array(LOST_EDGES), t_lost / t_n, "left"), 0)
    ca_t = np.where(t_abr > 0, 1 + np.searchsorted(np.array(ABR_EDGES), t_abr / t_n, "left"), 0)
    cc_t = cl_t * N_AC + ca_t
    class_w = np.zeros((nS, N_LC * N_AC))
    class_m = np.zeros((nS, N_LC * N_AC))
    np.add.at(class_w, (s_text, cc_t), t_n)
    np.add.at(class_m, (s_text, cc_t), t_lost + t_abr)
    class_w[0], class_m[0] = class_w[1:].sum(0), class_m[1:].sum(0)
    class_rate = np.where(class_w[0] > 0, class_m[0] / np.maximum(class_w[0], 1), 0.0)   # pooled: stable
    cl, ca = cl_t[tix], ca_t[tix]
    s1 = same[1:]
    first = ~same
    last = np.r_[~same[1:], True]

    def counts(keys, dims, w=None):
        idx = np.ravel_multi_index(keys, dims)
        c = np.bincount(idx, w, int(np.prod(dims))).reshape(dims)
        c[0] = c[1:].sum(0)
        return c

    # ---- edge chain
    e = edge_state(a, b, whole, vl)
    E_tr = counts((sl[1:][s1], cl[1:][s1], e[:-1][s1], e[1:][s1]), (nS, N_LC, NE, NE))
    E_mg = counts((sl, cl, e), (nS, N_LC, NE))
    E_fi = counts((sl[first], cl[first], e[first]), (nS, N_LC, NE))
    E_la = counts((sl[last], cl[last], e[last]), (nS, N_LC, NE))
    _, mg_all = _norm_cum(E_mg[0])
    _, tr_all = _norm_cum(E_tr[0], mg_all[:, None, :], 1.0)
    E_cum = np.stack([_norm_cum(E_tr[s], tr_all, SMOOTH if s else 0.0)[0] if s else _norm_cum(tr_all)[0]
                      for s in range(nS)])
    E_init = np.stack([_norm_cum(E_mg[s], mg_all, SMOOTH if s else 0.0)[0] for s in range(nS)])
    E_first = np.stack([_norm_cum(E_fi[s], E_fi[0] + 1e-9, SMOOTH if s else 0.0)[0] for s in range(nS)])
    E_last = np.stack([_norm_cum(E_la[s], E_la[0] + 1e-9, SMOOTH if s else 0.0)[0] for s in range(nS)])
    # ---- interior chain (whole lines count as 'no medial run')
    m = np.minimum(G["nmed"], 2)
    M_tr = counts((sl[1:][s1], cl[1:][s1], m[:-1][s1], m[1:][s1]), (nS, N_LC, 3, 3))
    M_mg = counts((sl, cl, m), (nS, N_LC, 3))
    _, mmg = _norm_cum(M_mg[0])
    _, mtr = _norm_cum(M_tr[0], mmg[:, None, :], 1.0)
    M_cum = np.stack([_norm_cum(M_tr[s], mtr, SMOOTH if s else 0.0)[0] if s else _norm_cum(mtr)[0] for s in range(nS)])
    M_init = np.stack([_norm_cum(M_mg[s], mmg, SMOOTH if s else 0.0)[0] for s in range(nS)])
    med = ~G["run_ini"] & ~G["run_fin"]
    rl, rp, rn = G["run_line"][med], G["run_pos"][med], G["run_len"][med]
    rc = (rp + 0.5 * rn) / vl[rl]
    isfirst = np.r_[True, rl[1:] != rl[:-1]]                      # first medial run of its line
    fr = np.flatnonzero(isfirst)
    first_run = np.full(nl, -1)
    first_run[rl[fr]] = fr
    up = np.full(len(rl), -1)                                     # the first medial run of the line above
    has_up = (rl[fr] > 0) & same[rl[fr]]
    up[fr[has_up]] = first_run[rl[fr[has_up]] - 1]
    cont = up >= 0
    srun = sl[rl]
    lenc = np.minimum(rn, MAX_MED)
    cen_bin = np.minimum((rc * N_CEN).astype(np.int64), N_CEN - 1)
    C_cen = counts((srun[~cont], cen_bin[~cont]), (nS, N_CEN))
    d = rc[cont] - rc[up[cont]]
    d_bin = np.clip(np.rint((d + 1) / 2 * (N_DCEN - 1)).astype(np.int64), 0, N_DCEN - 1)
    C_dcen = counts((srun[cont], d_bin), (nS, N_DCEN))
    prev_bin = np.zeros(len(rl), np.int64)
    prev_bin[cont] = _len_bin(rn[up[cont]])
    C_len = counts((srun, prev_bin, lenc), (nS, len(LEN_BINS) + 2, MAX_MED + 1))
    cen_cum = np.stack([_norm_cum(C_cen[s], C_cen[0] + 1e-9, SMOOTH if s else 0.0)[0] for s in range(nS)])
    dcen_cum = np.stack([_norm_cum(C_dcen[s], C_dcen[0] + 1e-9, SMOOTH if s else 0.0)[0] for s in range(nS)])
    len_cum = np.stack([_norm_cum(C_len[s], C_len[0] + 1e-9, SMOOTH if s else 0.0)[0] for s in range(nS)])
    # ---- abrasion chain: level of the line given the level above and whether this line has a loss
    nab = G["nabr"]
    lev = np.where(nab <= 2, nab, np.where(2 * nab > vl, 4, 3))
    abr_excess = float((nab[lev == 3] - 3).mean()) if (lev == 3).any() else 0.0
    hl = (G["nlost"] > 0).astype(np.int64)
    A_tr = counts((sl[1:][s1], ca[1:][s1], lev[:-1][s1] * 2 + hl[1:][s1], lev[1:][s1]), (nS, N_AC, N_AL * 2, N_AL))
    A_mg = counts((sl, ca, hl, lev), (nS, N_AC, 2, N_AL))
    _, amg = _norm_cum(A_mg[0])
    _, atr = _norm_cum(A_tr[0], np.tile(amg, (1, N_AL, 1)), 1.0)
    A_cum = np.stack([_norm_cum(A_tr[s], atr, SMOOTH if s else 0.0)[0] if s else _norm_cum(atr)[0] for s in range(nS)])
    A_init = np.stack([_norm_cum(A_mg[s], amg, SMOOTH if s else 0.0)[0] for s in range(nS)])
    # abraded signs next to a lost run: share of the free positions touching a lost run that are abraded,
    # in lines that have abraded signs; and the same for the other free positions
    off = np.r_[0, np.cumsum(vl)]
    lid = np.repeat(np.arange(nl), vl)
    lost = vs == LOST
    isfirst_s = np.zeros(len(vs), bool)
    isfirst_s[off[:-1]] = True
    islast_s = np.zeros(len(vs), bool)
    islast_s[off[1:] - 1] = True
    adj = np.zeros(len(vs), bool)
    adj[1:] |= lost[:-1] & ~isfirst_s[1:]
    adj[:-1] |= lost[1:] & ~islast_s[:-1]
    adj &= ~lost
    n_slot = np.bincount(lid[adj], minlength=nl)
    n_slot_abr = np.bincount(lid[adj & (vs == ABRADED)], minlength=nl)
    cand = (G["nabr"] > 0) & (n_slot > 0)
    hit = cand & (n_slot_abr > 0)
    p_slot_line = float(hit.sum() / max(1, cand.sum()))          # a line with abrasion has it at a break edge
    q_slot = float(n_slot_abr[hit].sum() / max(1, n_slot[hit].sum()))   # ... at each break edge of such a line
    # ---- library of connected lost regions
    fa = a / np.maximum(vl, 1)
    fb = b / np.maximum(vl, 1)
    reg = {k: [] for k in ("type", "side", "str", "k", "off")}
    lib_f, lib_len = [], []
    type_counts = np.zeros((nS, len(GAP_TYPES)), np.int64)
    pos = 0

    def add(tcode, side, s0, k, f, ln=None):
        nonlocal pos
        np.add.at(type_counts, (sl[s0], np.full(len(s0), tcode)), 1)
        for st_ in range(1, nS):
            cand = np.flatnonzero(sl[s0] == st_)
            if len(cand) > max_lib:
                cand = np.sort(rng.choice(cand, max_lib, replace=False))
            for j in cand:
                kk = min(int(k[j]), 40)
                sl_ = slice(int(s0[j]), int(s0[j]) + kk)
                reg["type"].append(tcode); reg["side"].append(side); reg["str"].append(st_)
                reg["k"].append(kk); reg["off"].append(pos)
                lib_f.append(f[sl_])
                lib_len.append(ln[sl_] if ln is not None else np.zeros(kk, np.int64))
                pos += kk

    for side, f in ((0, fa), (1, fb)):
        s0, k = _blocks(f > 0, same)
        nw = _block_sum((~whole).astype(np.int64), s0, k)
        s0, k = s0[nw > 0], k[nw > 0]                             # at least one partly preserved line
        df = np.r_[np.diff(f), 0.0]
        tot = f[s0 + k - 1] - f[s0]
        inner = np.ones(nl, bool)
        inner[s0 + k - 1] = False                                 # no difference across the block end
        sg = np.repeat(np.sign(tot), k)
        lines_in = np.repeat(s0, k) + (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k))
        contrary = np.zeros(nl)
        contrary[lines_in] = (df[lines_in] * sg < 0) & inner[lines_in]
        nc = _block_sum(contrary, s0, k)
        corner = (k >= 2) & (np.abs(tot) >= 0.4) & (nc <= 0.25 * (k - 1))
        add(GAP_TYPES.index("left" if side == 0 else "right"), side, s0[~corner], k[~corner], f)
        add(GAP_TYPES.index("corner"), side, s0[corner], k[corner], f)
    s0, k = _blocks(whole, same)
    add(GAP_TYPES.index("lines"), 0, s0, k, fa)
    # interior: the longest medial run of each line; consecutive lines whose runs lie within a quarter
    # of the line of each other form one hole
    o = np.lexsort((-rn, rl))
    keep = o[np.r_[True, rl[o][1:] != rl[o][:-1]]] if len(o) else o
    lc = np.zeros(nl)
    lc[rl[keep]] = rc[keep]
    llen = np.zeros(nl, np.int64)
    llen[rl[keep]] = rn[keep]
    hasm = llen > 0
    link = np.zeros(nl, bool)                                     # line continues the hole of the line above
    link[1:] = hasm[1:] & hasm[:-1] & s1 & (np.abs(lc[1:] - lc[:-1]) <= 0.25)
    s0 = np.flatnonzero(hasm & ~link)
    nxt = np.r_[link[1:], False]
    e0 = np.flatnonzero(hasm & ~nxt)
    k = e0 - s0 + 1
    add(GAP_TYPES.index("hole"), 0, s0[k >= 2], k[k >= 2], lc, llen)
    add(GAP_TYPES.index("medial"), 0, s0[k == 1], k[k == 1], lc, llen)
    type_counts[0] = type_counts[1:].sum(0)

    gap = F["flat"][0] == GAP
    return dict(
        strata=strata, n_texts=int(T), n_lines=int(nl), n_signs=int(F["flat"].shape[1]),
        gap_width_mean=float(width[gap].mean()) if gap.any() else 1.0,
        class_w=class_w, class_rate=class_rate,
        E_cum=E_cum, E_init=E_init, E_first=E_first, E_last=E_last,
        M_cum=M_cum, M_init=M_init, cen_cum=cen_cum, dcen_cum=dcen_cum, len_cum=len_cum,
        A_cum=A_cum, A_init=A_init, q_slot=q_slot, p_slot_line=p_slot_line, abr_excess=abr_excess,
        gap_type_counts=type_counts,
        reg_type=np.array(reg["type"], np.int8), reg_side=np.array(reg["side"], np.int8),
        reg_str=np.array(reg["str"], np.int8), reg_k=np.array(reg["k"], np.int32),
        reg_off=np.array(reg["off"], np.int64),
        lib_f=np.concatenate(lib_f).astype(np.float32) if lib_f else np.zeros(0, np.float32),
        lib_len=np.concatenate(lib_len).astype(np.int32) if lib_len else np.zeros(0, np.int32),
        summary=summarize(G),
    )


# ------------------------------------------------------------------ sampling
class _Prep:
    pass


def _prep(stats, stratum):
    cache = stats.setdefault("_prep", {})
    key = stratum if stratum in stats["strata"] else "all"
    if key in cache:
        return cache[key]
    s = stats["strata"].index(key)
    P = _Prep()
    P.s = s
    w = stats["class_w"][s].astype(np.float64)
    if w.sum() == 0:
        w = stats["class_w"][0].astype(np.float64)
    P.class_all = np.cumsum(w / w.sum()).tolist()
    P.class_all[-1] = 1.0
    wd = w.copy()
    wd[0] = 0.0
    P.class_dam_p = wd / wd.sum()
    P.class_dam = np.cumsum(P.class_dam_p).tolist()
    P.class_dam[-1] = 1.0
    P.class_lograte = np.log(np.maximum(stats["class_rate"], 1e-4))
    P.E = stats["E_cum"][s].tolist()
    P.Ei = stats["E_init"][s].tolist()
    P.Ef = stats["E_first"][s].tolist()
    P.El = stats["E_last"][s].tolist()
    P.M = stats["M_cum"][s].tolist()
    P.Mi = stats["M_init"][s].tolist()
    P.A = stats["A_cum"][s].tolist()
    P.Ai = stats["A_init"][s].tolist()
    P.cen = stats["cen_cum"][s].tolist()
    P.dcen = stats["dcen_cum"][s].tolist()
    P.len = stats["len_cum"][s].tolist()
    P.q_slot = stats["q_slot"]
    P.p_slot_line = stats["p_slot_line"]
    P.abr_log = np.log(stats["abr_excess"] / (1.0 + stats["abr_excess"])) if stats["abr_excess"] > 0 else -np.inf
    # library index
    tc = stats["gap_type_counts"][s].astype(np.float64)
    if tc.sum() == 0:
        tc = stats["gap_type_counts"][0].astype(np.float64)
    P.type_p = tc / tc.sum()
    P.type_cum = np.cumsum(P.type_p).tolist()
    P.type_cum[-1] = 1.0
    P.reg = []
    for t in range(len(GAP_TYPES)):
        ok = stats["reg_type"] == t
        own = np.flatnonzero(ok & (stats["reg_str"] == s)) if s else np.zeros(0, np.int64)
        if len(own) >= 50:
            P.reg.append((own, None))
        else:                         # pooled: strata in proportion to their real number of such regions
            idx = np.flatnonzero(ok)
            cnt = stats["gap_type_counts"][1:, t].astype(np.float64)
            kept = np.bincount(stats["reg_str"][idx], minlength=len(cnt) + 1)[1:]
            wr = (cnt / np.maximum(kept, 1))[stats["reg_str"][idx] - 1]
            P.reg.append((idx, np.cumsum(wr / wr.sum()) if len(idx) else None))
    cache[key] = P
    return P


def _edge_counts(code, n, u):
    """Signs lost at an edge for extent bin `code` (tenths) on lines of n signs: a count whose bin is
    `code` when the line has one (uniform among them); otherwise the bin centre, rounded stochastically,
    but at least one sign and not the whole line (a one-sign line is lost or kept)."""
    lo = (code - 1) * n // NB + 1
    hi = np.minimum(code * n // NB, n - 1)
    x = (code - 0.5) / NB * n + u
    k = np.where(hi >= lo, lo + (u * (hi - lo + 1)).astype(np.int64),
                 np.clip(x.astype(np.int64), 1, np.maximum(n - 1, 1)))
    k = np.where(n == 1, ((code - 0.5) / NB + u).astype(np.int64), k)
    return np.where(code > 0, k, 0)


def _grow(k0, n, f, u):
    """Scale a count of damaged signs of a line by f: linear when shrinking; when growing, the preserved
    share of the line shrinks geometrically, so a partly preserved line keeps something."""
    if f <= 1.0:
        return (f * k0 + u).astype(np.int64)
    return (n * (1.0 - (1.0 - k0 / n) ** f) + u).astype(np.int64)


def _paint(N, starts, ends):
    d = np.zeros(N + 1, np.int64)
    np.add.at(d, starts, 1)
    np.add.at(d, ends, -1)
    return np.cumsum(d[:-1]) > 0


def sample_mask(rng, line_lengths, stats, stratum=None, target_rate=None, allow_clean=False,
                text_start=False, text_end=False):
    """Realistic multi-line damage for a chunk whose lines have `line_lengths` signs (in order).

    Returns (mask, abraded): boolean arrays over the signs of the chunk. mask = damaged positions (to be
    hidden); abraded marks those of them that are damaged but partly legible ('#'), the others are lost.
    stratum       a stratum name of the fitted statistics (None or unknown: pooled)
    target_rate   requested share of masked signs (lost + abraded); the damage pattern is drawn from the
                  texts whose real damage is closest and its extents are scaled to the request.
                  None: the natural amount of a damaged text (or of any text with allow_clean=True,
                  in which case the mask can be empty)
    text_start / text_end   the chunk begins (ends) where the text does: the first (last) line gets the
                  fitted distribution of first (last) lines, which are more often lost
    Deterministic given rng."""
    P = _prep(stats, stratum)
    n = np.asarray(line_lengths, np.int64)
    n = n[n > 0]                                               # an empty line has no sign to mask
    nl = len(n)
    N = int(n.sum())
    if N == 0:
        return np.zeros(0, bool), np.zeros(0, bool)
    if target_rate is None:
        c = bisect_right(P.class_all if allow_clean else P.class_dam, rng.random())
    else:
        w = P.class_dam_p * np.exp(-0.5 * ((P.class_lograte - np.log(max(target_rate, 1e-4))) / 0.7) ** 2)
        cw = np.cumsum(w)
        c = int(np.searchsorted(cw, rng.random() * cw[-1], "right")) if cw[-1] > 0 else N_AC
    c = min(c, N_LC * N_AC - 1)
    cl, ca = divmod(c, N_AC)
    U = rng.random((9, nl))
    off = np.r_[0, np.cumsum(n)]
    e = [0] * nl
    m = [0] * nl
    al = [0] * nl
    rev = text_end and not text_start
    if cl:
        Ec, Mc = P.E[cl], P.M[cl]
        uE, uM = U[0].tolist(), U[1].tolist()
        s = bisect_right((P.Ef if text_start else P.El if rev else P.Ei)[cl], uE[0])
        mm = bisect_right(P.Mi[cl], uM[0])
        e[0], m[0] = s, mm
        for i in range(1, nl):
            s = bisect_right(Ec[s], uE[i])
            mm = bisect_right(Mc[mm], uM[i])
            e[i] = s
            m[i] = mm
        if rev:
            e.reverse()
            m.reverse()
    if ca:
        Ac = P.A[ca]
        uA = U[2].tolist()
        x = bisect_right(P.Ai[ca][1 if (e[0] or m[0]) else 0], uA[0])
        al[0] = x
        for i in range(1, nl):
            x = bisect_right(Ac[x * 2 + (1 if (e[i] or m[i]) else 0)], uA[i])
            al[i] = x
    ea = np.array(e)
    whole = ea == WHOLE
    lb, rb = np.divmod(np.where(whole, 0, ea), NB + 1)
    kl0 = np.where(whole, n, _edge_counts(lb, n, U[3]))
    kr0 = _edge_counts(rb, n, U[4])
    kr0 = np.where(~whole & (n > 1) & (kl0 + kr0 >= n), np.maximum(n - 1 - kl0, 0), kr0)
    # medial runs: (line, centre, length); a run under a run of the line above follows it
    ml, mc, mlen = [], [], []
    if cl and any(m):
        prev_i, prev_c, prev_len = -2, 0.0, 0
        cen, dcen, lenc = P.cen, P.dcen, P.len
        for i in range(nl):
            if not m[i] or e[i] == WHOLE or n[i] < 3:
                continue
            r = rng.random(3)
            if prev_i == i - 1:
                cc = prev_c + (bisect_right(dcen, r[0]) + r[1] - 0.5) * 2.0 / (N_DCEN - 1) - 1.0
                cc = min(max(cc, 0.0), 1.0)
                L = bisect_right(lenc[int(np.searchsorted(LEN_BINS, prev_len)) + 1], r[2])
            else:
                cc = (bisect_right(cen, r[0]) + r[1]) / N_CEN
                L = bisect_right(lenc[0], r[2])
            ml.append(i); mc.append(cc); mlen.append(max(L, 1))
            prev_i, prev_c, prev_len = i, cc, max(L, 1)
            if m[i] > 1:
                r = rng.random(3)
                ml.append(i); mc.append((bisect_right(cen, r[0]) + r[1]) / N_CEN)
                mlen.append(max(bisect_right(lenc[0], r[2]), 1))
    ml = np.array(ml, np.int64)
    mc = np.array(mc)
    mlen0 = np.array(mlen, np.int64)
    um = rng.random(len(ml))
    ala = np.array(al)
    ka0 = ala.copy()
    if ca and (ala >= 3).any():
        half = n // 2
        k3 = np.minimum(3 + (np.log(1.0 - U[5]) / P.abr_log).astype(np.int64), np.maximum(half, 3))
        k4 = half + 1 + (U[5] * (n - half)).astype(np.int64)
        ka0 = np.where(ala == 3, k3, np.where(ala == 4, k4, ala))
    ka0 = np.minimum(ka0, n)

    def render(f):
        if f == 1.0:
            kl, kr, L = kl0, kr0, mlen0
            ka = ka0
        else:
            kl = np.minimum(_grow(kl0, n, f, U[6]), n)
            kr = _grow(kr0, n, f, U[7])
            L = (f * mlen0 + um).astype(np.int64)
            ka = _grow(ka0, n, f ** 0.5 if f > 1.0 else f, U[8])
        full = kl + kr >= n
        kl = np.where(full, n, kl)
        kr = np.where(full, 0, kr)
        starts = [off[:-1], off[1:] - kr]
        ends = [off[:-1] + kl, off[1:]]
        if len(ml):                                            # inside the preserved interior of the line
            nn = n[ml]
            p0 = kl[ml] + 1
            p1 = nn - kr[ml] - 1
            L = np.minimum(L, p1 - p0)
            s = np.clip(np.rint(mc * nn - 0.5 * L).astype(np.int64), p0, np.maximum(p1 - L, p0))
            ok = L > 0
            starts.append(off[ml[ok]] + s[ok])
            ends.append(off[ml[ok]] + s[ok] + L[ok])
        lost = _paint(N, np.concatenate(starts), np.concatenate(ends))
        free = n - np.add.reduceat(lost, off[:-1])
        ka = np.minimum(ka, free)
        return lost, ka, int(N - free.sum() + ka.sum())

    lost, ka, total = render(1.0)
    deficit = 0
    if target_rate is not None:
        want = max(1, int(round(target_rate * N)))
        tol = max(1, int(0.01 * N))
        lo, hi, f = 0.0, None, 1.0
        for _ in range(14):
            if abs(total - want) <= tol:
                break
            if total < want:
                lo = f
            else:
                hi = f
            if hi is None:
                if f >= 64.0 or total == 0:
                    break
                f = min(64.0, max(2.0 * f, f * want / total))
            elif lo == 0.0:
                f = min(0.5 * f, f * want / total)
                if f < 1e-3:
                    f = 0.0
            else:
                f = (lo * hi) ** 0.5
            lost, ka, total = render(f)
            if f == 0.0:
                break
        if total > want + tol:                                 # between two roundings: the lower one
            lost, ka, total = render(lo)
        deficit = want - total
    elif total == 0 and not allow_clean:
        deficit = int(min(N, rng.geometric(0.4)))
    # abraded signs: positions next to a lost run first (with the fitted probability), then anywhere
    abr = np.zeros(N, bool)
    lid = None
    if ka.any():
        lid = np.repeat(np.arange(nl), n)
        r = rng.random((2, N))
        flag = (rng.random(nl) < P.p_slot_line)[lid]
        score = r[0]
        adj = np.zeros(N, bool)
        adj[1:] = lost[:-1]
        adj[off[:-1]] = False
        nx = np.zeros(N, bool)
        nx[:-1] = lost[1:]
        nx[off[1:] - 1] = False
        score[(adj | nx) & flag & (r[1] < P.q_slot)] -= 1.0
        score[lost] = 2.0
        order = np.lexsort((score, lid))
        sel = order[(np.arange(N) - off[lid[order]]) < ka[lid[order]]]
        abr[sel] = True
        abr &= ~lost
    if deficit > 0:                                            # scaling cannot reach the request
        masked = lost | abr
        fm = ~masked
        free = np.add.reduceat(fm, off[:-1])
        tot_free = int(free.sum())
        deficit = min(deficit, tot_free)
        if lid is None:
            lid = np.repeat(np.arange(nl), n)
        rank = np.cumsum(fm) - (np.cumsum(fm)[off[:-1]] - fm[off[:-1]])[lid]      # rank among the free signs of the line
        if deficit >= 0.1 * N and nl > 1 and rng.random() < 0.85:
            # a fragment: the same share of what is left of every line goes at one edge
            c = np.rint(np.cumsum(free) * (deficit / tot_free)).astype(np.int64)
            add = np.diff(np.r_[0, c])
            left = rng.random() < 0.5
        else:                                                  # further lines from the top or the bottom
            order = np.arange(nl) if rng.random() < 0.5 else np.arange(nl)[::-1]
            cum = np.cumsum(free[order])
            add = np.zeros(nl, np.int64)
            add[order] = np.clip(deficit - (cum - free[order]), 0, free[order])
            left = (order[0] == 0) if nl > 1 else rng.random() < 0.5
        take = fm & ((rank <= add[lid]) if left else (rank > (free - add)[lid]))
        lost |= take
        abr &= ~lost
    return lost | abr, abr


def sample_gap(rng, line_lengths, stats, max_signs, stratum=None, kind=None, eligible=None, min_signs=1,
               tries=50):
    """ONE connected realistic lost region for an evaluation instance: a real region (edge loss, corner,
    block of lost lines, interior hole, medial run) drawn from the library of the fitted statistics and
    transplanted onto consecutive lines of the chunk, one run per line, at most `max_signs` signs in
    total (a larger region is cut to a window of consecutive lines; a single longer run is shortened).

    kind       one of GAP_TYPES to force the type (default: the real frequencies of the types)
    eligible   optional boolean array over the signs: every masked sign must be eligible (for instance
               clean gold signs); positions are retried up to `tries` times
    Returns (mask, info) with info = dict(type, side, lines, start, end) (runs as positions within their
    lines), or (None, None) if no admissible placement was found."""
    P = _prep(stats, stratum)
    n = np.asarray(line_lengths, np.int64)
    nl = len(n)
    N = int(n.sum())
    if N == 0 or max_signs < 1:
        return None, None
    off = np.r_[0, np.cumsum(n)]
    reg_k, reg_off, reg_side = stats["reg_k"], stats["reg_off"], stats["reg_side"]
    for _ in range(tries):
        t = GAP_TYPES.index(kind) if kind is not None else bisect_right(P.type_cum, rng.random())
        idx, cum = P.reg[t]
        if not len(idx):
            if kind is not None:
                return None, None
            continue
        j = idx[int(rng.integers(len(idx)))] if cum is None else idx[min(int(np.searchsorted(cum, rng.random())), len(idx) - 1)]
        k, o, side = int(reg_k[j]), int(reg_off[j]), int(reg_side[j])
        f = stats["lib_f"][o:o + k]
        ln = stats["lib_len"][o:o + k]
        if k > nl:
            j0 = int(rng.integers(k - nl + 1))
            f, ln, k = f[j0:j0 + nl], ln[j0:j0 + nl], nl
        i0 = int(rng.integers(nl - k + 1))
        nn = n[i0:i0 + k]
        if t <= 3:                                             # edge, corner, lost lines
            ext = np.clip(np.rint(f * nn).astype(np.int64), 1, nn)
            s = np.zeros(k, np.int64) if side == 0 else nn - ext
            en = s + ext
        else:
            L = np.minimum(ln, nn - 2)
            s = np.clip(np.rint(f * nn - 0.5 * L).astype(np.int64), 1, np.maximum(nn - L - 1, 1))
            en = s + L
            bad = np.flatnonzero(L < 1)
            if len(bad):                                       # a line too short for a medial run ends the hole
                if bad[0] == 0:
                    continue
                s, en, k = s[:bad[0]], en[:bad[0]], int(bad[0])
        size = en - s
        if size.sum() > max_signs:                             # a window of consecutive lines of the region
            j0 = int(rng.integers(k))
            if size[j0] > max_signs:
                if side == 0 and t <= 3:
                    en[j0] = s[j0] + max_signs
                elif t <= 3:
                    s[j0] = en[j0] - max_signs
                else:
                    en[j0] = s[j0] + max_signs
                j1 = j0 + 1
                if t == 3:                                     # a lost line cut short is an edge loss
                    t = 0
            else:
                tot, j1 = 0, j0
                while j1 < k and tot + size[j1] <= max_signs:
                    tot += size[j1]
                    j1 += 1
            s, en, i0, k = s[j0:j1], en[j0:j1], i0 + j0, j1 - j0
        if (en - s).sum() < min_signs:
            continue
        lines = np.arange(i0, i0 + k)
        mask = _paint(N, off[lines] + s, off[lines] + en)
        if eligible is not None and not eligible[mask].all():
            continue
        return mask, {"type": GAP_TYPES[t], "side": "left" if side == 0 else "right", "lines": lines,
                      "start": s, "end": en}
    return None, None


def render_ascii(line_lengths, mask, abraded=None, max_width=60):
    """One row per line: '#' lost, '+' abraded, '.' preserved."""
    out, o = [], 0
    for k in line_lengths:
        k = int(k)
        row = np.full(k, ".")
        row[mask[o:o + k]] = "#"
        if abraded is not None:
            row[abraded[o:o + k]] = "+"
        out.append("".join(row[:max_width]) + (">" if k > max_width else ""))
        o += k
    return out


def subset(G, line_mask):
    """Geometry restricted to the lines of whole texts (line_mask must select complete texts)."""
    idx = np.cumsum(line_mask) - 1
    out = {k: G[k][line_mask] for k in ("ll", "grp", "a", "b", "whole", "nmed", "nlost", "nabr")}
    g = out["grp"]
    same = np.zeros(len(g), bool)
    same[1:] = g[1:] == g[:-1]
    out["same"] = same
    r = line_mask[G["run_line"]]
    out.update(run_line=idx[G["run_line"][r]], run_pos=G["run_pos"][r], run_len=G["run_len"][r],
               run_ini=G["run_ini"][r], run_fin=G["run_fin"][r])
    r = line_mask[G["abr_line"]]
    out.update(abr_line=idx[G["abr_line"][r]], abr_pos=G["abr_pos"][r], abr_len=G["abr_len"][r], abr_adj=G["abr_adj"][r])
    return out
