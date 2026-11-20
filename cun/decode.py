"""Decoding without retraining (PLAN_v3 Stage 1): beam search over the iterative fill, pseudo-likelihood
rescoring, remasking, unknown width by marginalisation over candidate widths, retrieval bonuses.

Everything works on 'bases': the row dicts of cun.evalset.gap_input (sign / read / join / det arrays with
the gap slots masked, lang, gap = first slot, M = number of slots). A 'fill' is an int array of M signs.

Drop-in decoders (same call as the methods of cun.evaluate.Decoder: fn(rows, cond) -> list of dicts
sign / read / join / det; extra keys 'cands', 'scores' carry the ranked whole-gap candidates):
    eng = Engine(model, ev, "cuda")
    decode_known(eng, rows, B=20, topk=20)                 known width
    decode_unknown(eng, rows, widths=None, B=20, ...)      unknown width / width range
    KnownDecoder(eng, ...), UnknownDecoder(eng, ...)       callables fn(rows, cond)
Building blocks:
    beam_search      B hypotheses over the most-confident-first fill (B = 1 is cun.evaluate's 'iter')
    pll              pseudo-log-likelihood per slot (each slot masked in turn; stride = cheaper folds)
    score_fill       chain log-probability of a given fill along the most-confident-first path
    remask           ReMDM-style: re-mask the least confident slots of a fill and re-predict
    blank_width_scores, width_prior, width_candidates, set_width, finalize
Scores are natural-log probabilities among the allowed signs (real signs; + BLANK for width scores).
"""
from __future__ import annotations

import math
import zlib

import numpy as np

try:                                            # the scoring scripts import this module without torch
    import torch
except ImportError:                             # pragma: no cover
    torch = None
_no_grad = torch.no_grad() if torch is not None else (lambda f: f)

from .consts import BLANK, FIRST_REAL_SIGN, MASK, READ_MASK, READ_NONE
from .noise import D_UNK, J_UNK

K_MAX = 30                                      # most slots ever tried for one gap


# ----------------------------------------------------------------------------- torch-free helpers
def half(text_id):
    """Dev half of a text: 'A' (tuning) or 'B' (confirmation), by an even / odd hash of the text id."""
    return "A" if zlib.crc32(str(int(text_id)).encode()) % 2 == 0 else "B"


def key(fill):
    return np.asarray(fill, np.int16).tobytes()


def unkey(k):
    return np.frombuffer(k, np.int16).astype(np.int64)


def set_width(base, k):
    """The same gap with k slots (all masked)."""
    g, M = base["gap"], base["M"]
    out = {}
    for name, f in (("sign", MASK), ("read", READ_MASK), ("join", J_UNK), ("det", D_UNK)):
        x = base[name]
        out[name] = np.r_[x[:g], np.full(k, f, np.int64), x[g + M:]]
    if base.get("status") is not None:
        out["status"] = np.r_[base["status"][:g], np.full(k, 3, np.int64), base["status"][g + M:]]
    if base.get("slot") is not None:                     # a candidate width is decoded as a known-width gap
        out["slot"] = np.zeros(len(out["sign"]), np.int64)
    out.update(lang=base["lang"], gap=g, M=k)
    return out


def width_prior(M, max_l=20):
    """log P(L = k | M slots) for k = 1..M under the elastic-gap construction of pretraining
    (cun.noise: L ~ U{1..max_l}, extra ~ U{0..ceil(0.3 L) + 2}, M = L + extra). -inf where impossible."""
    p = np.zeros(M)
    for k in range(1, min(M, max_l) + 1):
        top = int(math.ceil(0.3 * k)) + 2
        if M - k <= top:
            p[k - 1] = 1.0 / (top + 1)
    if p.sum() == 0:                             # more slots than pretraining ever gave: flat
        p[:] = 1.0
    with np.errstate(divide="ignore"):
        return np.log(p / p.sum())


def width_estimate(L, r, rng):
    """An editor's width estimate w with the true width inside w +- r*w (uniform over such integers)."""
    ws = [w for w in range(1, 4 * L + 2) if abs(L - w) <= r * w + 1e-9]
    return int(rng.choice(ws))


def width_candidates(w, r, cap=K_MAX):
    """Candidate widths given the estimate w +- r*w (at most `cap` slots)."""
    return [k for k in range(1, cap + 1) if abs(k - w) <= r * w + 1e-9]


def width_conditions(instances, seed=7):
    """Per instance the slots given and the admissible widths in the four width conditions:
    'unknown' (L + extra slots, any width), 'r50' / 'r25' (an estimate w with the truth inside w +- 50% /
    25%: widths within that range, slots = the largest), 'exact'. Deterministic (seeded)."""
    rng = np.random.default_rng(seed)
    out = []
    for x in instances:
        L = x["L"]
        d = {"unknown": (L + x["extra"], list(range(1, L + x["extra"] + 1))), "exact": (L, [L])}
        for name, r in (("r25", 0.25), ("r50", 0.5)):
            c = width_candidates(width_estimate(L, r, rng), r)
            d[name] = (max(c), c)
        out.append(d)
    return out


# ----------------------------------------------------------------------------- model access
class Engine:
    """Batched access to the model for gap hypotheses. run() takes items (base, gs): gs is the gap as an
    int array of base['M'] signs with MASK in the open slots (filled slots are fed back with their reading
    hidden, as in cun.evaluate)."""

    def __init__(self, model, ev, device="cuda", max_tokens=140_000, max_rows=1024):
        self.model, self.ev, self.device = model, ev, device
        self.max_tokens, self.max_rows = max_tokens, max_rows
        V, NR = model.cfg.n_sign, model.cfg.n_read
        allow = torch.zeros(V, dtype=torch.bool)
        allow[FIRST_REAL_SIGN:] = True
        self.allow_known = allow.to(device)
        a2 = allow.clone()
        a2[BLANK] = True
        self.allow_unknown = a2.to(device)
        R = torch.zeros(V, NR, dtype=torch.bool)
        R[:, 2:] = True
        for s, rs in ev["sign_reads"].items():
            R[int(s)] = False
            R[int(s), rs] = True
        self.R = R.to(device)
        self.rows = 0                              # forward rows so far (cost accounting)

    @_no_grad
    def run(self, items, topk=0, blank=False, targets=None, planes=False, raw=0):
        """Returns numpy arrays aligned with items, padded to the widest gap:
        topv / topi (n, M, topk) log-probabilities and ids of the best allowed signs per slot;
        blank (n, M) log P(BLANK) with BLANK allowed; tgt (n, M) log-probability of targets[i][slot];
        read / join / det (n, M) the other planes given the signs of gs (planes=True);
        rawv / rawi (n, M, raw) and rawt (n, M): unnormalised logits (calibration)."""
        n = len(items)
        lens = np.array([len(b["sign"]) for b, _ in items])
        order = np.argsort(lens, kind="stable")
        Mx = max(len(gs) for _, gs in items)
        out = {}
        if topk:
            out["topv"] = np.full((n, Mx, topk), -np.inf, np.float32)
            out["topi"] = np.zeros((n, Mx, topk), np.int64)
        if blank:
            out["blank"] = np.zeros((n, Mx), np.float32)
        if targets is not None:
            out["tgt"] = np.zeros((n, Mx), np.float32)
        if planes:
            for k in ("read", "join", "det"):
                out[k] = np.zeros((n, Mx), np.int64)
        if raw:
            out["rawv"] = np.zeros((n, Mx, raw), np.float32)
            out["rawi"] = np.zeros((n, Mx, raw), np.int64)
            out["rawt"] = np.zeros((n, Mx), np.float32)
        dev = self.device
        s = 0
        while s < n:
            e = s + 1
            while e < n and e - s < self.max_rows and (e - s + 1) * lens[order[e]] <= self.max_tokens:
                e += 1
            idx = order[s:e]
            s = e
            bs, T = len(idx), int(lens[idx[-1]])
            Mb = max(len(items[i][1]) for i in idx)
            sign = np.zeros((bs, T), np.int64)
            read = np.zeros((bs, T), np.int64)
            join = np.full((bs, T), J_UNK, np.int64)
            det = np.full((bs, T), D_UNK, np.int64)
            lang = np.zeros((bs, T), np.int64)
            status = np.full((bs, T), 3, np.int64)       # planes of round-3 models (ignored by older ones)
            slot = np.zeros((bs, T), np.int64)
            seg = np.zeros((bs, T), np.int64)
            gi = np.zeros((bs, Mb), np.int64)
            gsb = np.full((bs, Mb), FIRST_REAL_SIGN, np.int64)
            tg = np.full((bs, Mb), FIRST_REAL_SIGN, np.int64)
            for r, i in enumerate(idx):
                b, gs = items[i]
                nl, g, m = len(b["sign"]), b["gap"], len(gs)
                assert m == b["M"], (m, b["M"])
                sign[r, :nl], read[r, :nl], join[r, :nl], det[r, :nl] = b["sign"], b["read"], b["join"], b["det"]
                sign[r, g:g + m] = gs
                read[r, g:g + m] = np.where(gs == MASK, READ_MASK, READ_NONE)
                lang[r, :] = b["lang"]
                if b.get("status") is not None:
                    status[r, :nl] = b["status"]
                else:
                    status[r, :nl] = 0
                status[r, g:g + m] = 3
                if b.get("slot") is not None:
                    slot[r, :nl] = b["slot"]
                seg[r, nl:] = 10_000 + np.arange(T - nl)
                gi[r, :m] = g + np.arange(m)
                gsb[r, :m] = gs
                if targets is not None:
                    tg[r, :m] = targets[i]
            batch = {k: torch.from_numpy(v).to(dev) for k, v in
                     (("sign", sign), ("read", read), ("join", join), ("det", det), ("lang", lang), ("seg", seg),
                      ("status", status), ("slot", slot))}
            git = torch.from_numpy(gi).to(dev)
            ar = torch.arange(bs, device=dev)[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.model(batch, heads=())["hidden"][ar, git]
                lg = self.model.h_sign(h).float()
                if planes:
                    lr = self.model.h_read(h).float()
                    lj = self.model.h_join(h).float()
                    ld = self.model.h_det(h).float()
            self.rows += bs
            lk = lg.masked_fill(~self.allow_known, float("-inf"))
            if topk or targets is not None:
                lp = torch.log_softmax(lk, -1)
                if topk:
                    v, ix = lp.topk(topk, -1)
                    out["topv"][idx, :Mb] = v.cpu().numpy()
                    out["topi"][idx, :Mb] = ix.cpu().numpy()
                if targets is not None:
                    tgt = torch.from_numpy(tg).to(dev)
                    out["tgt"][idx, :Mb] = lp.gather(-1, tgt[..., None])[..., 0].cpu().numpy()
            if blank:
                lb = torch.log_softmax(lg.masked_fill(~self.allow_unknown, float("-inf")), -1)[..., BLANK]
                out["blank"][idx, :Mb] = lb.cpu().numpy()
            if planes:
                sg = torch.from_numpy(gsb).to(dev)
                out["read"][idx, :Mb] = lr.masked_fill(~self.R[sg], float("-inf")).argmax(-1).cpu().numpy()
                out["join"][idx, :Mb] = lj.argmax(-1).cpu().numpy()
                out["det"][idx, :Mb] = ld.argmax(-1).cpu().numpy()
            if raw:
                v, ix = lk.topk(raw, -1)
                out["rawv"][idx, :Mb] = v.cpu().numpy()
                out["rawi"][idx, :Mb] = ix.cpu().numpy()
                if targets is not None:
                    tgt = torch.from_numpy(tg).to(dev)
                    out["rawt"][idx, :Mb] = lk.gather(-1, tgt[..., None])[..., 0].cpu().numpy()
        return out


def _chunks(n, size):
    return [(s, min(n, s + size)) for s in range(0, n, size)]


# ----------------------------------------------------------------------------- decoders
def beam_search(eng, bases, B=5, n_slots=1, n_cand=None, init=None, pool_rows=8192):
    """Beam search over the iterative fill. Each step every hypothesis is expanded at its most confident
    open slot (the n_slots most confident ones) with that slot's n_cand (default B) best signs; the B
    hypotheses with the highest sum of log-probabilities survive (hypotheses of one gap always have the
    same number of choices, so length normalisation cannot change the ranking within a width).
    init: optional partial fills (MASK = open). Returns per gap a list of (fill, score, slot log-probs),
    best first; B = 1, n_slots = 1 is exactly the 'iter' decoder of cun.evaluate.
    Cost: sum over steps of the beam size = about B * M forward rows per gap."""
    n_cand = n_cand or B
    N = len(bases)
    hyps = []
    for i, b in enumerate(bases):
        gs = np.full(b["M"], MASK, np.int64) if init is None else np.asarray(init[i], np.int64).copy()
        hyps.append([(gs, 0.0, np.zeros(len(gs), np.float32))])
    nm = np.array([int((h[0][0] == MASK).sum()) for h in hyps])
    order = np.argsort(nm, kind="stable")
    per = max(1, pool_rows // B)
    for c0, c1 in _chunks(N, per):
        act = [int(i) for i in order[c0:c1] if nm[i] > 0]
        while act:
            items = [(bases[i], h[0]) for i in act for h in hyps[i]]
            out = eng.run(items, topk=n_cand)
            pos, nxt = 0, []
            for i in act:
                H = hyps[i]
                ex = []
                for hi, (gs, sc, lp) in enumerate(H):
                    tv = out["topv"][pos]
                    pos += 1
                    masked = np.flatnonzero(gs == MASK)
                    sl = masked[np.argsort(-tv[masked, 0], kind="stable")[:n_slots]]
                    for s_ in sl:
                        for c in range(n_cand):
                            ex.append((sc + float(tv[s_, c]), hi, int(s_), c))
                ex.sort(key=lambda t: -t[0])
                base_pos = pos - len(H)
                new, seen = [], set()
                for v, hi, s_, c in ex:
                    gs, sc, lp = H[hi]
                    g2 = gs.copy()
                    g2[s_] = out["topi"][base_pos + hi, s_, c]
                    kb = g2.tobytes()
                    if kb in seen:
                        continue
                    seen.add(kb)
                    lp2 = lp.copy()
                    lp2[s_] = out["topv"][base_pos + hi, s_, c]
                    new.append((g2, v, lp2))
                    if len(new) >= B:
                        break
                hyps[i] = new
                if (new[0][0] == MASK).any():
                    nxt.append(i)
            act = nxt
    return hyps


def pll(eng, bases, fills, stride=None, chunk=30000):
    """Pseudo-log-likelihood per slot: log P(fill[j] | context, the other slots of the fill), each slot
    masked in turn (M rows per fill). stride = s masks every s-th slot at once (min(s, M) rows; cheaper,
    each slot sees only the slots of the other folds). Returns a list of float arrays."""
    items, tg, meta = [], [], []
    for i, (b, x) in enumerate(zip(bases, fills)):
        x = np.asarray(x, np.int64)
        k = len(x)
        groups = [[j] for j in range(k)] if stride is None else [list(range(f, k, stride)) for f in range(min(stride, k))]
        for grp in groups:
            gs = x.copy()
            gs[grp] = MASK
            items.append((b, gs))
            tg.append(x)
            meta.append((i, grp))
    res = [np.zeros(len(x), np.float32) for x in fills]
    for c0, c1 in _chunks(len(items), chunk):
        out = eng.run(items[c0:c1], targets=tg[c0:c1])
        for r in range(c0, c1):
            i, grp = meta[r]
            res[i][grp] = out["tgt"][r - c0, grp]
    return res


def score_fill(eng, bases, fills, pool_rows=8192):
    """Chain log-probability of given fills along the most-confident-first path (what beam_search with
    n_slots = 1 assigns to the same fill). M rows per fill. Returns (scores, slot log-probs)."""
    N = len(bases)
    fills = [np.asarray(x, np.int64) for x in fills]
    gs = [np.full(len(x), MASK, np.int64) for x in fills]
    lps = [np.zeros(len(x), np.float32) for x in fills]
    order = np.argsort([len(x) for x in fills], kind="stable")
    for c0, c1 in _chunks(N, pool_rows):
        act = [int(i) for i in order[c0:c1] if len(fills[i])]
        while act:
            out = eng.run([(bases[i], gs[i]) for i in act], topk=1, targets=[fills[i] for i in act])
            nxt = []
            for r, i in enumerate(act):
                masked = np.flatnonzero(gs[i] == MASK)
                s_ = masked[np.argmax(out["topv"][r, masked, 0])]
                lps[i][s_] = out["tgt"][r, s_]
                gs[i][s_] = fills[i][s_]
                if len(masked) > 1:
                    nxt.append(i)
            act = nxt
    return [float(l.sum()) for l in lps], lps


def finalize(eng, bases, fills, chunk=30000):
    """Readings (constrained to the sign's attested readings), joiners and determinatives of complete
    fills: one row per fill. Returns dicts in the format of cun.evaluate."""
    res = []
    items = [(b, np.asarray(x, np.int64)) for b, x in zip(bases, fills)]
    for c0, c1 in _chunks(len(items), chunk):
        out = eng.run(items[c0:c1], planes=True)
        for r in range(c0, c1):
            x = items[r][1]
            m = len(x)
            res.append({"sign": x.copy(), "read": out["read"][r - c0, :m].copy(),
                        "join": out["join"][r - c0, :m].copy(), "det": out["det"][r - c0, :m].copy()})
    return res


def remask(eng, bases, fills, slot_lp=None, rounds=2, frac=0.25, conf="fill", accept="always", B=1):
    """ReMDM-style remasking without retraining. Each round the k = max(1, round(frac * M)) least
    confident slots of the current fill are re-masked together and re-predicted (most confident first)
    given the other slots. Gaps of one sign are returned unchanged.
    conf = 'fill': confidence = log-probability of the sign when it was filled in (slot_lp required);
    conf = 'pll' : confidence = pseudo-log-likelihood given the complete fill (M more rows per round).
    accept = 'always' | 'pll' (keep the new fill only if its summed pseudo-log-likelihood is higher; pll
    confidences only). Returns (fills, slot confidences, number of gaps changed)."""
    cur = [np.asarray(x, np.int64).copy() for x in fills]
    multi = [i for i, x in enumerate(cur) if len(x) > 1]
    sub = [bases[i] for i in multi]
    if conf == "pll":
        cf = pll(eng, sub, [cur[i] for i in multi])
    else:
        cf = [np.asarray(slot_lp[i], np.float32).copy() for i in multi]
    changed = np.zeros(len(cur), bool)
    for _ in range(rounds):
        init = []
        for j, i in enumerate(multi):
            k = max(1, int(round(frac * len(cur[i]))))
            gs = cur[i].copy()
            gs[np.argsort(cf[j], kind="stable")[:k]] = MASK
            init.append(gs)
        new = beam_search(eng, sub, B=B, init=init)
        nf = [h[0][0] for h in new]
        if conf == "pll":
            ncf = pll(eng, sub, nf)
        else:
            ncf = []
            for j, i in enumerate(multi):
                lp = cf[j].copy()
                m = init[j] == MASK
                lp[m] = new[j][0][2][m]
                ncf.append(lp)
        for j, i in enumerate(multi):
            if accept == "pll" and not ncf[j].sum() > cf[j].sum():
                continue
            if not np.array_equal(nf[j], cur[i]):
                changed[i] = True
            cur[i], cf[j] = nf[j], ncf[j]
    conf_out = [np.zeros(len(x), np.float32) for x in cur]
    for j, i in enumerate(multi):
        conf_out[i] = cf[j]
    return cur, conf_out, int(changed.sum())


def blank_width_scores(eng, bases):
    """The width score of cun.evaluate's width_first for k = 1..M: one pass over the M masked slots;
    score(k) = sum_{i<k} log(1 - pB_i) + sum_{i>=k} log pB_i with pB the probability of BLANK."""
    out = eng.run([(b, np.full(b["M"], MASK, np.int64)) for b in bases], blank=True)
    res = []
    for r, b in enumerate(bases):
        p = np.clip(np.exp(out["blank"][r, :b["M"]].astype(np.float64)), 1e-6, 1 - 1e-6)
        real, bl = np.log1p(-p), np.log(p)
        res.append(np.array([real[:k].sum() + bl[k:].sum() for k in range(1, b["M"] + 1)]))
    return res


# ----------------------------------------------------------------------------- drop-in decoders
def combine_width_score(blank, chain, pll_sum, k, prior, w):
    """Width score from its components (w: dict of weights; see results/decode/REPORT.md):
    w['blank'] * BLANK score + w['chain'] * chain log-prob / k**w['norm'] + w['pll'] * PLL / k**w['norm']
    + w['prior'] * log prior."""
    nk = float(k) ** w.get("norm", 0.0)
    s = w.get("blank", 0.0) * blank + w.get("chain", 0.0) * chain / nk + w.get("pll", 0.0) * pll_sum / nk
    if w.get("prior", 0.0):
        s += w["prior"] * max(prior, -30.0)
    return s


def decode_known(eng, rows, B=20, topk=20, n_slots=1, rescore=None, n_rescore=5, bonus=None):
    """Known width. Beam search (B hypotheses), optional pseudo-likelihood rescoring of the best n_rescore
    candidates (rescore = weight a: score = chain + a * PLL), optional additive bonuses
    (bonus: per row a dict {key(fill): bonus}, e.g. from retrieval; fills not in the beam are added and
    scored with score_fill). Returns per row the dict of the best candidate with
    'cands' (list of sign arrays, best first, at most topk) and 'scores'."""
    hyps = beam_search(eng, rows, B=B, n_slots=n_slots)
    cands = [[(h[0], h[1]) for h in H] for H in hyps]
    if bonus is not None:
        extra = [(i, unkey(k)) for i, d in enumerate(bonus) for k in (d or {})
                 if len(k) == 2 * rows[i]["M"] and all(key(c[0]) != k for c in cands[i])]
        if extra:
            sc, _ = score_fill(eng, [rows[i] for i, _ in extra], [x for _, x in extra])
            for (i, x), s in zip(extra, sc):
                cands[i].append((x, s))
        cands = [sorted(((x, s + (bonus[i] or {}).get(key(x), 0.0)) for x, s in C), key=lambda t: -t[1])
                 for i, C in enumerate(cands)]
    if rescore:
        pairs = [(i, j) for i, C in enumerate(cands) for j in range(min(n_rescore, len(C))) if rows[i]["M"] > 1]
        pl = pll(eng, [rows[i] for i, _ in pairs], [cands[i][j][0] for i, j in pairs])
        new = {}
        for (i, j), p in zip(pairs, pl):
            new.setdefault(i, []).append((cands[i][j][0], cands[i][j][1] + rescore * float(p.sum())))
        for i, top in new.items():
            rest = cands[i][len(top):]
            top.sort(key=lambda t: -t[1])
            cands[i] = top + rest
    fin = finalize(eng, rows, [C[0][0] for C in cands])
    for d, C in zip(fin, cands):
        d["cands"] = [x for x, _ in C[:topk]]
        d["scores"] = np.array([s for _, s in C[:topk]])
    return fin


def decode_unknown(eng, rows, widths=None, weights=None, n_widths=None, width_bonus=None, B=20, topk=20, **kw):
    """Unknown width by marginalisation over candidate widths. rows: gaps with M slots (the most the gap
    can hold); widths: per row the admissible widths (default 1..M; a width range restricts them).
    Stage 1: every admissible width k is decoded as a known-width gap with the iterative fill (B = 1) and
    scored by combine_width_score(BLANK score of the M slots, chain log-probability, pseudo-log-likelihood,
    prior); n_widths = n keeps only the n widths with the best BLANK score for this (cheaper).
    Stage 2: the best width is decoded with decode_known(B, topk, **kw).
    weights: dict for combine_width_score (default: the configuration recommended in REPORT.md).
    width_bonus: per row a dict {k: bonus added to the score of width k} (e.g. parallels retrieved for a
    gap of k signs). kw goes to decode_known (bonus = per row {key(fill): bonus} may hold fills of any width).
    Defaults: weights = DEFAULT_WIDTH_WEIGHTS (chain + prior), right when the M slots come from the
    elastic-gap construction; with a width range given by an editor use weights = {'chain': 1.0}.
    The returned dicts also carry 'width' and 'width_scores' {k: score}."""
    w = dict(DEFAULT_WIDTH_WEIGHTS if weights is None else weights)
    bl = blank_width_scores(eng, rows)
    need_fill = bool(w.get("chain") or w.get("pll"))
    cand = []
    for i, r in enumerate(rows):
        ks = list(range(1, r["M"] + 1)) if widths is None else [k for k in widths[i] if 1 <= k <= r["M"]]
        ks = ks or [min(r["M"], max(1, min(widths[i])))]
        if n_widths and len(ks) > n_widths:
            ks = sorted(sorted(ks, key=lambda k: -bl[i][k - 1])[:n_widths])
        cand.append(ks)
    comp = [{k: [bl[i][k - 1], 0.0, 0.0] for k in ks} for i, ks in enumerate(cand)]
    if need_fill:
        pairs = [(i, k) for i, ks in enumerate(cand) for k in ks if len(ks) > 1]
        sub = [set_width(rows[i], k) for i, k in pairs]
        hyps = beam_search(eng, sub, B=1)
        pl = pll(eng, sub, [h[0][0] for h in hyps]) if w.get("pll") else [np.zeros(1)] * len(pairs)
        for (i, k), h, p in zip(pairs, hyps, pl):
            comp[i][k][1], comp[i][k][2] = h[0][1], float(p.sum())
    best, scores = [], []
    for i, r in enumerate(rows):
        pr = width_prior(r["M"])
        sc = {k: combine_width_score(c[0], c[1], c[2], k, pr[k - 1], w) for k, c in comp[i].items()}
        if width_bonus is not None and width_bonus[i]:
            sc = {k: v + width_bonus[i].get(k, 0.0) for k, v in sc.items()}
        scores.append(sc)
        best.append(max(sc, key=sc.get))
    res = decode_known(eng, [set_width(r, k) for r, k in zip(rows, best)], B=B, topk=topk, **kw)
    for d, k, sc in zip(res, best, scores):
        d["width"], d["width_scores"] = k, sc
    return res


# chain log-probability of the fill + log prior of the width (results/decode/REPORT.md, section 4)
DEFAULT_WIDTH_WEIGHTS = {"chain": 1.0, "prior": 1.0}


class KnownDecoder:
    """fn(rows, cond) for cun.evaluate's plan: KnownDecoder(eng, B=20)."""

    def __init__(self, eng, **kw):
        self.eng, self.kw = eng, kw

    def __call__(self, rows, cond="known"):
        return decode_known(self.eng, rows, **self.kw)


class UnknownDecoder:
    """fn(rows, cond) for cun.evaluate's plan (cond 'unknown'): UnknownDecoder(eng, B=20)."""

    def __init__(self, eng, **kw):
        self.eng, self.kw = eng, kw

    def __call__(self, rows, cond="unknown"):
        return decode_unknown(self.eng, rows, **self.kw)


def load_engine(root, ckpt="ckpt/base-long/best.pt", split="dev", **kw):
    """(engine, evalset) for a checkpoint; refuses a checkpoint from another corpus version."""
    import os

    from . import evalset
    from .model import CunConfig, CunEncoder
    ck = torch.load(os.path.join(root, ckpt), map_location="cuda", weights_only=False)
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    ev = evalset.load(root, split, version=evalset.corpus_version(root))
    if ck.get("corpus_version") != ev["version"]:
        raise SystemExit(f"checkpoint corpus {ck.get('corpus_version')} != evalset corpus {ev['version']}")
    return Engine(model, ev, "cuda", **kw), ev
