"""Noising for sign-level masked diffusion (after Stoicheia's train/noising.py).

Per chunk a pattern is drawn (weights in NoiseConfig):
  span      contiguous runs of signs (geometric length): a break in the clay
  word      whole words (join plane 'w' marks word ends)
  iid       scattered single signs
  elastic   a run of L signs replaced by M >= L MASK slots; targets = the L signs then BLANKs
            (restoration of a gap whose width is uncertain, like [...] or [x x x])
  translit  no sign is masked; every reading is hidden -> the model transliterates bare signs
Rate t ~ Beta(a, b) clipped to [t_min, t_max]. Masked signs also hide their reading.
Plane availability (reading, join, det) is drawn per chunk: fully known, fully unknown, or patchy
(per-position Bernoulli with rate ~ U(0,1)); hidden planes become targets where a label exists.
Signs that are unknown in the source (<x>, <gap>, <unk-sign>) are never targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .consts import BLANK, GAP, MASK, N_DET, N_JOIN, READ_MASK, READ_NONE, UNK_SIGN, X_SIGN

IGN = -100
J_UNK = N_JOIN
D_UNK = N_DET


@dataclass
class NoiseConfig:
    w_span: float = 0.30
    w_word: float = 0.20
    w_iid: float = 0.15
    w_elastic: float = 0.15
    w_translit: float = 0.20
    t_min: float = 0.05
    t_max: float = 0.80
    beta_a: float = 2.0
    beta_b: float = 4.0
    span_mean: float = 3.0
    elastic_max: int = 20             # as long as the longest evaluated gap
    p_read_full: float = 0.5
    p_read_none: float = 0.3
    p_plane_full: float = 0.5        # join / det
    p_plane_none: float = 0.3
    p_lang_known: float = 0.8
    # ---- round 3 (all off by default: the round-2 noise is unchanged)
    w_gap: float = 0.0               # N1: known-width gaps, L log-uniform 1..gap_max, one or more per chunk
    gap_max: int = 20
    p_gap_extra: float = 0.5         # N1: a gap chunk also gets light span noise (t ~ U(0.02, 0.15))
    p_reveal: float = 0.0            # N1: in span / word / gap chunks, reveal a random share of the masked
    #                                  signs as input (reading hidden), as met in iterative decoding
    w_damage: float = 0.0            # R2: masks shaped like real breaks (cun.damage.sample_mask)
    no_restored_targets: bool = False   # D1: signs restored by the editor are input only, never targets
    p_gap_planes: float = 0.0        # P1: per chunk, probability that lost signs keep partial information:
    #                                  the joiner and / or the determinative flag of each lost sign stays visible
    #                                  (a break whose word boundaries or determinative survive)
    w_r2s: float = 0.0               # RS: reading -> sign: scattered signs hidden while their reading is shown


def _avail(rng, n, p_full, p_none):
    u = rng.random()
    if u < p_full:
        return np.ones(n, bool)
    if u < p_full + p_none:
        return np.zeros(n, bool)
    return rng.random(n) < rng.random()


DAMAGE_STATS = None          # set by the trainer (cun.damage.load_stats()) when w_damage > 0
S_UNK = 3                    # status plane: 0 clean, 1 damaged, 2 restored by the editor, 3 unknown / masked
W_MAX = 31                   # width head: classes 0..W_MAX real signs in an elastic gap


def noise_chunk(rng, sign, read, join, det, lang, brk=None, *, n_lang, cfg: NoiseConfig, pattern=None, t=None):
    """sign/read/join/det (and brk: 0 clean, 1 damaged, 2 restored): int arrays of one chunk.
    Returns dict of input and target arrays. Patterns: 0 span, 1 word, 2 iid, 3 elastic, 4 translit,
    5 known-width gaps (N1), 6 real-break masks (R2)."""
    n = len(sign)
    if brk is None:
        brk = np.zeros(n, np.int64)
    targetable = (sign != UNK_SIGN) & (sign != X_SIGN) & (sign != GAP)
    if cfg.no_restored_targets:
        targetable &= brk != 2
    w = np.array([cfg.w_span, cfg.w_word, cfg.w_iid, cfg.w_elastic, cfg.w_translit, cfg.w_gap, cfg.w_damage,
                  cfg.w_r2s])
    if pattern is None:
        pattern = rng.choice(len(w), p=w / w.sum())
    if t is None:
        t = float(np.clip(rng.beta(cfg.beta_a, cfg.beta_b), cfg.t_min, cfg.t_max))
    m = np.zeros(n, bool)
    want = max(1, int(round(t * n)))
    if pattern == 0:                                   # span
        tries = 0
        while m.sum() < want and tries < 100:
            L = min(n, 1 + rng.geometric(1 / cfg.span_mean))
            s = rng.integers(0, max(1, n - L + 1))
            m[s:s + L] = True
            tries += 1
    elif pattern == 1:                                 # word
        ends = np.flatnonzero(np.isin(join, [0, 6]))   # 'w' or line end
        if not len(ends) or ends[-1] != n - 1:
            ends = np.r_[ends, n - 1]                    # a chunk cut mid-word still ends a word
        starts = np.r_[0, ends[:-1] + 1]
        order = rng.permutation(len(starts))
        for k in order:
            if m.sum() >= want:
                break
            m[starts[k]:ends[k] + 1] = True
    elif pattern in (2, 7):                            # iid (7: reading -> sign, the reading stays visible)
        m = rng.random(n) < t
        if pattern == 7:
            m &= read != READ_NONE
    elif pattern == 5:                                 # known-width gaps of 1..gap_max signs
        k = 1 + int(rng.random() < 0.3) + int(rng.random() < 0.1)
        for _ in range(k):
            L = int(min(n, round(float(np.exp(rng.uniform(0.0, np.log(cfg.gap_max)))))))
            s0 = int(rng.integers(0, max(1, n - L + 1)))
            m[s0:s0 + L] = True
        if rng.random() < cfg.p_gap_extra:
            want2 = int(round(rng.uniform(0.02, 0.15) * n))
            tries = 0
            base = int(m.sum())
            while m.sum() - base < want2 and tries < 50:
                L = min(n, 1 + rng.geometric(1 / cfg.span_mean))
                s0 = rng.integers(0, max(1, n - L + 1))
                m[s0:s0 + L] = True
                tries += 1
    elif pattern == 6:                                 # masks shaped like real breaks
        from . import damage
        ends = np.flatnonzero(join == 6)
        if not len(ends) or ends[-1] != n - 1:
            ends = np.r_[ends, n - 1]
        lost, abr = damage.sample_mask(rng, np.diff(np.r_[-1, ends]), DAMAGE_STATS, target_rate=t)
        m = lost | (abr & (rng.random(n) < 0.5))
    m &= targetable
    if pattern == 2 and not m.any() and targetable.any():
        m[rng.choice(np.flatnonzero(targetable))] = True

    s_in, r_in = sign.copy(), read.copy()
    j_in, d_in = join.copy(), det.copy()
    s_tg = np.full(n, IGN, np.int64)
    r_tg = np.full(n, IGN, np.int64)
    j_tg = np.full(n, IGN, np.int64)
    d_tg = np.full(n, IGN, np.int64)

    if pattern == 4:                                   # transliteration: hide every reading
        r_in[:] = READ_NONE
        lab = (read != READ_NONE) & targetable
        r_tg[lab] = read[lab]
    else:
        rev = np.zeros(n, bool)                        # revealed: the sign is input, its other planes hidden
        if cfg.p_reveal and pattern in (0, 1, 5, 6) and rng.random() < cfg.p_reveal:
            rev = m & (rng.random(n) < rng.random())
        mm = m & ~rev
        s_in[mm] = MASK
        s_tg[mm] = sign[mm]
        if pattern == 7:                               # the reading is given: only the sign is predicted
            keep_read = mm.copy()
        else:
            keep_read = np.zeros(n, bool)
            r_in[mm] = READ_MASK
            r_in[rev] = READ_NONE
            lab = m & (read != READ_NONE)
            r_tg[lab] = read[lab]
        ra = _avail(rng, n, cfg.p_read_full, cfg.p_read_none)
        hide = ~ra & ~m
        r_in[hide] = READ_NONE
        lab = hide & (read != READ_NONE) & targetable
        r_tg[lab] = read[lab]
    ja = _avail(rng, n, cfg.p_plane_full, cfg.p_plane_none)
    da = _avail(rng, n, cfg.p_plane_full, cfg.p_plane_none)
    mj = md = m                                        # planes hidden at the lost signs
    if cfg.p_gap_planes and pattern not in (4, 7) and rng.random() < cfg.p_gap_planes:
        u = rng.random()                               # joiner, determinative flag, or both stay visible
        if u < 0.67:
            mj = np.zeros(n, bool)
        if u > 0.33:
            md = np.zeros(n, bool)
    if pattern == 7:
        mj = md = np.zeros(n, bool)
    j_in[~ja | mj] = J_UNK
    d_in[~da | md] = D_UNK
    j_tg[~ja | mj] = join[~ja | mj]
    d_tg[~da | md] = det[~da | md]
    st_in = brk.astype(np.int64).copy()                # status plane (used only by models that have it)
    st_in[(sign == X_SIGN) | (sign == GAP) | (sign == UNK_SIGN)] = S_UNK
    if pattern != 4:
        st_in[m] = S_UNK
    lang_in = lang if rng.random() < cfg.p_lang_known else n_lang

    out = dict(sign=s_in, read=r_in, join=j_in, det=d_in, s_tg=s_tg, r_tg=r_tg, j_tg=j_tg, d_tg=d_tg,
               status=st_in, slot=np.zeros(n, np.int64), w_tg=np.full(n, IGN, np.int64))
    if pattern == 3 and n >= 2:                        # elastic gap (changes length)
        L = int(rng.integers(1, min(cfg.elastic_max, n) + 1))
        s = int(rng.integers(0, n - L + 1))
        extra = int(rng.integers(0, int(np.ceil(0.3 * L)) + 3))
        M = L + extra
        gap = {
            "sign": np.full(M, MASK), "read": np.full(M, READ_MASK), "join": np.full(M, J_UNK),
            "det": np.full(M, D_UNK),
            "s_tg": np.r_[np.where(targetable[s:s + L], sign[s:s + L], IGN), np.full(extra, BLANK)],
            "r_tg": np.r_[np.where((read[s:s + L] != READ_NONE) & targetable[s:s + L], read[s:s + L], IGN),
                          np.full(extra, IGN)],
            "j_tg": np.r_[join[s:s + L], np.full(extra, IGN)],
            "d_tg": np.r_[det[s:s + L], np.full(extra, IGN)],
            "status": np.full(M, S_UNK), "slot": np.ones(M, np.int64),          # O1: elastic slots are flagged
            "w_tg": np.r_[min(L, W_MAX), np.full(M - 1, IGN)],                    # O1: width target at the first slot
        }
        for k in out:
            out[k] = np.r_[out[k][:s], gap[k].astype(out[k].dtype), out[k][s + L:]]
    out["lang"] = np.full(len(out["sign"]), lang_in)
    out["pattern"] = pattern
    return out
