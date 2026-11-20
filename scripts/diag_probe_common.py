"""Torch-free builders shared by the diagnostic probes (train and dev splits only; never the test split).

Instance sets (all on the dev split, known width):
  ev1 / ev5    the L = 1 / L = 5 instances of data/proc/evalset_dev.pkl.gz (300 each)
  ext1 / ext5  one further gap per eligible dev chunk, drawn with cun.evalset's rule (clean gold signs,
               chunk has >= L + 4 signs), seed 101
  multi1       up to 12 single-sign gaps per dev chunk (distinct clean positions), seed 102 (probes 3-5)
  evL          all evalset_dev instances with L = 2..20 (iterative decoding, probe 3)
  far1         every clean single-sign position of the dev chunks whose trigram (left neighbour, sign, right
               neighbour) recurs in the chunk only at a distance >= 128 signs (probe 5; rare, so exhaustive)
"""
import gzip, os, pickle, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN, MASK, READ_MASK, READ_NONE  # noqa: E402
from cun.devset import DEV_SEED, dev_noisings, load_chunks  # noqa: E402
from cun.noise import D_UNK, J_UNK, NoiseConfig, noise_chunk  # noqa: E402

OUT = os.path.join(ROOT, "results", "diag", "probes")
PLANES = ("sign", "read", "join", "det")
TRUNC = (4, 8, 16, 32, 64)
DROP = (8, 16, 32, 64)
KEEP_AFTER_DROP = 16


def load_corpus():
    return pickle.load(gzip.open(os.path.join(ROOT, "data", "proc", "corpus.pkl.gz"), "rb"))


# ------------------------------------------------------------------ gap instances
def clean_starts(a, L):
    n = a.shape[1]
    if n < L + 4:
        return np.zeros(0, int)
    ok = (a[0] >= FIRST_REAL_SIGN) & (a[4] == 0)
    return np.flatnonzero(np.convolve(ok.astype(int), np.ones(L, int), "valid") == L)


def build_instances(ev, corpus=None):
    sets = {"ev1": [dict(x) for x in ev["instances"] if x["L"] == 1],
            "ev5": [dict(x) for x in ev["instances"] if x["L"] == 5],
            "evL": [dict(x) for x in ev["instances"] if x["L"] >= 2]}
    rng = np.random.default_rng(101)
    for L in (1, 5):
        out = []
        for ci in rng.permutation(len(ev["chunks"])):
            st = clean_starts(ev["chunks"][ci][2], L)
            if len(st):
                out.append({"chunk": int(ci), "start": int(rng.choice(st)), "L": L, "extra": 0})
        sets[f"ext{L}"] = out
    rng = np.random.default_rng(102)
    out = []
    for ci in range(len(ev["chunks"])):
        st = clean_starts(ev["chunks"][ci][2], 1)
        if len(st):
            for s in rng.choice(st, min(12, len(st)), replace=False):
                out.append({"chunk": ci, "start": int(s), "L": 1, "extra": 0})
    sets["multi1"] = out
    out = []
    for ci in range(len(ev["chunks"])):
        if ev["chunks"][ci][2].shape[1] < 131:
            continue
        for s in clean_starts(ev["chunks"][ci][2], 1):
            x = {"chunk": ci, "start": int(s), "L": 1, "extra": 0}
            ri = repeat_info(ev, x)
            if ri is not None and ri["dist"] >= 128:
                out.append(x)
    sets["far1"] = out
    if corpus is not None:                               # novelty against the TRAIN texts (cun.evalset rule)
        for k in ("ext1", "ext5", "multi1", "far1"):
            for x, v in zip(sets[k], evalset.novelty(corpus, ev["chunks"], sets[k])):
                x["novel"] = bool(v)
    return sets


def meta(ev, x):
    ti, lg, a = ev["chunks"][x["chunk"]]
    n, s, L = a.shape[1], x["start"], x["L"]
    ends = np.flatnonzero(a[2] == 6)
    prev = ends[ends < s]
    lo = int(prev[-1]) + 1 if len(prev) else 0
    nxt = ends[ends >= s + L - 1]
    hi = int(nxt[0]) + 1 if len(nxt) else n
    return {"chunk": x["chunk"], "text": int(ti), "lang": ev["langs"][lg], "n": int(n), "start": s, "L": L,
            "novel": x.get("novel"), "line_lo": lo, "line_hi": hi}


# ------------------------------------------------------------------ row transforms
def base_row(ev, x):
    r = evalset.gap_input(ev, x, "known")
    return r


def copy_row(r):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def hide_read(r):
    r = copy_row(r)
    g, M = r["gap"], r["M"]
    keep = r["read"][g:g + M].copy()
    r["read"][:] = READ_NONE
    r["read"][g:g + M] = keep
    return r


def hide_jd(r):
    r = copy_row(r)
    r["join"][:] = J_UNK
    r["det"][:] = D_UNK
    return r


def set_lang(r, lg):
    r = copy_row(r)
    r["lang"] = lg
    return r


def crop(r, lo, hi):
    r2 = {k: r[k][lo:hi].copy() for k in PLANES}
    r2.update(lang=r["lang"], gap=r["gap"] - lo, M=r["M"], text=r["text"])
    return r2


def mask_positions(r, pos):
    r = copy_row(r)
    r["sign"][pos] = MASK
    r["read"][pos] = READ_MASK
    r["join"][pos] = J_UNK
    r["det"][pos] = D_UNK
    return r


def extra_mask(r, p, rng):
    """Mask a further share p of the identified signs outside the gap (iid, as the 'iid' pattern does)."""
    g, M = r["gap"], r["M"]
    ok = r["sign"] >= FIRST_REAL_SIGN
    ok[g:g + M] = False
    pos = np.flatnonzero(ok & (rng.random(len(ok)) < p))
    return mask_positions(r, pos), len(pos)


def repeat_info(ev, x):
    """Other occurrences in the chunk of the trigram (left neighbour, gold sign, right neighbour)."""
    a = ev["chunks"][x["chunk"]][2][0].astype(np.int64)
    s, n = x["start"], len(a)
    if s < 1 or s > n - 2 or a[s - 1] < FIRST_REAL_SIGN or a[s + 1] < FIRST_REAL_SIGN:
        return None
    hit = np.flatnonzero((a[:-2] == a[s - 1]) & (a[1:-1] == a[s]) & (a[2:] == a[s + 1])) + 1
    hit = hit[hit != s]
    uni = np.flatnonzero(a == a[s])
    uni = uni[uni != s]
    return {"centres": hit, "dist": int(np.abs(hit - s).min()) if len(hit) else -1,
            "dist_earlier": int((s - hit[hit < s]).min()) if (hit < s).any() else -1,
            "n_rep": len(hit), "uni_dist": int(np.abs(uni - s).min()) if len(uni) else -1}


def gap_tasks(ev, sets, n_lang):
    """Yield (task name, instance set, decode mode(s), [(instance index, row)], extra per-instance info)."""
    langs = ev["langs"]
    sux, akk = langs.index("sux"), langs.index("akk")
    for key in ("ev1", "ext1", "ev5", "ext5"):
        inst = sets[key]
        rows = [base_row(ev, x) for x in inst]
        mt = [meta(ev, x) for x in inst]
        idx = list(range(len(inst)))
        yield "base", key, list(zip(idx, rows)), None
        yield "a_no_readings", key, [(i, hide_read(r)) for i, r in zip(idx, rows)], None
        yield "b_no_join_det", key, [(i, hide_jd(r)) for i, r in zip(idx, rows)], None
        yield "c_signs_only", key, [(i, hide_jd(hide_read(r))) for i, r in zip(idx, rows)], None
        yield "d_lang_unknown", key, [(i, set_lang(r, n_lang)) for i, r in zip(idx, rows)], None
        yield "e_lang_wrong", key, [(i, set_lang(r, akk if r["lang"] == sux else sux)) for i, r in zip(idx, rows)], None
        for k in TRUNC:
            yield f"f_trunc{k}", key, [(i, crop(r, max(0, r["gap"] - k), r["gap"] + r["M"] + k))
                                       for i, r in zip(idx, rows)], None
        yield "g_own_line", key, [(i, crop(r, m["line_lo"], m["line_hi"])) for i, r, m in zip(idx, rows, mt)], None
        for k in DROP:
            yield f"h_drop_first{k}", key, [(i, crop(r, k, len(r["sign"]))) for i, r in zip(idx, rows)
                                            if r["gap"] >= k + KEEP_AFTER_DROP], None
            yield f"h_drop_last{k}", key, [(i, crop(r, 0, len(r["sign"]) - k)) for i, r in zip(idx, rows)
                                           if len(r["sign"]) - r["gap"] - r["M"] >= k + KEEP_AFTER_DROP], None
        if key in ("ext1", "ext5"):                       # probe 4: heavier damage elsewhere in the chunk
            for p in (0.15, 0.40):
                rng = np.random.default_rng(400 + int(100 * p))
                rr = [extra_mask(r, p, rng) for r in rows]
                yield f"m_extra_mask{int(100 * p)}", key, [(i, r2) for i, (r2, _) in zip(idx, rr)], \
                    {"n_masked": [c for _, c in rr]}
    # probes 3 and 5 on the large single-sign sets
    for key in ("multi1", "far1"):
        inst = sets[key]
        rows = [base_row(ev, x) for x in inst]
        idx = list(range(len(inst)))
        yield "base", key, list(zip(idx, rows)), None
        rng = np.random.default_rng(500)
        rep = [repeat_info(ev, x) for x in inst]
        abl, ctl = [], []
        for i, (r, ri) in enumerate(zip(rows, rep)):
            if ri is None or not ri["n_rep"]:
                continue
            abl.append((i, mask_positions(r, ri["centres"])))
            ok = r["sign"] >= FIRST_REAL_SIGN
            ok[r["gap"]] = False
            ok[ri["centres"]] = False
            cand = np.flatnonzero(ok)
            if len(cand) >= ri["n_rep"]:
                ctl.append((i, mask_positions(r, rng.choice(cand, ri["n_rep"], replace=False))))
        yield "r_mask_repeats", key, abl, None
        yield "r_mask_random_control", key, ctl, None
    # probe 3: iterative decoding, every L >= 2 instance of the dev evaluation set
    inst = sets["evL"]
    yield "base", "evL", [(i, base_row(ev, x)) for i, x in enumerate(inst)], None


# ------------------------------------------------------------------ probe 1: noisings
def noisings(chunks, n_lang):
    """{set name: [(chunk index, noised dict)]}:
    devlog   exactly cun.devset.dev_noisings (600 chunks; span t=0.15 and transliteration; what cun.train logs)
    fix      every chunk, pattern p in 0..4, t in {0.15, 0.40}, language known, planes drawn as in training
    nat      every chunk, pattern p in 0..4, t ~ Beta(2,4) and everything else as in training"""
    out = {}
    for p, i, d in dev_noisings(chunks, n_lang):
        out.setdefault(f"devlog_p{p}", []).append((i, d))
    rng = np.random.default_rng(DEV_SEED + 1)
    for p in range(5):
        for t in (0.15, 0.40):
            out[f"fix_p{p}_t{int(100 * t)}"] = [(i, noise_chunk(rng, *c, n_lang=n_lang, cfg=NoiseConfig(p_lang_known=1.0),
                                                                pattern=p, t=t)) for i, c in enumerate(chunks)]
    rng = np.random.default_rng(DEV_SEED + 2)
    for p in range(5):
        out[f"nat_p{p}"] = [(i, noise_chunk(rng, *c, n_lang=n_lang, cfg=NoiseConfig(), pattern=p))
                            for i, c in enumerate(chunks)]
    return out


def train_sample(train_chunks, n, seed=777):
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(len(train_chunks), n, replace=False))
    return pick, [train_chunks[i] for i in pick]


if __name__ == "__main__":                               # dry run on the login node (no torch)
    import collections, time
    t0 = time.time()
    ev = evalset.load(ROOT, "dev")
    corpus = load_corpus()
    print("corpus", round(time.time() - t0), "s")
    n_lang = len(corpus["langs"])
    assert corpus["langs"] == ev["langs"]
    sets = build_instances(ev, corpus)
    for k, v in sets.items():
        print(k, len(v), "texts", len({ev["chunks"][x["chunk"]][0] for x in v}),
              "novel", np.mean([bool(x.get("novel")) for x in v]).round(3))
    tot = 0
    for name, key, rows, info in gap_tasks(ev, sets, n_lang):
        for i, r in rows[:50]:
            g, M = r["gap"], r["M"]
            assert (r["sign"][g:g + M] == MASK).all() and len({len(r[k]) for k in PLANES}) == 1, name
            assert 0 <= g and g + M <= len(r["sign"]), name
        L = np.mean([len(r["sign"]) for _, r in rows]) if rows else 0
        tot += len(rows) * (1 if key not in ("ev5", "ext5", "evL") else 6)
        print(f"{name:24s} {key:7s} n {len(rows):6d} mean len {L:6.1f}")
    print("row-forwards approx", tot)
    rep = [repeat_info(ev, x) for x in sets["multi1"]]
    d = np.array([r["dist"] if r else -2 for r in rep])
    print("multi1 trigram: not testable", (d == -2).sum(), "no repeat", (d == -1).sum(), "<128", ((d > 0) & (d < 128)).sum(),
          ">=128", (d >= 128).sum(), "texts with >=128", len({ev['chunks'][x['chunk']][0] for x, v in zip(sets['multi1'], d) if v >= 128}))
    dev_chunks = load_chunks(corpus, "dev")
    assert len(dev_chunks) == len(ev["chunks"]) and all((a[0] == c[2][0]).all() for a, c in zip(dev_chunks, ev["chunks"]))
    tr = load_chunks(corpus, "train")
    pick, trs = train_sample(tr, len(dev_chunks))
    for nm, ch in (("dev", dev_chunks), ("train sample", trs)):
        ln = np.array([len(c[0]) for c in ch])
        print(nm, len(ch), "signs", ln.sum(), "median", np.median(ln), "mean", ln.mean().round(1),
              collections.Counter(corpus["langs"][c[4]] for c in ch).most_common(4))
        nz = noisings(ch, n_lang)
        for k, v in nz.items():
            print("  ", k, len(v), "sign targets", sum(int((d["s_tg"] >= 0).sum()) for _, d in v),
                  "read targets", sum(int((d["r_tg"] >= 0).sum()) for _, d in v), "max len", max(len(d["sign"]) for _, d in v))
    print("total", round(time.time() - t0), "s")
