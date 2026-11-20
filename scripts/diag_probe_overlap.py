"""Probe 1 follow-up (torch-free): is the train-dev gap memorisation of the text itself, or the duplicated
material that train chunks share with OTHER train texts (dev texts had their near-duplicates removed from train)?

For every chunk of the probe-1 samples: overlap = share of its sign 6-grams (identified signs only) that occur
in at least one other train text (for a train chunk its own text is not counted). The per-chunk losses of
gen_gap.pkl.gz (best.pt and pre_anneal.pt; span + word + iid at t = 0.15) are then compared at matched overlap.
"""
import gzip, json, os, pickle, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_probe_common as C
from cun.consts import FIRST_REAL_SIGN
from cun.devset import CHUNK

W = 6
P = np.uint64(1000003)
corpus = C.load_corpus()
texts = corpus["texts"]


def chunks_with_text(split):
    by_text = {}
    for ti, a in corpus["lines"]:
        if texts[ti]["split"] == split:
            by_text.setdefault(ti, []).append(a[0])
    out = []
    for ti, arrs in by_text.items():
        a = np.concatenate(arrs)
        for s in range(0, len(a), CHUNK):
            if len(a[s:s + CHUNK]) >= 2:
                out.append((ti, a[s:s + CHUNK].astype(np.int64)))
    return by_text, out


def hashes(a):
    """(hash, valid) of every W-gram of a sign array."""
    n = len(a) - W + 1
    if n <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    h = np.zeros(n, np.uint64)
    bad = np.zeros(n, np.int64)
    for j in range(W):
        seg = a[j:j + n]
        h = h * P + (seg + 2).astype(np.uint64)
        bad += seg < FIRST_REAL_SIGN
    return h, bad == 0


by_text, train_chunks = chunks_with_text("train")
_, dev_chunks = chunks_with_text("dev")
hs, ts = [], []
for ti, arrs in by_text.items():                       # W-grams of whole train texts (lines concatenated)
    h, ok = hashes(np.concatenate(arrs).astype(np.int64))
    h = np.unique(h[ok])
    hs.append(h); ts.append(np.full(len(h), ti))
hs, ts = np.concatenate(hs), np.concatenate(ts)
uh, cnt = np.unique(hs, return_counts=True)            # number of distinct train texts per W-gram
print("train 6-grams", len(hs), "distinct", len(uh), "in >= 2 texts", (cnt >= 2).sum())


def overlap(a, need):
    h, ok = hashes(a)
    h = h[ok]
    if not len(h):
        return np.nan
    i = np.searchsorted(uh, h)
    i[i >= len(uh)] = 0
    c = np.where(uh[i] == h, cnt[i], 0)
    return float((c >= need).mean())


g = pickle.load(gzip.open(os.path.join(C.OUT, "gen_gap.pkl.gz"), "rb"))
pick = g["train_pick"]
ov = {"dev": np.array([overlap(a, 1) for _, a in dev_chunks]),
      "train": np.array([overlap(train_chunks[i][1], 2) for i in pick])}
assert all(len(train_chunks[i][1]) == l for i, l in zip(pick, g["chunk_len"]["train"]))
assert all(len(a) == l for (_, a), l in zip(dev_chunks, g["chunk_len"]["dev"]))
res = {"W": W}
L = ["## Probe 1 follow-up. Train-dev gap at matched overlap with other train texts\n",
     f"overlap = share of a chunk's sign {W}-grams found in at least one OTHER train text. Sign CE / accuracy: span + word + iid at t = 0.15 pooled "
     "(fix_p0/p1/p2_t15). Chunks without a valid 6-gram are left out.\n",
     f"Median overlap: dev {np.nanmedian(ov['dev']):.2f}, train sample {np.nanmedian(ov['train']):.2f}; mean: dev {np.nanmean(ov['dev']):.2f}, train {np.nanmean(ov['train']):.2f}.\n"]
for ck in ("best", "pre_anneal"):
    L += [f"\n### {ck}.pt\n", "| overlap | chunks dev / train | sign targets dev / train | sign CE dev | sign CE train | gap (dev - train) | sign acc dev | sign acc train |", "|---|---|---|---|---|---|---|---|"]
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01), (0, 1.01)):
        v = {}
        for sp in ("dev", "train"):
            m = (ov[sp] >= lo) & (ov[sp] < hi)
            r = sum(g["res"][(ck, sp, f"fix_p{p}_t15")] for p in range(3))[m]
            v[sp] = (int(m.sum()), int(r[:, 0, 1].sum()), r[:, 0, 0].sum() / max(r[:, 0, 1].sum(), 1), r[:, 0, 2].sum() / max(r[:, 0, 1].sum(), 1))
        lab = f"{lo:.1f}-{min(hi, 1.0):.1f}" if (lo, hi) != (0, 1.01) else "all"
        res[f"{ck}/{lab}"] = v
        L.append(f"| {lab} | {v['dev'][0]} / {v['train'][0]} | {v['dev'][1]} / {v['train'][1]} | {v['dev'][2]:.3f} | {v['train'][2]:.3f} | {v['dev'][2] - v['train'][2]:+.3f} | {100 * v['dev'][3]:.1f} | {100 * v['train'][3]:.1f} |")
# same, short chunks only
L += ["\n### best.pt, chunks of at most 32 signs\n", "| overlap | chunks dev / train | sign CE dev | sign CE train | gap | sign acc dev | sign acc train |", "|---|---|---|---|---|---|---|"]
for lo, hi in ((0, 0.3), (0.3, 0.9), (0.9, 1.01)):
    v = {}
    for sp in ("dev", "train"):
        m = (ov[sp] >= lo) & (ov[sp] < hi) & (g["chunk_len"][sp] <= 32)
        r = sum(g["res"][("best", sp, f"fix_p{p}_t15")] for p in range(3))[m]
        v[sp] = (int(m.sum()), r[:, 0, 0].sum() / max(r[:, 0, 1].sum(), 1), r[:, 0, 2].sum() / max(r[:, 0, 1].sum(), 1))
    L.append(f"| {lo:.1f}-{min(hi, 1.0):.1f} | {v['dev'][0]} / {v['train'][0]} | {v['dev'][1]:.3f} | {v['train'][1]:.3f} | {v['dev'][1] - v['train'][1]:+.3f} | {100 * v['dev'][2]:.1f} | {100 * v['train'][2]:.1f} |")
open(os.path.join(C.OUT, "tables_overlap.md"), "w").write("\n".join(L) + "\n")
json.dump(res, open(os.path.join(C.OUT, "tables_overlap.json"), "w"), indent=1, default=float)
print("\n".join(L))
