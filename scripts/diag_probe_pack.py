"""Probe 6 side check (torch-free): which chunks does cun.train.pack drop, and how full are the rows?
Re-implements only the row assignment of pack() (same greedy rule) on the stream of batch_stream."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_probe_common as C
from cun.devset import load_chunks
from cun.noise import NoiseConfig, noise_chunk

ROW, rows = 512, 64
corpus = C.load_corpus()
n_lang = len(corpus["langs"])
chunks = load_chunks(corpus, "train")
ln = np.array([len(c[0]) for c in chunks])
rng = np.random.default_rng(0)
kept, dropped, fills, Ts = [], [], [], []
for _ in range(300):
    noised, tot = [], 0
    while tot < rows * ROW * 0.95:
        c = chunks[rng.integers(len(chunks))]
        d = noise_chunk(rng, *c, n_lang=n_lang, cfg=NoiseConfig())
        noised.append(len(d["sign"])); tot += noised[-1]
    used = [0] * rows
    for L in noised:
        k = int(np.argmin(used))
        if used[k] + L > ROW:
            dropped.append(L); continue
        used[k] += L; kept.append(L)
    Ts.append(max(used)); fills.append(sum(used) / (rows * max(used)))
kept, dropped = np.array(kept), np.array(dropped)
res = {"batches": 300, "chunks_drawn": int(len(kept) + len(dropped)), "chunks_dropped": int(len(dropped)),
       "share_chunks_dropped": len(dropped) / (len(kept) + len(dropped)),
       "share_signs_dropped": dropped.sum() / (kept.sum() + dropped.sum()),
       "mean_fill": float(np.mean(fills)), "mean_T": float(np.mean(Ts)),
       "signs_per_step": float(kept.sum() / 300)}
for lo, hi in ((2, 64), (65, 128), (129, 200), (201, 255), (256, 300)):
    k = ((kept >= lo) & (kept <= hi)).sum(); d = ((dropped >= lo) & (dropped <= hi)).sum()
    res[f"drop_rate_len_{lo}-{hi}"] = d / max(1, k + d)
    res[f"n_len_{lo}-{hi}"] = int(k + d)
res["train_chunks"] = len(chunks)
res["train_signs"] = int(ln.sum())
res["share_signs_in_chunks_ge129"] = float(ln[ln >= 129].sum() / ln.sum())
res["share_signs_in_full_256_chunks"] = float(ln[ln == 256].sum() / ln.sum())
res["share_chunks_len_le32"] = float((ln <= 32).mean())
res["share_signs_in_chunks_le32"] = float(ln[ln <= 32].sum() / ln.sum())
res["chunk_len_percentiles_10_25_50_75_90"] = np.percentile(ln, [10, 25, 50, 75, 90]).tolist()
res["epochs_76k_steps"] = 76000 * res["signs_per_step"] / ln.sum()
# texts longer than one chunk: share of signs after the first 256
print(json.dumps(res, indent=1, default=float))
json.dump(res, open(os.path.join(C.OUT, "pack_stats.json"), "w"), indent=1, default=float)
