"""Noise audit, step 0: compact cache of the TRAIN split of the corpus (CUN_PROC, default data/proc_v3; read-only).

Writes results/noise_audit/cache/train.npz (planes 5 x N int32 concatenated over train lines, text index per
sign, line start offsets) and meta.json (vocabularies; metadata of every text: id, src, lang, period, genre,
split -- metadata only, no line of a dev / test text is copied).
  python scripts/noise_audit_cache.py
"""
import gzip
import json
import os
import pickle

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from noise_audit_lib import OUT as _OUT, PROC, VERSION  # noqa: E402
OUT = os.path.join(_OUT, "cache")
os.makedirs(OUT, exist_ok=True)
c = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
assert c["version"] == VERSION, c["version"]
texts = c["texts"]
train = np.array([t["split"] == "train" for t in texts])
arrs, tix, starts = [], [], [0]
for ti, a in c["lines"]:
    if train[ti]:
        arrs.append(a)
        tix.append(np.full(a.shape[1], ti, np.int32))
        starts.append(starts[-1] + a.shape[1])
A = np.concatenate(arrs, 1)
np.savez(os.path.join(OUT, "train.npz"), planes=A, text=np.concatenate(tix), line_start=np.array(starts, np.int64))
json.dump({"version": c["version"], "sign_vocab": c["sign_vocab"], "sign_names": c["sign_names"],
           "read_vocab": c["read_vocab"], "joins": c["joins"], "dets": c["dets"], "brks": c["brks"], "langs": c["langs"],
           "sign_reads": {str(k): v for k, v in c["sign_reads"].items()},
           "texts": [{k: t.get(k) for k in ("id", "src", "lang", "period", "genre", "cdli", "n", "split", "stratum", "period_bin", "dialect", "genre_class")} for t in texts]},
          open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False)
print("train texts", int(train.sum()), "train signs", A.shape[1], "lines", len(arrs))
