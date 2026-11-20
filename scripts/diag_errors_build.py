"""Build the cached tables for the error diagnosis (DEV split only; train texts are read for frequencies).

  python scripts/diag_errors_build.py     -> results/diag/errors/cache.pkl.gz
Test and excluded texts are skipped when reading the corpus.
"""
from __future__ import annotations

import collections
import gzip
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results/diag/errors")


def main():
    c = pickle.load(gzip.open(os.path.join(ROOT, "data/proc/corpus.pkl.gz"), "rb"))
    texts = c["texts"]
    V, NR = len(c["sign_vocab"]), len(c["read_vocab"])
    langs = c["langs"]
    li = {l: i for i, l in enumerate(langs)}
    split = np.array([{"train": 0, "dev": 1}.get(t["split"], 2) for t in texts])
    tl = np.array([li[t["lang"]] for t in texts])
    f_sign = np.zeros((len(langs), V), np.int64)            # train sign counts by language
    f_det = np.zeros((len(langs), V), np.int64)             # ... used as determinative
    f_clean = np.zeros(V, np.int64)
    pair = collections.Counter()                            # (lang, sign, read) train counts
    n_train = 0
    dev_len = collections.Counter()
    brk_tot = np.zeros(3, np.int64)
    for ti, a in c["lines"]:
        if split[ti] == 2:
            continue
        if split[ti] == 1:
            dev_len[ti] += a.shape[1]
            continue
        lg = tl[ti]
        np.add.at(f_sign[lg], a[0], 1)
        np.add.at(f_det[lg], a[0][a[3] > 0], 1)
        np.add.at(f_clean, a[0][a[4] == 0], 1)
        np.add.at(brk_tot, a[4], 1)
        n_train += a.shape[1]
        for k, n in zip(*np.unique(a[0].astype(np.int64) * NR + a[1], return_counts=True)):
            pair[(int(lg), int(k))] += int(n)
    sr = {}
    for (lg, k), n in pair.items():
        sr[(lg, k // NR, k % NR)] = n
    meta = {ti: {k: texts[ti].get(k) for k in ("id", "src", "lang", "period", "genre", "n")} | {"n_signs": dev_len[ti]}
            for ti in dev_len}
    out = {"f_sign": f_sign, "f_det": f_det, "f_clean": f_clean, "sign_read_lang": sr, "n_train": n_train,
           "dev_meta": meta, "langs": langs, "joins": c["joins"], "dets": c["dets"], "brks": c["brks"],
           "brk_tot": brk_tot, "sign_names": c["sign_names"], "read_vocab": c["read_vocab"],
           "sign_reads": c["sign_reads"], "version": c["version"],
           "train_texts_by": {k: collections.Counter((texts[i]["lang"], texts[i].get(k)) for i in np.flatnonzero(split == 0))
                              for k in ("period", "genre", "src")},
           "train_signs_by_text_meta": None}
    # train signs by (lang, period) and (lang, genre), (lang, src)
    tn = collections.Counter()
    for ti, a in c["lines"]:
        if split[ti] == 0:
            tn[ti] += a.shape[1]
    agg = {k: collections.Counter() for k in ("period", "genre", "src")}
    for ti, n in tn.items():
        for k in agg:
            agg[k][(texts[ti]["lang"], texts[ti].get(k))] += n
    out["train_signs_by_text_meta"] = agg
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, "cache.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    print("train signs", n_train, "dev texts", len(meta), "joins", c["joins"], "dets", c["dets"], "brks", c["brks"], brk_tot)


if __name__ == "__main__":
    main()
