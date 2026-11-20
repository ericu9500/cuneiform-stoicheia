"""Fit the damage statistics (cun.damage.fit) on one split of the corpus in cun.paths.PROC (CUN_PROC) and
save them next to the corpus (damage_stats.pkl).

  python scripts/damage_fit.py [--split train] [--out PATH]"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import damage as D  # noqa: E402
from cun.paths import PROC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    split = a.split if isinstance(a.split, str) else (a.split[0] if len(a.split) == 1 else a.split)
    with gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb") as fh:
        corpus = pickle.load(fh)
    t0 = time.time()
    st = D.fit(corpus, split)
    path = D.save_stats(st, a.out)
    print(f"corpus {st['version']} split {st['split']}: {st['n_texts']} texts, {st['n_lines']} lines, {st['n_signs']} signs; "
          f"strata {st['strata'][1:]}; fitted in {time.time() - t0:.1f} s -> {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
