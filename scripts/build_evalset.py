"""Build the gap-restoration evaluation sets (cun.evalset) for dev and test."""
import collections
import gzip
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.paths import PROC  # noqa: E402

corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
for split in sys.argv[1:] or ("dev", "test"):
    ev = evalset.build(corpus, ROOT, split)
    inst = ev["instances"]
    L = collections.Counter(i["L"] for i in inst)
    lg = collections.Counter(ev["langs"][ev["chunks"][i["chunk"]][1]] for i in inst)
    print(split, "chunks", len(ev["chunks"]), "instances", len(inst), "per L min", min(L.values()),
          "texts", len({ev["chunks"][i["chunk"]][0] for i in inst}), dict(lg))
