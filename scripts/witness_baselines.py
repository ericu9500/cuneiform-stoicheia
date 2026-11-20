"""CPU baselines on the duplicate-witness set (witness-dev only; cun.witness), known width:
  4gram    Witten-Bell sign 4-gram trained on split == 'train', language-specific (sux / akk / pooled), beam 8,
           right context scored: exactly cun.eval3.ngram
  editor   the restoration of A's editor where the lacuna is fully restored (biased: editors often restore
           FROM the duplicate)
Output: RESULTS/witness/ngram__witness-dev.pkl.gz (predictions for every lacuna, by lacuna id) and
RESULTS/witness/baselines_dev.json. Needs the corpus: run on a compute node.

  CUN_PROC=data/proc_v3 python scripts/witness_baselines.py [--workers 8]
"""
import argparse
import collections
import gzip
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cun import eval3, witness  # noqa: E402
from cun.consts import FIRST_REAL_SIGN  # noqa: E402
from cun.devset import load_chunks  # noqa: E402
from cun.ngram import WB, fill_runs  # noqa: E402
from cun.paths import RESULTS  # noqa: E402

G = {}
LBINS = ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20))


def one(k):
    ev = G["ev"]
    x = ev["instances"][k]
    r = eval3.masked_input(ev, x)
    lang = ev["langs"][r["lang"]]
    filled, _ = fill_runs(G["lm"].get(lang, G["lm"]["all"]), G["cands"], r["sign"], beam=8)
    return filled[x["pos"]]


def table(rows):
    """rows: [(L, distance, exact, exact against any witness)] -> exact match %, SER % (Levenshtein / L,
    averaged over lacunae), exact match with the reading of any answering witness %, n."""
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "exact": 100 * float(np.mean([r[2] for r in rows])),
            "ser": 100 * float(np.mean([r[1] / r[0] for r in rows])),
            "exact_any": 100 * float(np.mean([r[3] for r in rows]))}


def summary(lac, pred):
    """pred: {lacuna id: predicted signs}; tables overall and by subgroup."""
    from rapidfuzz.distance import Levenshtein
    rows = {}
    for r in lac:
        if r["id"] in pred:
            p = [int(v) for v in pred[r["id"]]]
            d = Levenshtein.distance(p, r["answer"][0].tolist())
            rows[r["id"]] = (r["L"], d, float(d == 0), float(d == 0 or any(p == v.tolist() for _, v in r["alts"])))
    out = {"all": table(list(rows.values()))}
    for name, key in (("by_length", lambda r: next(f"{a}-{b}" if a != b else str(a) for a, b in LBINS if a <= r["L"] <= b)),
                      ("by_regime", lambda r: r.get("regime")), ("by_stratum", lambda r: str(r["stratum"])),
                      ("by_kind", lambda r: r["kind"]), ("by_line_pos", lambda r: r["line_pos"]),
                      ("by_lang", lambda r: r["lang"])):
        g = collections.defaultdict(list)
        for r in lac:
            if r["id"] in rows:
                g[key(r)].append(rows[r["id"]])
        out[name] = {k: table(v) for k, v in g.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    corpus = eval3.load_corpus()
    obj = witness.load("dev", corpus["version"])
    ev = witness.instances(obj, "known", strict=False, clean_only=False, width_conflicts=True)
    train = load_chunks(corpus, "train")
    V = len(corpus["sign_vocab"])
    lms = {}
    for gk in ("all", "sux", "akk"):
        sub = train if gk == "all" else [c for c in train if corpus["langs"][c[4]] == gk]
        lms[gk] = WB([c[0] for c in sub], 4, 4096, V)
    del corpus, train
    G.update(ev=ev, lm=lms, cands=np.arange(FIRST_REAL_SIGN, V, dtype=np.int64))
    print(time.strftime("%H:%M:%S"), "instances", len(ev["instances"]), flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        preds = pool.map(one, range(len(ev["instances"])), chunksize=16)
    ids = [x["lac"] for x in ev["instances"]]
    os.makedirs(os.path.join(RESULTS, "witness"), exist_ok=True)
    with gzip.open(os.path.join(RESULTS, "witness", "ngram__witness-dev.pkl.gz"), "wb") as fh:
        pickle.dump({"version": ev["version"], "split": "witness-dev", "lac": ids, "known/4gram": preds}, fh, protocol=5)
    d4 = dict(zip(ids, preds))
    de = {r["id"]: r["editor"] for r in obj["lacunae"] if r["kind"] == "restored"}
    res = {"version": ev["version"], "sets": {}}
    for name, sel in (("strict", dict()), ("strict, at most 30 per composite", dict(max_per_composite=30)),
                      ("strict, witnesses agree", dict(agree_only=True)),
                      ("all (flanks may be damaged)", dict(clean_only=False))):
        lac = witness.select(obj, **sel)
        rest = [r for r in lac if r["kind"] == "restored"]
        res["sets"][name] = {"4gram": summary(lac, d4), "editor (restored lacunae)": summary(rest, de),
                             "4gram (restored lacunae)": summary(rest, d4),
                             "4gram (not restored)": summary([r for r in lac if r["kind"] != "restored"], d4)}
    json.dump(res, open(os.path.join(RESULTS, "witness", "baselines_dev.json"), "w"), indent=1)
    for name, v in res["sets"].items():
        print(name, {k: t["all"] for k, t in v.items()})


if __name__ == "__main__":
    main()
