"""Generalisation gap (analysis only): the same gap protocol and the same fixed noisings on 2,000 random TRAIN
texts and on the DEV texts. Test and excluded texts are never touched.

  python scripts/diag_errors_gpu3.py
Output: results/diag/errors/gpu_traingap.pkl.gz
"""
from __future__ import annotations

import gzip
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from rapidfuzz.distance import Levenshtein

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.devset import dev_noisings  # noqa: E402
from cun.evaluate import Decoder  # noqa: E402
from cun.model import CunConfig, CunEncoder  # noqa: E402
from cun.noise import IGN  # noqa: E402
from cun.train import pack  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ck = torch.load(os.path.join(ROOT, "ckpt/base-long/best.pt"), map_location="cuda", weights_only=False)
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    corpus = pickle.load(gzip.open(os.path.join(ROOT, "data/proc/corpus.pkl.gz"), "rb"))
    assert ck.get("corpus_version") == corpus["version"]
    n_lang = len(corpus["langs"])
    rng = np.random.default_rng(0)
    train_ids = [i for i, t in enumerate(corpus["texts"]) if t["split"] == "train"]
    pick = set(int(i) for i in rng.choice(train_ids, 2000, replace=False))
    out = {"version": corpus["version"]}
    sets = {"train-sample": pick, "dev": evalset.split_texts(corpus, "dev")}
    for name, texts in sets.items():
        chunks = evalset.eval_chunks(corpus, texts)
        # (1) fixed noisings as in cun.train's dev evaluation
        tup = [(c[0], c[1], c[2], c[3], lg) for ti, lg, c in chunks]
        noised = dev_noisings(tup, n_lang)
        res = {}
        for pattern in (0, 4):
            ds = [d for p, _, d in noised if p == pattern]
            tot = {"sign": [0.0, 0, 0], "read": [0.0, 0, 0]}
            for s in range(0, len(ds), 64):
                b = pack(ds[s:s + 64], 32, "cuda")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    o = model(b)
                for h, tg in (("sign", "s_tg"), ("read", "r_tg")):
                    t = b[tg]
                    m = t != IGN
                    if m.any():
                        lg_ = o[h].float()[m]
                        tot[h][0] += float(F.cross_entropy(lg_, t[m], reduction="sum"))
                        tot[h][1] += int((lg_.argmax(-1) == t[m]).sum())
                        tot[h][2] += int(m.sum())
            res[pattern] = {h: {"nll": v[0] / max(1, v[2]), "acc": v[1] / max(1, v[2]), "n": v[2]} for h, v in tot.items()}
        log(name, "fixed noisings (0 = span t=0.15, 4 = transliteration)", res)
        # (2) gap protocol
        inst = evalset.make_instances(chunks, "dev")
        ev = {"chunks": chunks, "instances": inst, "sign_reads": corpus["sign_reads"]}
        dec = Decoder(model, ev, "cuda")
        rows = [evalset.gap_input(ev, x, "known") for x in inst]
        order = np.argsort([r["M"] for r in rows], kind="stable")
        per = {}
        for dn, fn in (("indep", dec.indep), ("iter", dec.iterative)):
            pred = [None] * len(rows)
            for s in range(0, len(order), 96):
                idx = order[s:s + 96]
                for i, p in zip(idx, fn([rows[i] for i in idx], "known")):
                    pred[i] = p
            ser = np.array([Levenshtein.distance([int(v) for v in p["sign"]], evalset.gold(ev, x)[0].tolist()) / x["L"] for p, x in zip(pred, inst)])
            ex = np.array([float(np.array_equal(p["sign"], evalset.gold(ev, x)[0])) for p, x in zip(pred, inst)])
            per[dn] = {"ser": ser, "exact": ex}
            log(name, dn, "SER", round(100 * ser.mean(), 2), "exact", round(100 * ex.mean(), 2))
        out[name] = {"fixed": res, "L": np.array([x["L"] for x in inst]), "lang": [corpus["langs"][chunks[x["chunk"]][1]] for x in inst],
                     "text": [chunks[x["chunk"]][0] for x in inst], **{f"{dn}_{k}": v for dn, d in per.items() for k, v in d.items()}}
    with gzip.open(os.path.join(ROOT, "results/diag/errors/gpu_traingap.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
