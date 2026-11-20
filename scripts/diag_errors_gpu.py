"""Diagnostics that need model probabilities (DEV split only; analysis, no training).

  python scripts/diag_errors_gpu.py

A  per-slot probabilities of the one-pass (indep) prediction, known width: log p(gold), rank of gold, top-10,
   entropy; unknown width: p(BLANK) per slot.
B  search error or model error: chain-rule log-probability (left to right) and pseudo-log-likelihood (each
   slot masked, the other gap slots given) of the gold gap and of the iterative decoder's gap.
C  context ablations, one-pass argmax, known width: context readings hidden; bare-sign context; language
   unknown; left-only / right-only context; context windows of +-5, 10, 20, 50 signs.
Output: results/diag/errors/gpu_probs.pkl.gz
"""
from __future__ import annotations

import gzip
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import evalset  # noqa: E402
from cun.consts import BLANK, FIRST_REAL_SIGN, MASK, READ_MASK, READ_NONE  # noqa: E402
from cun.evaluate import forward  # noqa: E402
from cun.model import CunConfig, CunEncoder  # noqa: E402
from cun.noise import D_UNK, J_UNK  # noqa: E402

SPLIT = "dev"
B = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def batches(n_rows, length_of):
    order = np.argsort([length_of(i) for i in range(n_rows)], kind="stable")
    for s in range(0, len(order), B):
        yield order[s:s + B]


def main():
    ck = torch.load(os.path.join(ROOT, "ckpt/base-long/best.pt"), map_location="cuda", weights_only=False)
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    ev = evalset.load(ROOT, SPLIT, version=evalset.corpus_version(ROOT))
    assert ev["split"] == "dev" and ck.get("corpus_version") == ev["version"]
    inst = ev["instances"]
    pred = pickle.load(gzip.open(os.path.join(ROOT, "results/gaps/base-long__dev.pkl.gz"), "rb"))
    V = model.cfg.n_sign
    n_lang = model.cfg.n_lang
    allow_k = torch.zeros(V, dtype=torch.bool, device="cuda")
    allow_k[FIRST_REAL_SIGN:] = True
    allow_u = allow_k.clone()
    allow_u[BLANK] = True
    out = {"version": ev["version"], "step": ck.get("step")}

    def lsm(logits, allow):
        return torch.log_softmax(logits.masked_fill(~allow, float("-inf")), -1)

    # ---- A
    rows_k = [evalset.gap_input(ev, x, "known") for x in inst]
    rows_u = [evalset.gap_input(ev, x, "unknown") for x in inst]
    golds = [evalset.gold(ev, x)[0] for x in inst]
    A = [None] * len(inst)
    for idx in batches(len(inst), lambda i: len(rows_k[i]["sign"])):
        o = forward(model, [rows_k[i] for i in idx], "cuda")
        for j, i in enumerate(idx):
            g, M = rows_k[i]["gap"], rows_k[i]["M"]
            lp = lsm(o["sign"][j, g:g + M], allow_k)
            gd = torch.from_numpy(golds[i]).cuda()
            lpg = lp.gather(1, gd[:, None])[:, 0]
            rank = (lp > lpg[:, None]).sum(-1)
            tk = lp.topk(10, -1)
            ent = -(lp.exp() * lp.clamp(min=-50)).sum(-1)
            A[i] = {"lp_gold": lpg.cpu().numpy().astype(np.float32), "rank": rank.cpu().numpy(),
                    "top_ids": tk.indices.cpu().numpy().astype(np.int32),
                    "top_lp": tk.values.cpu().numpy().astype(np.float32), "ent": ent.cpu().numpy().astype(np.float32)}
    for idx in batches(len(inst), lambda i: len(rows_u[i]["sign"])):
        o = forward(model, [rows_u[i] for i in idx], "cuda")
        for j, i in enumerate(idx):
            g, M = rows_u[i]["gap"], rows_u[i]["M"]
            lp = lsm(o["sign"][j, g:g + M], allow_u)
            A[i]["pB"] = lp[:, BLANK].exp().cpu().numpy().astype(np.float32)
            A[i]["arg_u"] = lp.argmax(-1).cpu().numpy().astype(np.int32)
    out["A"] = A
    log("A done")

    # ---- B
    tasks = []                                      # (inst, seq kind, mode, k)
    seqs = {}
    for i, x in enumerate(inst):
        ps = np.asarray(pred["known/iter"][i]["sign"], np.int64)
        seqs[(i, "gold")] = golds[i]
        kinds = ["gold"]
        if not np.array_equal(ps, golds[i]):
            seqs[(i, "pred")] = ps
            kinds.append("pred")
        for kind in kinds:
            for k in range(x["L"]):
                tasks.append((i, kind, "l2r", k))
                if x["L"] > 1:
                    tasks.append((i, kind, "pll", k))

    def make(task):
        i, kind, mode, k = task
        r = rows_k[i]
        g, M = r["gap"], r["M"]
        y = seqs[(i, kind)]
        s, rd = r["sign"].copy(), r["read"].copy()
        if mode == "l2r":
            s[g:g + k] = y[:k]
            rd[g:g + k] = READ_NONE
        else:
            s[g:g + M] = y
            rd[g:g + M] = READ_NONE
            s[g + k] = MASK
            rd[g + k] = READ_MASK
        return {"sign": s, "read": rd, "join": r["join"], "det": r["det"], "lang": r["lang"]}

    res = np.zeros((len(tasks), 3), np.float32)     # lp(y_k), rank(y_k), argmax
    for idx in batches(len(tasks), lambda t: len(rows_k[tasks[t][0]]["sign"])):
        o = forward(model, [make(tasks[t]) for t in idx], "cuda")
        for j, t in enumerate(idx):
            i, kind, mode, k = tasks[t]
            g = rows_k[i]["gap"]
            lp = lsm(o["sign"][j, g + k], allow_k)
            y = int(seqs[(i, kind)][k])
            res[t] = (float(lp[y]), float((lp > lp[y]).sum()), float(lp.argmax()))
    out["B_tasks"] = tasks
    out["B_res"] = res
    log("B done", len(tasks))

    # ---- C
    def variant(r, name):
        g, M = r["gap"], r["M"]
        n = len(r["sign"])
        d = {k: r[k].copy() for k in ("sign", "read", "join", "det")}
        d["lang"] = r["lang"]
        gap = np.zeros(n, bool)
        gap[g:g + M] = True
        lo, hi = 0, n
        if name == "base":
            pass
        elif name == "noread":
            d["read"][~gap] = READ_NONE
        elif name == "bare":
            d["read"][~gap] = READ_NONE
            d["join"][~gap] = J_UNK
            d["det"][~gap] = D_UNK
        elif name == "nolang":
            d["lang"] = n_lang
        elif name == "left":
            hi = g + M
        elif name == "right":
            lo = g
        elif name.startswith("win"):
            w = int(name[3:])
            lo, hi = max(0, g - w), min(n, g + M + w)
        for k in ("sign", "read", "join", "det"):
            d[k] = d[k][lo:hi]
        d["gap"], d["M"] = g - lo, M
        return d

    C = {}
    for name in ("base", "noread", "bare", "nolang", "left", "right", "win5", "win10", "win20", "win50"):
        rows = [variant(r, name) for r in rows_k]
        ps = [None] * len(rows)
        for idx in batches(len(rows), lambda i: len(rows[i]["sign"])):
            o = forward(model, [rows[i] for i in idx], "cuda")
            for j, i in enumerate(idx):
                g, M = rows[i]["gap"], rows[i]["M"]
                ps[i] = lsm(o["sign"][j, g:g + M], allow_k).argmax(-1).cpu().numpy().astype(np.int32)
        C[name] = ps
        log("C", name, "hamming err", float(np.mean(np.concatenate([p != g_ for p, g_ in zip(ps, golds)]))))
    out["C"] = C
    os.makedirs(os.path.join(ROOT, "results/diag/errors"), exist_ok=True)
    with gzip.open(os.path.join(ROOT, "results/diag/errors/gpu_probs.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
