"""What-if decoders on DEV (analysis only): how much of the measured headroom a different decoder realises
with the SAME checkpoint, so that decoder effects are not mistaken for training needs.

  python scripts/diag_errors_gpu2.py

D1  known width: left-to-right greedy; left-to-right beam (8) under the chain-rule score.
D2  known width: iterative most-confident-first with the predicted reading / joiner / determinative of a filled
    slot fed back (the evaluated decoder feeds back the sign only).
D3  unknown width: the two-stage decoder with a penalty per slot in the width choice
    (k = argmax score(k) - lam * k; lam = 0 is the evaluated iter-w).
Output: results/diag/errors/gpu_decoders.pkl.gz
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
from cun.consts import BLANK, FIRST_REAL_SIGN, READ_NONE  # noqa: E402
from cun.evaluate import Decoder, forward  # noqa: E402
from cun.model import CunConfig, CunEncoder  # noqa: E402

FB = 192
BEAM = 8
LAMS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.5)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ck = torch.load(os.path.join(ROOT, "ckpt/base-long/best.pt"), map_location="cuda", weights_only=False)
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    ev = evalset.load(ROOT, "dev", version=evalset.corpus_version(ROOT))
    assert ev["split"] == "dev" and ck.get("corpus_version") == ev["version"]
    inst = ev["instances"]
    dec = Decoder(model, ev, "cuda")
    allow_k, allow_u = dec.allow_known, dec.allow_unknown
    rows_k = [evalset.gap_input(ev, x, "known") for x in inst]
    rows_u = [evalset.gap_input(ev, x, "unknown") for x in inst]
    golds = [evalset.gold(ev, x)[0] for x in inst]
    out = {"version": ev["version"]}

    def fwd(rows):
        res = []
        for s in range(0, len(rows), FB):
            o = forward(model, rows[s:s + FB], "cuda")
            res.append(o)
        return res

    def ham(preds):
        return float(np.mean(np.concatenate([np.asarray(p) != g for p, g in zip(preds, golds)])))

    # ---- D1 beam (beam 1 = greedy)
    def l2r(beam):
        preds, scores = [None] * len(inst), np.zeros(len(inst))
        order = np.argsort([x["L"] for x in inst], kind="stable")
        G = max(1, FB // beam)
        for s in range(0, len(order), G):
            idx = order[s:s + G]
            beams = {i: [(0.0, [])] for i in idx}
            Lmax = max(inst[i]["L"] for i in idx)
            for k in range(Lmax):
                rows, owner = [], []
                for i in idx:
                    if inst[i]["L"] <= k:
                        continue
                    r = rows_k[i]
                    g = r["gap"]
                    for b, (sc, seq) in enumerate(beams[i]):
                        sg, rd = r["sign"].copy(), r["read"].copy()
                        sg[g:g + k] = seq
                        rd[g:g + k] = READ_NONE
                        rows.append({"sign": sg, "read": rd, "join": r["join"], "det": r["det"], "lang": r["lang"]})
                        owner.append((i, b))
                if not rows:
                    break
                o = forward(model, rows, "cuda")
                cand = {}
                for j, (i, b) in enumerate(owner):
                    g = rows_k[i]["gap"]
                    lp = torch.log_softmax(o["sign"][j, g + k].masked_fill(~allow_k, float("-inf")), -1)
                    tk = lp.topk(beam)
                    sc, seq = beams[i][b]
                    for v, t in zip(tk.values.tolist(), tk.indices.tolist()):
                        cand.setdefault(i, []).append((sc + v, seq + [t]))
                for i, c in cand.items():
                    c.sort(key=lambda z: -z[0])
                    beams[i] = c[:beam]
            for i in idx:
                preds[i] = np.array(beams[i][0][1], np.int64)
                scores[i] = beams[i][0][0]
        return preds, scores

    for beam in (1, BEAM):
        t0 = time.time()
        p, sc = l2r(beam)
        out[f"known/l2r-beam{beam}"] = {"sign": p, "score": sc}
        log("l2r beam", beam, "hamming err", ham(p), round(time.time() - t0), "s")

    # ---- D2 iterative with feedback of the other planes
    def iter_fb(rows):
        st = [{k: r[k].copy() for k in ("sign", "read", "join", "det")} | {"lang": r["lang"], "gap": r["gap"], "M": r["M"],
              "masked": np.ones(r["M"], bool)} for r in rows]
        while True:
            act = [i for i, s in enumerate(st) if s["masked"].any()]
            if not act:
                break
            o = forward(model, [st[i] for i in act], "cuda")
            for j, i in enumerate(act):
                s = st[i]
                g, M = s["gap"], s["M"]
                p = torch.softmax(o["sign"][j, g:g + M].masked_fill(~allow_k, float("-inf")), -1)
                conf, arg = p.max(-1)
                conf = conf.cpu().numpy()
                conf[~s["masked"]] = -1
                k = int(conf.argmax())
                choice = int(arg[k])
                s["sign"][g + k] = choice
                sg = torch.tensor([choice], device="cuda")
                s["read"][g + k] = int(dec.read_choice(o["read"][j, g + k:g + k + 1], sg)[0])
                s["join"][g + k] = int(o["join"][j, g + k].argmax())
                s["det"][g + k] = int(o["det"][j, g + k].argmax())
                s["masked"][k] = False
        return [{k: s[k][s["gap"]:s["gap"] + s["M"]].copy() for k in ("sign", "read", "join", "det")} for s in st]

    t0 = time.time()
    order = np.argsort([r["M"] for r in rows_k], kind="stable")
    pred = [None] * len(inst)
    for s in range(0, len(order), 96):
        idx = order[s:s + 96]
        for i, p in zip(idx, iter_fb([rows_k[i] for i in idx])):
            pred[i] = p
    out["known/iter-fb"] = pred
    log("iter-fb hamming err", ham([p["sign"] for p in pred]), round(time.time() - t0), "s")

    # ---- D3 width penalty
    score = [None] * len(inst)
    order = np.argsort([len(r["sign"]) for r in rows_u], kind="stable")
    for s in range(0, len(order), FB):
        idx = order[s:s + FB]
        o = forward(model, [rows_u[i] for i in idx], "cuda")
        for j, i in enumerate(idx):
            g, M = rows_u[i]["gap"], rows_u[i]["M"]
            p = torch.softmax(o["sign"][j, g:g + M].masked_fill(~allow_u, float("-inf")), -1)[:, BLANK]
            p = p.clamp(1e-6, 1 - 1e-6).double()
            real, blank = torch.log1p(-p), torch.log(p)
            score[i] = np.array([float(real[:k].sum() + blank[k:].sum()) for k in range(1, M + 1)])
    out["width_score"] = score
    ks = {lam: [1 + int(np.argmax(sc - lam * np.arange(1, len(sc) + 1))) for sc in score] for lam in LAMS}
    out["width_k"] = ks
    need = sorted({(i, k) for lam in LAMS for i, k in enumerate(ks[lam])})
    log("width: decodes needed", len(need))
    rows2 = []
    for i, k in need:
        r = rows_u[i]
        g, M = r["gap"], r["M"]
        keep = np.r_[np.arange(0, g + k), np.arange(g + M, len(r["sign"]))]
        rows2.append({**{key: r[key][keep] for key in ("sign", "read", "join", "det")}, "lang": r["lang"], "gap": g, "M": k})
    order = np.argsort([r["M"] for r in rows2], kind="stable")
    res = {}
    t0 = time.time()
    for s in range(0, len(order), 96):
        idx = order[s:s + 96]
        for j, p in zip(idx, dec.iterative([rows2[j] for j in idx], "known")):
            res[need[j]] = p
    out["width_pred"] = res
    log("width decodes done", round(time.time() - t0), "s")
    with gzip.open(os.path.join(ROOT, "results/diag/errors/gpu_decoders.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
