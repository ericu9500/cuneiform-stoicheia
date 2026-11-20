"""GPU side of the decoding experiments (PLAN_v3 Stage 1): raw candidates and scores on the DEV gaps.
Scoring, tuning on half A and the report: scripts/decode_score.py.

  python scripts/decode_run.py --stage known     beams, candidate planes, PLL, remasking, calibration logits
  python scripts/decode_run.py --stage widths    every candidate width decoded and scored (unknown width)
  python scripts/decode_run.py --stage final     recommended configurations through the drop-in decoders

Output: results/decode/raw_<stage>[tag].pkl.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import decode as D  # noqa: E402
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN, MASK  # noqa: E402

OUT = os.path.join(ROOT, "results", "decode")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(obj, name):
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(os.path.join(OUT, name), "wb", compresslevel=3) as fh:
        pickle.dump(obj, fh, protocol=5)
    log("saved", name)


class Timer:
    def __init__(self, eng):
        self.eng, self.cost = eng, {}

    def start(self):
        self.t, self.r = time.time(), self.eng.rows

    def stop(self, name):
        self.cost[name] = {"rows": self.eng.rows - self.r, "sec": round(time.time() - self.t, 1)}
        log(name, self.cost[name])


def pack(H, top=None):
    return [(D.key(g), float(s), lp.astype(np.float16)) for g, s, lp in H[:top]]


def stage_known(eng, ev, ids, a):
    inst = [ev["instances"][i] for i in ids]
    bases = [evalset.gap_input(ev, x, "known") for x in inst]
    N = len(inst)
    tm = Timer(eng)
    out = {"beams": {}, "remask": {}}
    for name, B, ns in (("b1", 1, 1), ("b5", 5, 1), ("b20", 20, 1), ("b5s2", 5, 2), ("b20s2", 20, 2), ("b5s3", 5, 3)):
        tm.start()
        out["beams"][name] = [pack(H) for H in D.beam_search(eng, bases, B=B, n_slots=ns)]
        tm.stop(name)
    b1 = [D.unkey(H[0][0]) for H in out["beams"]["b1"]]
    b1lp = [H[0][2].astype(np.float32) for H in out["beams"]["b1"]]
    for name, kw in (("fill_r1_f25", dict(rounds=1, frac=0.25)), ("fill_r2_f25", dict(rounds=2, frac=0.25)),
                     ("fill_r2_f50", dict(rounds=2, frac=0.5)),
                     ("pll_r1_f25", dict(rounds=1, frac=0.25, conf="pll")),
                     ("pll_r2_f25", dict(rounds=2, frac=0.25, conf="pll")),
                     ("pll_r2_f25_acc", dict(rounds=2, frac=0.25, conf="pll", accept="pll")),
                     ("pll_r3_f50_acc", dict(rounds=3, frac=0.5, conf="pll", accept="pll")),
                     ("pll_r4_one_acc", dict(rounds=4, frac=0.0, conf="pll", accept="pll"))):
        tm.start()
        f, _, nch = D.remask(eng, bases, b1, slot_lp=b1lp, **kw)
        out["remask"][name] = [D.key(x) for x in f]
        tm.stop("remask_" + name)
        log("  changed", nch)
    # retrieved fills (scripts/decode_retrieve.py)
    retr = None
    rp = os.path.join(OUT, "retrieval__dev.pkl.gz")
    if os.path.exists(rp):
        retr = pickle.load(gzip.open(rp, "rb"))
        assert retr["version"] == ev["version"]
        retr = [[D.key(f[0]) for f in retr["inst"][i]["known"]["fills"][:3]] for i in ids]
    # candidate sets
    allc = [dict() for _ in range(N)]       # every candidate: planes
    plc = [dict() for _ in range(N)]        # candidates that get a PLL
    for name, top in (("b1", 1), ("b5", 5), ("b20", 20), ("b5s2", 5), ("b20s2", 20), ("b5s3", 5)):
        for i, H in enumerate(out["beams"][name]):
            for j, h in enumerate(H):
                allc[i][h[0]] = 1
                if j < min(top, a.n_pll):
                    plc[i][h[0]] = 1
    for v in out["remask"].values():
        for i, k in enumerate(v):
            allc[i][k] = 1
            plc[i][k] = 1
    forced_keys = [dict() for _ in range(N)]
    for name in ("b5s2", "b20s2", "b5s3"):
        for i, H in enumerate(out["beams"][name]):
            for h in H[:5]:
                forced_keys[i][h[0]] = 1
    if retr:
        for i, ks in enumerate(retr):
            for k in ks:
                allc[i][k] = 1
                plc[i][k] = 1
                forced_keys[i][k] = 1
    tm.start()
    pairs = [(i, k) for i in range(N) for k in allc[i]]
    fin = D.finalize(eng, [bases[i] for i, _ in pairs], [D.unkey(k) for _, k in pairs])
    out["planes"] = [dict() for _ in range(N)]
    for (i, k), d in zip(pairs, fin):
        out["planes"][i][k] = tuple(d[p].astype(np.int16) for p in ("read", "join", "det"))
    tm.stop("planes")
    tm.start()
    pairs = [(i, k) for i in range(N) for k in forced_keys[i]]
    sc, _ = D.score_fill(eng, [bases[i] for i, _ in pairs], [D.unkey(k) for _, k in pairs])
    out["forced"] = [dict() for _ in range(N)]
    for (i, k), s in zip(pairs, sc):
        out["forced"][i][k] = s
    tm.stop("forced")
    pairs = [(i, k) for i in range(N) for k in plc[i]]
    for name, stride in (("pll", None), ("pll2", 2)):
        tm.start()
        pl = D.pll(eng, [bases[i] for i, _ in pairs], [D.unkey(k) for _, k in pairs], stride=stride)
        out[name] = [dict() for _ in range(N)]
        for (i, k), p in zip(pairs, pl):
            out[name][i][k] = p
        tm.stop(name)
    # calibration: first pass, all slots masked
    tm.start()
    gold = [evalset.gold(ev, x)[0] for x in inst]
    items = [(b, np.full(b["M"], MASK, np.int64)) for b in bases]
    r = eng.run(items, raw=128, targets=gold)
    one = [i for i, x in enumerate(inst) if x["L"] == 1]
    V = eng.model.cfg.n_sign - FIRST_REAL_SIGN
    r1 = eng.run([items[i] for i in one], raw=V, targets=[gold[i] for i in one])
    out["calib"] = {"rawv": r["rawv"].astype(np.float16), "rawi": r["rawi"].astype(np.int16), "rawt": r["rawt"],
                    "one": one, "one_v": r1["rawv"][:, 0], "one_i": r1["rawi"][:, 0].astype(np.int16)}
    tm.stop("calib")
    out["cost"] = tm.cost
    return out


def stage_widths(eng, ev, ids, a):
    inst = [ev["instances"][i] for i in ids]
    conds = D.width_conditions(ev["instances"])
    conds = [conds[i] for i in ids]
    tm = Timer(eng)
    out = {"conds": conds, "blank": {}}
    tm.start()
    for c in ("unknown", "r25", "r50"):
        bases = [D.set_width(evalset.gap_input(ev, x, "known"), cd[c][0]) for x, cd in zip(inst, conds)]
        out["blank"][c] = D.blank_width_scores(eng, bases)
    tm.stop("blank")
    known = [evalset.gap_input(ev, x, "known") for x in inst]
    pairs = [(i, k) for i, cd in enumerate(conds) for k in sorted(set(cd["unknown"][1]) | set(cd["r25"][1]) | set(cd["r50"][1]))]
    bases = [D.set_width(known[i], k) for i, k in pairs]
    log("pairs", len(pairs), "rows", sum(k for _, k in pairs))
    tm.start()
    hyps = D.beam_search(eng, bases, B=1)
    tm.stop("fill")
    fills = [h[0][0] for h in hyps]
    tm.start()
    pl = D.pll(eng, bases, fills)
    tm.stop("pll")
    tm.start()
    pl2 = D.pll(eng, bases, fills, stride=2)
    tm.stop("pll2")
    tm.start()
    fin = D.finalize(eng, bases, fills)
    tm.stop("planes")
    out["w"] = [dict() for _ in inst]
    for (i, k), h, p, p2, d in zip(pairs, hyps, pl, pl2, fin):
        out["w"][i][k] = {"key": D.key(h[0][0]), "chain": float(h[0][1]), "lp": h[0][2].astype(np.float16), "pll": p,
                          "pll2": p2, "planes": tuple(d[q].astype(np.int16) for q in ("read", "join", "det"))}
    out["cost"] = tm.cost
    return out


def stage_final(eng, ev, ids, a):
    """Recommended configurations through the drop-in decoders. results/decode/final_plan.pkl.gz
    (scripts/decode_score.py --write-plan): {'known': {name: decode_known kwargs}, 'unknown': {name:
    decode_unknown kwargs + cond}, 'bonus': {'known' | 'all': per instance {key(fill): bonus}},
    'width_bonus': per instance {k: bonus}}."""
    plan = pickle.load(gzip.open(os.path.join(OUT, "final_plan.pkl.gz"), "rb"))
    inst = [ev["instances"][i] for i in ids]
    conds = D.width_conditions(ev["instances"])
    conds = [conds[i] for i in ids]
    known = [evalset.gap_input(ev, x, "known") for x in inst]
    tm = Timer(eng)
    out = {"pred": {}}

    def run(fn, rows, **kw):
        order = np.argsort([r["M"] for r in rows], kind="stable")
        res = [None] * len(rows)
        for c0 in range(0, len(rows), 512):
            idx = order[c0:c0 + 512]
            sub = {k: [v[i] for i in idx] for k, v in kw.items() if isinstance(v, list)}
            rest = {k: v for k, v in kw.items() if not isinstance(v, list)}
            for i, d in zip(idx, fn(eng, [rows[i] for i in idx], **sub, **rest)):
                res[i] = d
        return res

    for name, cfg in plan["known"].items():
        kw = dict(cfg)
        if kw.get("bonus"):
            kw["bonus"] = [plan["bonus"][kw["bonus"]][i] for i in ids]
        tm.start()
        out["pred"]["known/" + name] = run(D.decode_known, known, **kw)
        tm.stop("known/" + name)
    for name, cfg in plan["unknown"].items():
        kw = dict(cfg)
        c = kw.pop("cond")
        rows = [D.set_width(b, cd[c][0]) for b, cd in zip(known, conds)]
        kw["widths"] = [cd[c][1] for cd in conds]
        if kw.get("bonus"):
            kw["bonus"] = [plan["bonus"][kw["bonus"]][i] for i in ids]
        if kw.pop("width_bonus", False):
            kw["width_bonus"] = [plan["width_bonus"][i] for i in ids]
        tm.start()
        out["pred"][f"{c}/{name}"] = run(D.decode_unknown, rows, **kw)
        tm.stop(f"{c}/{name}")
    out["cost"] = tm.cost
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("known", "widths", "final"))
    ap.add_argument("--ckpt", default="ckpt/base-long/best.pt")
    ap.add_argument("--every", type=int, default=1, help="smoke test: every n-th instance")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--n-pll", type=int, default=10, help="candidates per beam that get a pseudo-likelihood")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    eng, ev = D.load_engine(ROOT, a.ckpt, "dev")
    assert ev["split"] == "dev"
    si, sn = (int(v) for v in a.shard.split("/"))
    ids = [i for i in range(0, len(ev["instances"]), a.every) if i % sn == si]
    log("stage", a.stage, "instances", len(ids))
    out = {"known": stage_known, "widths": stage_widths, "final": stage_final}[a.stage](eng, ev, ids, a)
    out.update(version=ev["version"], split="dev", ids=ids, ckpt=a.ckpt)
    save(out, f"raw_{a.stage}{a.tag}.pkl.gz")
    log(json.dumps(out["cost"]))


if __name__ == "__main__":
    main()
