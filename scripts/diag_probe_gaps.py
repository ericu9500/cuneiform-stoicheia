"""Probes 2-5 (GPU): run best.pt on the dev gap tasks of diag_probe_common.gap_tasks and dump per-instance
predictions; the tables are made by scripts/diag_probe_report.py (torch-free).

  python scripts/diag_probe_gaps.py [--ckpt ckpt/base-long/best.pt]
"""
import argparse, gzip, os, pickle, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_probe_common as C  # noqa: E402
from cun import evalset  # noqa: E402
from cun.consts import FIRST_REAL_SIGN, READ_NONE  # noqa: E402
from cun.evaluate import collate  # noqa: E402
from cun.model import CunConfig, CunEncoder  # noqa: E402

BATCH = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


@torch.no_grad()
def gap_probs(model, rows, allow):
    """Per row: (M, V) probabilities over the allowed signs at the gap slots (float32, on GPU)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(collate(rows, "cuda"), heads=("sign",))
    lg = out["sign"].float()
    res = []
    for i, r in enumerate(rows):
        g, M = r["gap"], r["M"]
        res.append(torch.softmax(lg[i, g:g + M].masked_fill(~allow, float("-inf")), -1))
    return res


def batches(rows):
    order = np.argsort([len(r["sign"]) for r in rows], kind="stable")
    for s in range(0, len(order), BATCH):
        yield order[s:s + BATCH]


def indep(model, rows, gold, allow):
    n = len(rows)
    pred, conf, pgold, rank, ent = [None] * n, [None] * n, [None] * n, [None] * n, [None] * n
    for idx in batches(rows):
        ps = gap_probs(model, [rows[i] for i in idx], allow)
        for i, p in zip(idx, ps):
            g = torch.from_numpy(gold[i]).cuda()
            c, a = p.max(-1)
            pg = p.gather(1, g[:, None])[:, 0]
            pred[i] = a.cpu().numpy().astype(np.int32)
            conf[i] = c.cpu().numpy()
            pgold[i] = pg.cpu().numpy()
            rank[i] = (p > pg[:, None]).sum(-1).cpu().numpy().astype(np.int32)      # 0 = gold is top-1
            ent[i] = (-(p * torch.log(p.clamp_min(1e-12))).sum(-1)).cpu().numpy()
    return {"pred": pred, "conf": conf, "pgold": pgold, "rank": rank, "ent": ent}


def iterative(model, rows, allow):
    """cun.evaluate.Decoder.iterative for a known width, recording the fill order and the confidence at fill."""
    n = len(rows)
    pred, order, conf = [None] * n, [None] * n, [None] * n
    for idx in batches(rows):
        st = [C.copy_row(rows[i]) for i in idx]
        masked = [np.ones(s["M"], bool) for s in st]
        od = [np.full(s["M"], -1, np.int32) for s in st]
        cf = [np.zeros(s["M"], np.float32) for s in st]
        step = 0
        while True:
            act = [j for j in range(len(st)) if masked[j].any()]
            if not act:
                break
            ps = gap_probs(model, [st[j] for j in act], allow)
            for j, p in zip(act, ps):
                s = st[j]
                c, a = p.max(-1)
                c = c.cpu().numpy()
                c[~masked[j]] = -1
                k = int(c.argmax())
                s["sign"][s["gap"] + k] = int(a[k])
                s["read"][s["gap"] + k] = READ_NONE
                masked[j][k] = False
                od[j][k] = step
                cf[j][k] = c[k]
            step += 1
        for j, i in enumerate(idx):
            s = st[j]
            pred[i] = s["sign"][s["gap"]:s["gap"] + s["M"]].astype(np.int32)
            order[i], conf[i] = od[j], cf[j]
    return {"it_pred": pred, "it_order": order, "it_conf": conf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/base-long/best.pt")
    ap.add_argument("--out", default="gaps_best.pkl.gz")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ck = torch.load(os.path.join(C.ROOT, a.ckpt), map_location="cuda", weights_only=False)
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    ev = evalset.load(C.ROOT, "dev", version=evalset.corpus_version(C.ROOT))
    assert ck.get("corpus_version") == ev["version"]
    corpus = C.load_corpus()
    n_lang = len(corpus["langs"])
    assert n_lang == model.cfg.n_lang
    sets = C.build_instances(ev, corpus)
    del corpus
    allow = torch.zeros(model.cfg.n_sign, dtype=torch.bool)
    allow[FIRST_REAL_SIGN:] = True
    allow = allow.cuda()
    out = {"ckpt": a.ckpt, "step": ck.get("step"), "sets": sets,
           "meta": {k: [C.meta(ev, x) for x in v] for k, v in sets.items()},
           "gold": {k: [evalset.gold(ev, x)[0] for x in v] for k, v in sets.items()},
           "repeat": {k: [C.repeat_info(ev, x) for x in sets[k]] for k in ("multi1", "far1")}, "res": {}}
    os.makedirs(C.OUT, exist_ok=True)
    t0 = time.time()
    for name, key, pairs, info in C.gap_tasks(ev, sets, n_lang):
        if a.smoke:
            pairs = pairs[:200]
        idx = [i for i, _ in pairs]
        rows = [r for _, r in pairs]
        gold = [out["gold"][key][i] for i in idx]
        r = {"idx": np.array(idx), "info": info}
        if rows:
            r.update(indep(model, rows, gold, allow))
            if key in ("ev5", "ext5", "evL"):
                r.update(iterative(model, rows, allow))
        out["res"][(key, name)] = r
        acc = np.mean([float((p == g).mean()) for p, g in zip(r.get("pred", []), gold)]) if rows else float("nan")
        log(f"{key:7s} {name:24s} n {len(rows):6d} slot acc {acc:.4f}  ({time.time() - t0:.0f} s)")
    with gzip.open(os.path.join(C.OUT, a.out), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
