"""Run a pretrained sign-level model on the evaluation sets (predictions only; scoring: scripts/score_gaps.py).

  python -m cun.evaluate --ckpt ckpt/base/best.pt --name base --split dev

Gap restoration (cun.evalset), conditions known / unknown width, two decoders:
  indep   one forward pass; every slot takes its argmax sign independently
  iter    fill the single most confident slot (max sign probability), feed the sign back (its reading
          hidden, as in pretraining), repeat; in the unknown-width condition a slot whose choice is
          BLANK is deleted. A final pass reads off readings, joiners and determinatives.
  iter-w  (--width-first; unknown width only) decide the width first, then fill iteratively.
Signs are chosen among real signs (+ BLANK when the width is unknown). Readings are constrained to the
readings attested for the chosen sign (>= 2 occurrences in the pretraining corpus).

Transliteration ('bare signs'): every chunk of the split with all readings, joiners and determinatives
hidden; the reading of every identified sign that has one is predicted (constrained and unconstrained).

Output: results/gaps/<name>__<split>.pkl.gz
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import time

import numpy as np
import torch

from . import evalset
from .consts import BLANK, FIRST_REAL_SIGN, MASK, READ_NONE
from .model import CunConfig, CunEncoder
from .noise import D_UNK, J_UNK
from .paths import RESULTS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLANES = ("sign", "read", "join", "det", "status", "slot")
BATCH = 96


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def collate(rows, device):
    """rows: dicts with sign/read/join/det arrays and lang; optional 'status' (default 0 = clean, masked
    signs 3) and 'slot' (1 on the slots of an unknown-width gap)."""
    T = max(len(r["sign"]) for r in rows)
    b = {k: np.full((len(rows), T), f, np.int64) for k, f in
         (("sign", 0), ("read", 0), ("join", J_UNK), ("det", D_UNK), ("lang", 0), ("status", 3), ("slot", 0))}
    seg = np.zeros((len(rows), T), np.int64)
    for i, r in enumerate(rows):
        n = len(r["sign"])
        for k in ("sign", "read", "join", "det"):
            b[k][i, :n] = r[k]
        st = r.get("status")
        b["status"][i, :n] = st if st is not None else np.where(np.asarray(r["sign"]) == MASK, 3, 0)
        if r.get("slot") is not None:
            b["slot"][i, :n] = r["slot"]
        b["lang"][i, :] = r["lang"]
        seg[i, n:] = 10_000 + np.arange(T - n)
    b["seg"] = seg
    return {k: torch.from_numpy(v).to(device) for k, v in b.items()}


@torch.no_grad()
def forward(model, rows, device):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(collate(rows, device))
    return {k: out[k].float() for k in ("sign", "read", "join", "det", "width") if k in out}


class Decoder:
    def __init__(self, model, ev, device):
        self.model, self.ev, self.device = model, ev, device
        V = model.cfg.n_sign
        NR = model.cfg.n_read
        allow = torch.zeros(V, dtype=torch.bool)
        allow[FIRST_REAL_SIGN:] = True
        self.allow_known = allow.to(device)
        a2 = allow.clone()
        a2[BLANK] = True
        self.allow_unknown = a2.to(device)
        R = torch.zeros(V, NR, dtype=torch.bool)
        R[:, 2:] = True                                   # signs without attested readings: anything
        for s, rs in ev["sign_reads"].items():
            R[int(s)] = False
            R[int(s), rs] = True
        self.R = R.to(device)

    def read_choice(self, read_logits, signs):
        lg = read_logits.masked_fill(~self.R[signs], float("-inf"))
        return lg.argmax(-1)

    def indep(self, rows, cond):
        out = forward(self.model, rows, self.device)
        allow = self.allow_unknown if cond == "unknown" else self.allow_known
        res = []
        for i, r in enumerate(rows):
            g, M = r["gap"], r["M"]
            sl = out["sign"][i, g:g + M].masked_fill(~allow, float("-inf"))
            sg = sl.argmax(-1)
            rd = self.read_choice(out["read"][i, g:g + M], sg)
            jn = out["join"][i, g:g + M].argmax(-1)
            dt = out["det"][i, g:g + M].argmax(-1)
            keep = (sg != BLANK).cpu().numpy()
            d = {"sign": sg.cpu().numpy()[keep], "read": rd.cpu().numpy()[keep],
                 "join": jn.cpu().numpy()[keep], "det": dt.cpu().numpy()[keep]}
            if M == 1 and cond == "known":
                d["topk"] = sl[0].topk(10).indices.cpu().numpy()
            res.append(d)
        return res

    def iterative(self, rows, cond):
        allow = self.allow_unknown if cond == "unknown" else self.allow_known
        st = [{k: r[k].copy() for k in PLANES if r.get(k) is not None} | {"lang": r["lang"], "gap": r["gap"],
              "M": r["M"], "masked": np.ones(r["M"], bool)} for r in rows]
        while True:
            act = [i for i, s in enumerate(st) if s["masked"].any()]
            if not act:
                break
            out = forward(self.model, [st[i] for i in act], self.device)
            for j, i in enumerate(act):
                s = st[i]
                g, M = s["gap"], s["M"]
                p = torch.softmax(out["sign"][j, g:g + M].masked_fill(~allow, float("-inf")), -1)
                conf, arg = p.max(-1)
                conf = conf.cpu().numpy()
                conf[~s["masked"]] = -1
                k = int(conf.argmax())
                choice = int(arg[k])
                if choice == BLANK:
                    for key in PLANES:
                        if s.get(key) is not None:
                            s[key] = np.delete(s[key], g + k)
                    s["masked"] = np.delete(s["masked"], k)
                    s["M"] -= 1
                else:
                    s["sign"][g + k] = choice
                    s["read"][g + k] = READ_NONE
                    s["masked"][k] = False
        out = forward(self.model, st, self.device)
        res = []
        for i, s in enumerate(st):
            g, M = s["gap"], s["M"]
            sg = torch.from_numpy(s["sign"][g:g + M]).to(self.device)
            rd = self.read_choice(out["read"][i, g:g + M], sg) if M else sg
            res.append({"sign": s["sign"][g:g + M].copy(), "read": rd.cpu().numpy(),
                        "join": out["join"][i, g:g + M].argmax(-1).cpu().numpy(),
                        "det": out["det"][i, g:g + M].argmax(-1).cpu().numpy()})
        return res

    def width_first(self, rows, cond):
        """Unknown width, two stages: (1) one pass decides how many slots are real. In pretraining the
        empty slots of an elastic gap are always the trailing ones, so the width k maximises
        sum_{i<k} log(1 - pB_i) + sum_{i>=k} log pB_i (pB = probability of BLANK among allowed signs).
        (2) the trailing M - k slots are deleted and the k slots are filled as a known-width gap (iterative)."""
        out = forward(self.model, rows, self.device)
        rows2 = []
        for i, r in enumerate(rows):
            g, M = r["gap"], r["M"]
            p = torch.softmax(out["sign"][i, g:g + M].masked_fill(~self.allow_unknown, float("-inf")), -1)[:, BLANK]
            p = p.clamp(1e-6, 1 - 1e-6).double()
            real, blank = torch.log1p(-p), torch.log(p)
            score = [float(real[:k].sum() + blank[k:].sum()) for k in range(1, M + 1)]
            k = 1 + int(np.argmax(score))
            keep = np.r_[np.arange(0, g + k), np.arange(g + M, len(r["sign"]))]
            rows2.append({**{key: r[key][keep] for key in PLANES if r.get(key) is not None},
                          "lang": r["lang"], "gap": g, "M": k})
            if rows2[-1].get("slot") is not None:
                rows2[-1]["slot"] = np.zeros_like(rows2[-1]["slot"])     # the width is decided: a known-width gap
        return self.iterative(rows2, "known")

    def translit(self, chunks):
        res = []
        for s in range(0, len(chunks), BATCH):
            rows = []
            for ti, lg, a in chunks[s:s + BATCH]:
                n = a.shape[1]
                rows.append({"sign": a[0].astype(np.int64), "read": np.full(n, READ_NONE, np.int64),
                             "join": np.full(n, J_UNK, np.int64), "det": np.full(n, D_UNK, np.int64), "lang": lg})
            out = forward(self.model, rows, self.device)
            for i, r in enumerate(rows):
                n = len(r["sign"])
                sg = torch.from_numpy(r["sign"]).to(self.device)
                rl = out["read"][i, :n]
                res.append({"con": self.read_choice(rl, sg).cpu().numpy(),
                            "free": rl[:, 2:].argmax(-1).cpu().numpy() + 2})
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--max-inst", type=int, default=None, help="smoke test")
    ap.add_argument("--beam", type=int, default=0,
                    help="beam search with ranked candidates for known width and width marginalisation for "
                         "unknown width (saved as <name>-beam)")
    ap.add_argument("--width-first", action="store_true",
                    help="only the two-stage unknown-width decoder (saved as <name>-w)")
    a = ap.parse_args()
    ck = torch.load(os.path.join(ROOT, a.ckpt), map_location="cuda", weights_only=False)
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    ev = evalset.load(ROOT, a.split, version=evalset.corpus_version(ROOT))
    if ck.get("corpus_version") != ev["version"]:
        raise SystemExit(f"checkpoint corpus {ck.get('corpus_version')} != evalset corpus {ev['version']}")
    dec = Decoder(model, ev, "cuda")
    inst = ev["instances"][:a.max_inst]
    out = {"ckpt": a.ckpt, "step": ck.get("step"), "split": a.split, "n_inst": len(inst), "version": ev["version"]}
    plan = {"known": (("indep", dec.indep), ("iter", dec.iterative)),
            "unknown": (("indep", dec.indep), ("iter", dec.iterative))}
    if a.width_first:
        plan = {"unknown": (("iter-w", dec.width_first),)}
        a.name += "-w"
    if a.beam:                                      # ranked whole-gap candidates (cun.decode)
        from .decode import Engine, KnownDecoder, UnknownDecoder
        eng = Engine(model, ev, "cuda")
        plan = {"known": (("beam", KnownDecoder(eng, B=a.beam, topk=20)),),
                "unknown": (("marg", UnknownDecoder(eng, B=5, topk=10)),)}
        a.name += "-beam"
    for cond, decoders in plan.items():
        rows = [evalset.gap_input(ev, x, cond) for x in inst]
        order = np.argsort([r["M"] for r in rows], kind="stable")
        for name, fn in decoders:
            t0 = time.time()
            pred = [None] * len(rows)
            for s in range(0, len(order), BATCH):
                idx = order[s:s + BATCH]
                for i, p in zip(idx, fn([rows[i] for i in idx], cond)):
                    pred[i] = p
            out[f"{cond}/{name}"] = pred
            log(cond, name, "done", round(time.time() - t0), "s")
    if not a.width_first and not a.beam:
        chunks = ev["chunks"] if a.max_inst is None else ev["chunks"][:200]
        out["translit"] = dec.translit(chunks)
        log("translit done")
    os.makedirs(os.path.join(RESULTS, "gaps"), exist_ok=True)
    tag = "" if a.max_inst is None else "_smoke"
    with gzip.open(os.path.join(RESULTS, "gaps", f"{a.name}__{a.split}{tag}.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
