"""Pretrain the sign-level encoder.

  python -m cun.train --name base-long [--d-model 768 --depth 12 --lr 3e-4]

Texts are cut into chunks of <= CHUNK signs (lines concatenated, line ends on the join plane),
noised (cun.noise) and packed into rows of ROW positions (segment ids keep chunks apart).
Dev: fixed noisings of the dev texts (span t=0.15, and full transliteration) every --eval-every steps;
best checkpoint by dev loss. Schedule (default, as in Stoicheia): warmup, constant peak LR until the dev
loss stalls (best of the last 4 evals not 0.002 better than the best of the 4 before; not before
--min-anneal-step), then a cosine anneal to 2% over --decay-steps, stopped early if dev has not improved
for 3 evals from mid-anneal on; --max-steps caps the whole run. `--schedule cosine` reproduces the
first runs (base, small: cosine over 60k steps, patience stop). Full state saved every 30 minutes and
resumed; the anneal decision persists in anneal_start.json.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import PAD, CunConfig, CunEncoder, num_params
from .devset import DEV_PATTERNS, WindowSampler, dev_noisings, load_chunks, load_texts
from .noise import D_UNK, IGN, J_UNK, NoiseConfig, noise_chunk
from .paths import CKPT, PROC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROW = 512


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pack(noised, rows, device, leftover=None):
    """noised: list of dicts from noise_chunk -> padded batch tensors with segment ids.
    First-fit, longest first. A chunk that fits in no row is appended to `leftover` (the caller puts it
    into the next batch); nothing is ever discarded. Without `leftover` every chunk must fit."""
    keys = ("sign", "read", "join", "det", "lang", "s_tg", "r_tg", "j_tg", "d_tg", "status", "slot", "w_tg")
    fill = {"sign": PAD, "read": 0, "join": J_UNK, "det": D_UNK, "lang": 0,
            "s_tg": IGN, "r_tg": IGN, "j_tg": IGN, "d_tg": IGN, "status": 3, "slot": 0, "w_tg": IGN}
    R = [[] for _ in range(rows)]
    used = [0] * rows
    for d in sorted(noised, key=lambda d: -len(d["sign"])):
        L = len(d["sign"])
        k = next((i for i in range(rows) if used[i] + L <= ROW), None)
        if k is None:
            if leftover is None:
                raise ValueError("pack: chunk does not fit and no leftover list given")
            leftover.append(d)
            continue
        R[k].append(d)
        used[k] += L
    T = max(used)
    b = {k: np.full((rows, T), fill[k], np.int64) for k in keys}
    b["seg"] = np.full((rows, T), -1, np.int64)
    for i, row in enumerate(R):
        p = 0
        for j, d in enumerate(row):
            L = len(d["sign"])
            for k in keys:
                b[k][i, p:p + L] = d[k]
            b["seg"][i, p:p + L] = j
            p += L
        b["seg"][i, p:] = 10_000 + np.arange(T - p)       # padding attends only to itself
    return {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in b.items()}


def batch_stream(draw, n_lang, rows, seed, device, cfg=NoiseConfig(), stats=None):
    """draw(rng) -> one clean chunk (sign, read, join, det, lang). Chunks left over by the packer open the
    next batch. `stats` (dict) accumulates signs drawn / packed and positions, to verify nothing is lost."""
    rng = np.random.default_rng(seed)
    left = []
    while True:
        noised, tot = left, sum(len(d["sign"]) for d in left)
        left = []
        while tot < rows * ROW * 0.97:
            d = noise_chunk(rng, *draw(rng), n_lang=n_lang, cfg=cfg)
            noised.append(d)
            tot += len(d["sign"])
        b = pack(noised, rows, device, leftover=left)
        if stats is not None:
            stats["drawn"] = stats.get("drawn", 0) + tot - sum(len(d["sign"]) for d in left)
            stats["positions"] = stats.get("positions", 0) + b["sign"].numel()
            stats["carried"] = stats.get("carried", 0) + len(left)
        yield b


class EMA:
    """Exponential moving average of the weights (evaluated and saved in place of the raw weights)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1 + step) / (10 + step))          # warm-up of the average
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def swap(self, model):
        """Exchange the model's weights with the average (call twice to restore)."""
        sd = model.state_dict()
        for k, v in self.shadow.items():
            tmp = sd[k].detach().clone()
            sd[k].copy_(v)
            v.copy_(tmp)


def wsd_mult(step, anneal_start, warmup, decay, min_frac):
    """Warmup-stable-decay (as in Stoicheia's train/schedule.py wsd_dyn): linear warmup, flat peak until
    the anneal starts, cosine to min_frac over `decay` steps."""
    if step < warmup:
        return (step + 1) / warmup
    if step < anneal_start:
        return 1.0
    t = min((step - anneal_start) / max(decay, 1), 1.0)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t))


def stalled(hist, window, eps):
    """Stoicheia's eval_stalled: the best dev loss of the latest window/2 evals is not eps better than the
    best of the window/2 before."""
    if len(hist) < window:
        return False
    v = [h["dev_loss"] for h in hist]
    half = window // 2
    return min(v[-half:]) > min(v[-window:-half]) - eps


def losses(out, b):
    ls = {}
    for h, tg in (("sign", "s_tg"), ("read", "r_tg"), ("join", "j_tg"), ("det", "d_tg")):
        t = b[tg]
        if (t != IGN).any():
            ls[h] = F.cross_entropy(out[h].float().view(-1, out[h].shape[-1]), t.view(-1), ignore_index=IGN)
    total = ls.get("sign", 0) + ls.get("read", 0) + 0.5 * ls.get("join", 0) + 0.5 * ls.get("det", 0)
    if "width" in out and (b["w_tg"] != IGN).any():
        ls["width"] = F.cross_entropy(out["width"].float().view(-1, out["width"].shape[-1]), b["w_tg"].view(-1),
                                      ignore_index=IGN)
        total = total + 0.3 * ls["width"]
    return total, ls


def dev_batches(chunks, n_lang, device, rows=32):
    """Fixed dev noisings (cun.devset): pattern span at t=0.15 and full transliteration."""
    noised = dev_noisings(chunks, n_lang)
    out = []
    for pattern, _ in DEV_PATTERNS:
        ds = [d for p, _, d in noised if p == pattern]
        for s in range(0, len(ds), rows * 2):
            out.append((pattern, pack(ds[s:s + rows * 2], rows, device)))
    return out


@torch.no_grad()
def evaluate(model, dev):
    model.eval()
    tot, n = 0.0, 0
    acc = {"sign": [0, 0], "read": [0, 0]}
    for pattern, b in dev:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        l, _ = losses(out, b)
        tot += float(l)
        n += 1
        for h, tg in (("sign", "s_tg"), ("read", "r_tg")):
            t = b[tg]
            m = t != IGN
            if pattern == 4 and h == "sign":
                continue
            if pattern == 0 and h == "read":
                continue
            acc[h][0] += int((out[h].argmax(-1)[m] == t[m]).sum())
            acc[h][1] += int(m.sum())
    model.train()
    return tot / n, {k: v[0] / max(v[1], 1) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--schedule", choices=("wsd", "cosine"), default="wsd")
    ap.add_argument("--max-steps", type=int, default=300000, help="cosine: length; wsd: hard cap incl. decay")
    ap.add_argument("--warmup-steps", type=int, default=2000, help="wsd")
    ap.add_argument("--decay-steps", type=int, default=30000, help="wsd: length of the anneal")
    ap.add_argument("--min-anneal-step", type=int, default=30000, help="wsd: no anneal before this step")
    ap.add_argument("--stall-window", type=int, default=8, help="wsd: evals compared (two halves)")
    ap.add_argument("--stall-eps", type=float, default=0.002, help="wsd: min dev-loss gain to count as progress")
    ap.add_argument("--anneal-patience", type=int, default=3, help="wsd: stop after this many evals without a best, from mid-anneal")
    ap.add_argument("--min-frac", type=float, default=0.02, help="wsd: final LR as a fraction of peak")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=10, help="cosine only")
    ap.add_argument("--windows", choices=("fixed", "random"), default="fixed",
                    help="fixed: 256-sign chunks cut at fixed offsets; random: line-aligned random windows (R1)")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.0, help="decay of the weight average (0 = off), e.g. 0.999")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--wd-matrices-only", action="store_true", help="weight decay on >= 2-d parameters only")
    ap.add_argument("--noise", default=None, help="JSON overrides of NoiseConfig fields")
    ap.add_argument("--status-plane", action="store_true", help="R2: damage-status input plane")
    ap.add_argument("--elastic-flag", action="store_true", help="O1: flag the slots of unknown-width gaps")
    ap.add_argument("--width-head", action="store_true", help="O1: predict the width of unknown-width gaps")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--max-texts", type=int, default=None, help="smoke test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_dir = os.path.join(CKPT, a.name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "done.json")):
        log("exists")
        return
    torch.manual_seed(a.seed)
    corpus = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    n_lang = len(corpus["langs"])
    train_chunks = load_chunks(corpus, "train", a.max_texts)
    dev_chunks = load_chunks(corpus, "dev", a.max_texts and 200)
    log("corpus version", corpus["version"], "splits", dict(collections.Counter(t["split"] for t in corpus["texts"])))
    log("chunks", len(train_chunks), "dev", len(dev_chunks), "signs",
        sum(len(c[0]) for c in train_chunks))
    cfg = CunConfig(n_sign=len(corpus["sign_vocab"]), n_read=len(corpus["read_vocab"]), n_lang=n_lang,
                    d_model=a.d_model, depth=a.depth, n_heads=a.heads, dropout=a.dropout,
                    status_plane=a.status_plane, elastic_flag=a.elastic_flag, width_head=a.width_head)
    model = CunEncoder(cfg).cuda()
    log("params", num_params(model))
    if a.wd_matrices_only:
        dec = [p for p in model.parameters() if p.ndim >= 2]
        nodec = [p for p in model.parameters() if p.ndim < 2]
        groups = [{"params": dec, "weight_decay": a.wd}, {"params": nodec, "weight_decay": 0.0}]
    else:
        groups = [{"params": list(model.parameters()), "weight_decay": a.wd}]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98))
    ema = EMA(model, a.ema) if a.ema else None
    fwd = torch.compile(model) if a.compile else model
    if a.noise and not a.noise.lstrip().startswith("{"):      # "key=value,key=value" (safe on a shell line)
        a.noise = json.dumps({k.strip(): json.loads(v) for k, v in (kv.split("=", 1) for kv in a.noise.split(","))})
    ncfg = NoiseConfig(**json.loads(a.noise)) if a.noise else NoiseConfig()
    log("noise", ncfg)
    if ncfg.w_damage > 0:
        from . import damage, noise as noise_mod
        noise_mod.DAMAGE_STATS = damage.load_stats()
    if a.windows == "random":
        draw = WindowSampler(load_texts(corpus, "train", a.max_texts))
    else:
        draw = lambda rng: train_chunks[rng.integers(len(train_chunks))]
    if a.schedule == "cosine":
        warm = max(1, int(0.02 * a.max_steps))
        mult = lambda s_, a0: (s_ + 1) / warm if s_ < warm else max(0.0, 0.5 * (1 + math.cos(math.pi * (s_ - warm) / (a.max_steps - warm))))
    else:
        mult = lambda s_, a0: wsd_mult(s_, a0, a.warmup_steps, a.decay_steps, a.min_frac)
    dev = dev_batches(dev_chunks, n_lang, "cuda")
    step, best, bad, hist, anneal_start = 0, float("inf"), 0, [], None
    marker = os.path.join(out_dir, "anneal_start.json")
    last = os.path.join(out_dir, "last.pt")
    state_p = os.path.join(out_dir, "state.json")
    if os.path.exists(last):
        st = torch.load(last, map_location="cuda", weights_only=False)
        if st.get("corpus_version") != corpus["version"]:
            raise SystemExit(f"last.pt was trained on corpus {st.get('corpus_version')}, current is {corpus['version']}")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        if ema is not None and st.get("ema"):
            ema.shadow = st["ema"]
        step, best, bad, hist = st["step"], st["best"], st["bad"], st["hist"]
        if os.path.exists(state_p):                     # evals after the last full save (written every eval)
            sj = json.load(open(state_p))
            best = min(best, sj["best"])                # never let a resumed run overwrite a better best.pt
            hist = [h for h in sj["hist"] if h["step"] <= step]
        log("resumed at", step, "best", best)
    if os.path.exists(marker):
        anneal_start = json.load(open(marker))["step"]
    max_start = a.max_steps - a.decay_steps
    pstats = {}
    stream = batch_stream(draw, n_lang, a.rows, [a.seed, step], "cuda", cfg=ncfg, stats=pstats)
    json.dump({"args": vars(a), "cfg": cfg.__dict__, "corpus_version": corpus["version"]},
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    t_save = t0 = time.time()
    model.train()
    stop = None

    def end_step():
        if a.schedule == "cosine":
            return a.max_steps
        return (anneal_start if anneal_start is not None else max_start) + a.decay_steps

    while step < end_step():
        if a.schedule == "wsd" and anneal_start is None and step >= max_start:
            anneal_start = max_start
            json.dump({"step": anneal_start, "reason": "cap"}, open(marker, "w"))
        lr = a.lr * mult(step, anneal_start if anneal_start is not None else max_start)
        for g in opt.param_groups:
            g["lr"] = lr
        b = next(stream)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fwd(b)
        loss, ls = losses(out, b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if ema is not None:
            ema.update(model, step)
        if step % 100 == 0 and step % a.eval_every:
            log(f"step {step} lr {lr:.2e} loss {float(loss):.4f} " + " ".join(f"{k} {float(v):.3f}" for k, v in ls.items())
                + f" {100 * step / (time.time() - t0 + 1e-9):.0f} steps/100s")
        if step % a.eval_every == 0 or step == end_step():
            if ema is not None:
                ema.swap(model)                         # evaluate and save the averaged weights
            dl, acc = evaluate(model, dev)
            if step % (10 * a.eval_every) == 0 or step == a.eval_every:
                log(f"packing: fill {pstats['drawn'] / max(pstats['positions'], 1):.3f} of positions, "
                    f"{pstats['carried']} chunks carried to a later batch, 0 discarded")
            hist.append({"step": step, "lr": lr, "dev_loss": dl, **{f"dev_acc_{k}": v for k, v in acc.items()},
                         "sec": round(time.time() - t0), "anneal": anneal_start is not None and step > anneal_start})
            log(f"EVAL step {step} lr {lr:.2e} dev loss {dl:.4f} sign acc {acc['sign']:.4f} translit acc {acc['read']:.4f}")
            if dl < best - 1e-4:
                best, bad = dl, 0
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step, "dev_loss": dl,
                            "corpus_version": corpus["version"], "ema": ema is not None},
                           os.path.join(out_dir, "best.pt"))
            else:
                bad += 1
            if ema is not None:
                ema.swap(model)                         # back to the raw weights for training
            json.dump({"best": best, "bad": bad, "hist": hist}, open(state_p + ".tmp", "w"))
            os.replace(state_p + ".tmp", state_p)
            if a.schedule == "cosine" and bad >= a.patience:
                stop = "early"
                log("early stop")
                break
            if a.schedule == "wsd":
                if anneal_start is None and step >= a.min_anneal_step and stalled(hist, a.stall_window, a.stall_eps):
                    anneal_start = step
                    json.dump({"step": step, "reason": "stall"}, open(marker, "w"))
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "step": step,
                                "corpus_version": corpus["version"]}, os.path.join(out_dir, "pre_anneal.pt"))
                    log(f"AUTO-ANNEAL: dev loss stalled; annealing from step {step}, run ends by {end_step()}")
                elif anneal_start is not None and step >= anneal_start + a.decay_steps // 2:
                    ann = [h for h in hist if h["step"] > anneal_start]
                    if len(ann) > a.anneal_patience:
                        bi = min(range(len(ann)), key=lambda i: ann[i]["dev_loss"])
                        if len(ann) - 1 - bi >= a.anneal_patience:
                            stop = "anneal-early"
                            log(f"ANNEAL EARLY-STOP: no dev improvement in {a.anneal_patience} evals")
                            break
        if time.time() - t_save > 1800:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "corpus_version": corpus["version"],
                        "ema": ema.shadow if ema is not None else None,
                        "step": step, "best": best, "bad": bad, "hist": hist}, last + ".tmp")
            os.replace(last + ".tmp", last)
            t_save = time.time()
    json.dump({"steps": step, "best_dev_loss": best, "history": hist, "anneal_start": anneal_start,
               "stopped": stop or "end"}, open(os.path.join(out_dir, "done.json"), "w"), indent=1)
    if os.path.exists(last):
        os.remove(last)
    log("done", step, best)


if __name__ == "__main__":
    main()
