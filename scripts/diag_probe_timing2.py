"""Probe 6, second pass: check the speed-up options found by diag_probe_timing.py on the real loop.

  python scripts/diag_probe_timing2.py

(1) torch.compile: same loss as eager on the same weights and batch? speed of the full loop with fresh
    batches of varying length (as in training), data on the main thread / a thread / worker processes.
(2) boolean instead of additive float attention masks (cun.model.build_attn_mask patched in this process
    only; no repository file is changed).
"""
import gzip, json, os, pickle, sys, time, threading, queue
import multiprocessing as mp

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun.devset import load_chunks  # noqa: E402

OUT = os.path.join(ROOT, "results", "diag", "probes")
corpus = pickle.load(gzip.open(os.path.join(ROOT, "data", "proc", "corpus.pkl.gz"), "rb"))
n_lang = len(corpus["langs"])
chunks = load_chunks(corpus, "train")
del corpus
NW = 4


def worker(q, seed):
    import torch
    torch.set_num_threads(1)
    from cun import train as T
    for b in T.batch_stream(chunks, n_lang, 64, [1, seed], "cpu"):
        q.put({k: v.numpy() for k, v in b.items()})


ctx = mp.get_context("fork")                             # forked before CUDA is initialised
mq = ctx.Queue(16)
procs = [ctx.Process(target=worker, args=(mq, s), daemon=True) for s in range(NW)]
for p in procs:
    p.start()

import torch  # noqa: E402
from cun import model as M  # noqa: E402
from cun import train as T  # noqa: E402
from cun.model import CunConfig, CunEncoder  # noqa: E402

res = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sync():
    torch.cuda.synchronize()


cfg = CunConfig(**json.load(open(os.path.join(ROOT, "ckpt", "base-long", "config.json")))["cfg"])


def new_model():
    torch.manual_seed(0)
    m = CunEncoder(cfg).cuda()
    return m, torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01)


def train_step(model, opt, b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b)
    loss, ls = T.losses(out, b)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    return loss


def main_stream(seed):
    return T.batch_stream(chunks, n_lang, 64, [0, seed], "cuda")


def thread_stream(seed, k=8):
    q = queue.Queue(k)
    def work():
        for x in T.batch_stream(chunks, n_lang, 64, [0, seed], "cuda"):
            q.put(x)
    threading.Thread(target=work, daemon=True).start()
    while True:
        yield q.get()


def proc_stream():
    while True:
        b = mq.get()
        yield {k: torch.from_numpy(v).cuda(non_blocking=True) for k, v in b.items()}


def loop(m, opt, st, n, label, warm=10):
    for _ in range(warm):
        train_step(m, opt, next(st))
    sync(); t0 = time.perf_counter()
    for _ in range(n):
        train_step(m, opt, next(st))
    sync(); dt = time.perf_counter() - t0
    res[label] = {"ms_per_step": 1000 * dt / n, "steps_per_s": n / dt}
    log(label, res[label])
    json.dump(res, open(os.path.join(OUT, "timing2.json"), "w"), indent=1)


def suite(tag):
    # eager
    model, opt = new_model(); model.train()
    loop(model, opt, main_stream(11), 100, f"{tag}eager_main_thread")
    # compiled: equality check first (same weights, same batch, no update)
    cm = torch.compile(model)
    st = main_stream(12)
    bs = [next(st) for _ in range(3)]
    diffs = []
    for b in bs:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            l1 = float(T.losses(model(b), b)[0]); l2 = float(T.losses(cm(b), b)[0])
        diffs.append((l1, l2))
    res[f"{tag}compile_loss_check_eager_vs_compiled"] = diffs
    log("loss eager vs compiled", diffs)
    t0 = time.perf_counter()
    st = main_stream(13)
    for _ in range(60):                                   # compilation over the batch lengths that occur
        train_step(cm, opt, next(st))
    sync(); res[f"{tag}compile_first60_steps_s"] = time.perf_counter() - t0
    loop(cm, opt, main_stream(14), 300, f"{tag}compiled_main_thread", warm=0)
    loop(cm, opt, thread_stream(15), 300, f"{tag}compiled_bg_thread")
    loop(cm, opt, proc_stream(), 300, f"{tag}compiled_{NW}_worker_procs")
    res[f"{tag}peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, cm
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


suite("")

# ---- boolean attention masks (True = may attend), same connectivity as build_attn_mask
def bool_mask(seg_id, window, device, dtype):
    B, Tn = seg_id.shape
    same = seg_id[:, None, :] == seg_id[:, :, None]
    if window and window > 0:
        idx = torch.arange(Tn, device=device)
        same = same & ((idx[None, :] - idx[:, None]).abs() < window)[None]
    return same[:, None]


model, _ = new_model(); model.eval()
b = next(main_stream(16))
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    l1 = float(T.losses(model(b), b)[0])
    orig = M.build_attn_mask
    M.build_attn_mask = bool_mask
    l2 = float(T.losses(model(b), b)[0])
res["bool_mask_loss_check_float_vs_bool"] = (l1, l2)
log("loss float mask vs bool mask", l1, l2)
del model
try:
    suite("boolmask_")
except Exception as e:
    res["boolmask_error"] = repr(e)[:400]
    log("bool mask suite failed", repr(e)[:400])
M.build_attn_mask = orig
json.dump(res, open(os.path.join(OUT, "timing2.json"), "w"), indent=1)
log("done")
os._exit(0)
