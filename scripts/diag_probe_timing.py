"""Probe 6: where does training time go? (data pipeline on the main thread vs GPU)

  python scripts/diag_probe_timing.py

Uses cun.train's batch_stream / pack / losses unchanged, the full train split, a randomly initialised
model of the base-long size and the base-long optimiser settings. Nothing is saved except the timing JSON.
"""
import collections, gzip, json, os, pickle, sys, threading, queue, time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun.devset import load_chunks  # noqa: E402
from cun.model import CunConfig, CunEncoder  # noqa: E402
from cun.noise import NoiseConfig, noise_chunk  # noqa: E402
from cun import train as T  # noqa: E402

OUT = os.path.join(ROOT, "results", "diag", "probes")
res = {"host": os.uname().nodename, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
       "cpus": len(os.sched_getaffinity(0))}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sync():
    torch.cuda.synchronize()


corpus = pickle.load(gzip.open(os.path.join(ROOT, "data", "proc", "corpus.pkl.gz"), "rb"))
n_lang = len(corpus["langs"])
chunks = load_chunks(corpus, "train")
cfgj = json.load(open(os.path.join(ROOT, "ckpt", "base-long", "config.json")))["cfg"]
cfg = CunConfig(**cfgj)
log("chunks", len(chunks))


def new_model():
    torch.manual_seed(0)
    m = CunEncoder(cfg).cuda()
    o = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01)
    return m, o


def train_step(model, opt, b):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(b)
    loss, ls = T.losses(out, b)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    return loss


# ---- 1. data pipeline alone (200 batches), rows = 64, as in training
N = 200
for dev in ("cpu", "cuda"):
    st = T.batch_stream(chunks, n_lang, 64, [0, 0], dev)
    next(st)
    t0 = time.perf_counter()
    pos = fill = 0
    for _ in range(N):
        b = next(st)
        pos += b["sign"].numel(); fill += int((b["seg"] < 10_000).sum())
    if dev == "cuda":
        sync()
    dt = time.perf_counter() - t0
    res[f"stream_{dev}_ms_per_batch"] = 1000 * dt / N
    res[f"stream_{dev}_batches_per_s"] = N / dt
    res["positions_per_batch"] = pos / N
    res["fill_fraction"] = fill / pos
    log("stream", dev, f"{1000 * dt / N:.1f} ms/batch")

# split of the pipeline: noise_chunk vs pack
rng = np.random.default_rng(0)
t_noise = t_pack = 0.0
nchunk = 0
for _ in range(100):
    t0 = time.perf_counter()
    noised, tot = [], 0
    while tot < 64 * T.ROW * 0.95:
        c = chunks[rng.integers(len(chunks))]
        d = noise_chunk(rng, *c, n_lang=n_lang, cfg=NoiseConfig())
        noised.append(d); tot += len(d["sign"])
    t1 = time.perf_counter()
    T.pack(noised, 64, "cpu")
    t2 = time.perf_counter()
    t_noise += t1 - t0; t_pack += t2 - t1; nchunk += len(noised)
res["noise_ms_per_batch"] = 10 * t_noise
res["pack_ms_per_batch"] = 10 * t_pack
res["chunks_per_batch"] = nchunk / 100
log("noise", 10 * t_noise, "pack", 10 * t_pack, "ms/batch; chunks/batch", nchunk / 100)

# ---- 2. GPU alone: forward/backward/step on pre-built batches (no data work inside the timed loop)
def gpu_only(rows, n=100, compile_=False, label=None):
    model, opt = new_model()
    model.train()
    m = torch.compile(model) if compile_ else model
    st = T.batch_stream(chunks, n_lang, rows, [0, 1], "cuda")
    bs = [next(st) for _ in range(8)]
    t0 = time.perf_counter()
    for i in range(10 if not compile_ else 24):          # warm-up (and compilation of the batch shapes)
        train_step(m, opt, bs[i % 8])
    sync()
    warm = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(n):
        train_step(m, opt, bs[i % 8])
    sync()
    dt = time.perf_counter() - t0
    pos = np.mean([b["sign"].numel() for b in bs])
    r = {"ms_per_step": 1000 * dt / n, "steps_per_s": n / dt, "positions_per_s": pos * n / dt,
         "warmup_s": warm, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    res[label or f"gpu_only_rows{rows}{'_compiled' if compile_ else ''}"] = r
    log(label or rows, compile_, r)
    del model, opt, m
    torch.cuda.empty_cache()
    return r


gpu_only(64)
json.dump(res, open(os.path.join(OUT, "timing.json"), "w"), indent=1)

# forward only / backward share
model, opt = new_model(); model.train()
st = T.batch_stream(chunks, n_lang, 64, [0, 1], "cuda")
bs = [next(st) for _ in range(8)]
for i in range(5):
    train_step(model, opt, bs[i])
sync(); t0 = time.perf_counter()
for i in range(50):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(bs[i % 8])
sync(); res["forward_only_ms"] = 1000 * (time.perf_counter() - t0) / 50
# cost of building the two dense attention masks
from cun.layers import build_attn_mask  # noqa: E402
sync(); t0 = time.perf_counter()
for i in range(50):
    b = bs[i % 8]
    build_attn_mask(b["seg"], 128, "cuda", torch.float32); build_attn_mask(b["seg"], 0, "cuda", torch.float32)
sync(); res["attn_mask_build_ms"] = 1000 * (time.perf_counter() - t0) / 50
log("forward only", res["forward_only_ms"], "mask build", res["attn_mask_build_ms"])

# ---- 3. the training loop as it is (stream on the main thread + GPU), 200 steps
st = T.batch_stream(chunks, n_lang, 64, [0, 2], "cuda")
for i in range(5):
    train_step(model, opt, next(st))
sync(); t0 = time.perf_counter(); t_data = 0.0
for i in range(N):
    a0 = time.perf_counter()
    b = next(st)
    t_data += time.perf_counter() - a0
    loss = train_step(model, opt, b)
    if (i + 1) % 100 == 0:
        float(loss)                                       # the training loop reads the loss every 100 steps
sync(); dt = time.perf_counter() - t0
res["loop_main_thread"] = {"ms_per_step": 1000 * dt / N, "steps_per_s": N / dt,
                           "data_ms_per_step_wallclock": 1000 * t_data / N}
log("loop", res["loop_main_thread"])

# ---- 4. same loop with the stream in one background thread (queue of 8 batches)
def prefetch(gen, k=8):
    q = queue.Queue(k)
    def work():
        for x in gen:
            q.put(x)
    threading.Thread(target=work, daemon=True).start()
    while True:
        yield q.get()


st = prefetch(T.batch_stream(chunks, n_lang, 64, [0, 3], "cuda"))
for i in range(5):
    train_step(model, opt, next(st))
sync(); t0 = time.perf_counter()
for i in range(N):
    train_step(model, opt, next(st))
sync(); dt = time.perf_counter() - t0
res["loop_bg_thread"] = {"ms_per_step": 1000 * dt / N, "steps_per_s": N / dt}
log("loop bg thread", res["loop_bg_thread"])
del model, opt
torch.cuda.empty_cache()
json.dump(res, open(os.path.join(OUT, "timing.json"), "w"), indent=1)

# ---- 5. options: bigger batch, torch.compile
for rows in (128, 256):
    try:
        gpu_only(rows, n=40)
    except Exception as e:                                # e.g. out of memory
        res[f"gpu_only_rows{rows}"] = {"error": repr(e)[:300]}
        log("rows", rows, "failed", repr(e)[:300])
    json.dump(res, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
try:
    gpu_only(64, compile_=True)
except Exception as e:
    res["gpu_only_rows64_compiled"] = {"error": repr(e)[:300]}
    log("compile failed", repr(e)[:300])
json.dump(res, open(os.path.join(OUT, "timing.json"), "w"), indent=1)
log("done")
