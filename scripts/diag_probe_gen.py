"""Probe 1 (+ per-pattern losses for probe 4) on GPU: the same fixed noisings on dev chunks and on an
equal-sized random sample of train chunks, for best.pt and pre_anneal.pt. Dumps per-chunk sums of the
cross-entropy, target counts and correct counts per head; tables: scripts/diag_probe_report.py.

  python scripts/diag_probe_gen.py
"""
import gzip, os, pickle, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_probe_common as C  # noqa: E402
from cun import train as T  # noqa: E402
from cun.devset import load_chunks  # noqa: E402
from cun.model import PAD, CunConfig, CunEncoder  # noqa: E402
from cun.noise import D_UNK, IGN, J_UNK  # noqa: E402

BATCH = 128
HEADS = (("sign", "s_tg"), ("read", "r_tg"), ("join", "j_tg"), ("det", "d_tg"))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def collate(ds):
    Tm = max(len(d["sign"]) for d in ds)
    fill = {"sign": PAD, "read": 0, "join": J_UNK, "det": D_UNK, "lang": 0, "s_tg": IGN, "r_tg": IGN, "j_tg": IGN, "d_tg": IGN}
    b = {k: np.full((len(ds), Tm), f, np.int64) for k, f in fill.items()}
    seg = np.zeros((len(ds), Tm), np.int64)
    for i, d in enumerate(ds):
        n = len(d["sign"])
        for k in fill:
            b[k][i, :n] = d[k]
        seg[i, n:] = 10_000 + np.arange(Tm - n)
    b["seg"] = seg
    return {k: torch.from_numpy(v).cuda() for k, v in b.items()}


@torch.no_grad()
def score(model, ds):
    """Per chunk and head: [sum CE, n targets, n correct] -> array (n chunks, 4, 3)."""
    res = np.zeros((len(ds), 4, 3))
    order = np.argsort([len(d["sign"]) for d in ds], kind="stable")
    for s in range(0, len(order), BATCH):
        idx = order[s:s + BATCH]
        b = collate([ds[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b)
        for h, (head, tg) in enumerate(HEADS):
            t = b[tg]
            lg = out[head].float()
            ce = F.cross_entropy(lg.view(-1, lg.shape[-1]), t.view(-1), ignore_index=IGN, reduction="none").view(t.shape)
            m = t != IGN
            res[idx, h, 0] = ce.sum(1).cpu().numpy()
            res[idx, h, 1] = m.sum(1).cpu().numpy()
            res[idx, h, 2] = ((lg.argmax(-1) == t) & m).sum(1).cpu().numpy()
    return res


def main():
    corpus = C.load_corpus()
    n_lang = len(corpus["langs"])
    dev = load_chunks(corpus, "dev")
    train_all = load_chunks(corpus, "train")
    pick, tr = C.train_sample(train_all, len(dev))
    del train_all
    data = {"dev": dev, "train": tr}
    nz = {sp: C.noisings(ch, n_lang) for sp, ch in data.items()}
    out = {"train_pick": pick, "langs": corpus["langs"],
           "chunk_lang": {sp: np.array([c[4] for c in ch]) for sp, ch in data.items()},
           "chunk_len": {sp: np.array([len(c[0]) for c in ch]) for sp, ch in data.items()},
           "chunk_idx": {sp: {k: np.array([i for i, _ in v]) for k, v in nz[sp].items()} for sp in nz},
           "pattern_t": {}, "res": {}, "as_logged": {}}
    for name in ("best", "pre_anneal"):
        ck = torch.load(os.path.join(C.ROOT, "ckpt", "base-long", name + ".pt"), map_location="cuda", weights_only=False)
        assert ck.get("corpus_version") == corpus["version"]
        model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
        model.load_state_dict(ck["model"])
        for sp, ch in data.items():
            dl, acc = T.evaluate(model, T.dev_batches(ch, n_lang, "cuda"))      # exactly what cun.train logs
            model.eval()
            out["as_logged"][(name, sp)] = {"loss": dl, **acc}
            log(name, ck.get("step"), sp, "loss as logged by cun.train", round(dl, 4), acc)
            for k, v in nz[sp].items():
                r = score(model, [d for _, d in v])
                out["res"][(name, sp, k)] = r
                tot = r.sum(0)
                log(f"  {k:14s} " + " ".join(f"{h} {tot[i, 0] / max(tot[i, 1], 1):.4f} (n {int(tot[i, 1])}, acc {tot[i, 2] / max(tot[i, 1], 1):.4f})"
                                             for i, (h, _) in enumerate(HEADS)))
        del model
    with gzip.open(os.path.join(C.OUT, "gen_gap.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved")


if __name__ == "__main__":
    main()
