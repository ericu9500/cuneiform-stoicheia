"""Realistic-damage gap evaluation (PLAN_v3 stage 0.3) and the pre-registered selection metric.

An instance is one chunk of a held-out text with ONE connected realistic lost region (cun.damage.sample_gap:
edge loss, corner, block of lines, interior hole or medial run transplanted from real damage; at most 20
signs; every lost sign is a clean gold sign). In the sign sequence the region is one or more runs of masked
slots. Width is known (each run has its true number of slots).

Per run: regime (retrieval by verbatim context in the train texts): 'parallel' = the run with 4 signs of
context on each side occurs in train; 'formulaic' = with 2 signs of context but not 4; 'novel' = not even
with 2. Per slot: class 'numeral' (the gold reading is a numeral), 'name' (the slot is in a word that
carries a name determinative: d, m, f, ki, ...), 'other'. Per text: stratum and period bin (corpus v3).

Selection metric (pre-registered): sign error rate (Levenshtein per run / run length, averaged over runs)
on runs in the novel regime, macro-averaged over the strata sux-admin, sux-lit, akk-doc, akk-lit.

  python -m cun.eval3 build --split devA [--n 6000]
  python -m cun.eval3 predict --ckpt <path under CKPT> --name R0a --split devA          (GPU)
  python -m cun.eval3 ngram --split devA                                               (CPU)
  python -m cun.eval3 score --split devA [--base R0a]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import pickle
import re
import time

import numpy as np

from . import damage, evalset
from .consts import FIRST_REAL_SIGN, MASK, READ_MASK
from .devset import LINE_END, check_version
from .noise import D_UNK, J_UNK
from .paths import PROC, RESULTS

STRATA = ("sux-admin", "sux-lit", "akk-doc", "akk-lit")
NAME_DETS = {"d", "m", "f", "ki", "diš", "dingir", "munus", "uru", "kur", "id₂", "i₇", "lu₂"}
NUM = re.compile(r"^\d+(/\d+)?\(")
SEEDS = {"devA": 11, "devB": 12, "test2": 13, "test1": 14, "dev": 15}
MAX_SIGNS = 20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_corpus():
    return pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))


def path(split):
    return os.path.join(PROC, f"eval3_{split}.pkl.gz")


# ------------------------------------------------------------------ build

def build(split, n_inst):
    corpus = load_corpus()
    stats = damage.load_stats()
    chunks = evalset.eval_chunks(corpus, evalset.split_texts(corpus, split))
    texts = corpus["texts"]
    RV = corpus["read_vocab"]
    is_num_read = np.array([bool(NUM.match(r)) for r in RV])
    name_det_read = np.array([r in NAME_DETS for r in RV])
    rng = np.random.default_rng(SEEDS[split])
    inst = []
    order = rng.permutation(len(chunks))
    kinds = collections.Counter()
    tries = 0
    while len(inst) < n_inst and tries < 40 * n_inst:
        ci = int(order[tries % len(order)])
        tries += 1
        ti, lg, a = chunks[ci]
        n = a.shape[1]
        if n < 6:
            continue
        ends = np.flatnonzero(a[2] == LINE_END)
        if not len(ends) or ends[-1] != n - 1:
            ends = np.r_[ends, n - 1]
        ll = np.diff(np.r_[-1, ends])
        elig = (a[0] >= FIRST_REAL_SIGN) & (a[4] == 0)
        mask, info = damage.sample_gap(rng, ll, stats, MAX_SIGNS, stratum=texts[ti].get("stratum"), eligible=elig)
        if mask is None or not mask.any() or mask.sum() > n - 4:
            continue
        pos = np.flatnonzero(mask)
        brk = np.flatnonzero(np.diff(pos) > 1)
        starts = np.r_[pos[0], pos[brk + 1]]
        lens = np.diff(np.r_[0, brk + 1, len(pos)])
        # slot classes
        wend = np.flatnonzero(np.isin(a[2], (0, LINE_END)))
        if not len(wend) or wend[-1] != n - 1:
            wend = np.r_[wend, n - 1]
        wid = np.searchsorted(wend, np.arange(n))
        named = np.zeros(len(wend), bool)
        dn = (a[3] != 0) & name_det_read[a[1]]
        named[wid[dn]] = True
        cls = np.where(is_num_read[a[1][pos]], "numeral", np.where(named[wid[pos]], "name", "other"))
        inst.append({"chunk": ci, "pos": pos.astype(np.int32), "runs": [(int(s), int(l)) for s, l in zip(starts, lens)],
                     "type": info["type"], "cls": cls, "text": int(ti)})
        kinds[info["type"]] += 1
    # regimes per run
    runs = [{"chunk": x["chunk"], "start": s, "L": l} for x in inst for s, l in x["runs"]]
    nov2 = evalset.novelty(corpus, chunks, runs, ctx=2)
    nov4 = evalset.novelty(corpus, chunks, runs, ctx=4)
    k = 0
    for x in inst:
        x["regime"] = []
        for _ in x["runs"]:
            x["regime"].append("novel" if nov2[k] else ("formulaic" if nov4[k] else "parallel"))
            k += 1
    used = sorted({x["chunk"] for x in inst})
    meta = {ci: {kk: texts[chunks[ci][0]].get(kk) for kk in ("stratum", "period_bin", "lang", "src", "id")} for ci in used}
    obj = {"version": corpus["version"], "split": split, "chunks": {ci: chunks[ci] for ci in used}, "meta": meta,
           "instances": inst, "langs": corpus["langs"], "sign_reads": corpus["sign_reads"],
           "sign_names": corpus["sign_names"], "read_vocab": RV}
    with gzip.open(path(split), "wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    reg = collections.Counter(r for x in inst for r in x["regime"])
    st = collections.Counter(meta[x["chunk"]]["stratum"] for x in inst)
    log(split, "instances", len(inst), "runs", len(runs), "signs", sum(len(x["pos"]) for x in inst),
        "types", dict(kinds), "regimes", dict(reg), "strata", dict(st))


def load(split, version=None):
    ev = pickle.load(gzip.open(path(split), "rb"))
    if version is not None:
        check_version(version, ev["version"], f"eval3_{split}")
    return ev


def crop_lines(ev):
    """The same instances with the context cut to the lines that contain lost signs (what a model sees when
    it is given only the broken lines)."""
    chunks, inst = {}, []
    for k, x in enumerate(ev["instances"]):
        ti, lg, a = ev["chunks"][x["chunk"]]
        n = a.shape[1]
        ends = np.flatnonzero(a[2] == LINE_END)
        if not len(ends) or ends[-1] != n - 1:
            ends = np.r_[ends, n - 1]
        starts = np.r_[0, ends[:-1] + 1]
        lo = int(starts[np.searchsorted(ends, x["pos"][0])])
        hi = int(ends[np.searchsorted(ends, x["pos"][-1])]) + 1
        chunks[k] = (ti, lg, a[:, lo:hi])
        y = dict(x)
        y["chunk"], y["pos"] = k, (x["pos"] - lo).astype(np.int32)
        y["runs"] = [(s0 - lo, l) for s0, l in x["runs"]]
        inst.append(y)
    meta = {k: ev["meta"][x["chunk"]] for k, x in enumerate(ev["instances"])}
    return {**ev, "chunks": chunks, "instances": inst, "meta": meta}


def get_ev(name, version=None):
    """Evaluation set by name: 'devA' ...; '<split>-line' = line-only context; 'witness-dev' / 'witness-test'
    = real lacunae answered by another witness (cun.witness), at most 30 per composite."""
    if name.startswith("witness-"):
        from . import witness
        ev = witness.instances(witness.load(name.split("-", 1)[1]), "known", max_per_composite=30)
        if version is not None:
            check_version(version, ev["version"], name)
        return ev
    if name.endswith("-line"):
        return crop_lines(load(name[:-5], version))
    return load(name, version)


def masked_input(ev, x):
    ti, lg, a = ev["chunks"][x["chunk"]]
    planes = {}
    for k, row, fill in (("sign", 0, MASK), ("read", 1, READ_MASK), ("join", 2, J_UNK), ("det", 3, D_UNK)):
        v = a[row].astype(np.int64).copy()
        v[x["pos"]] = fill
        planes[k] = v
    st = a[4].astype(np.int64).copy()
    st[a[0] < FIRST_REAL_SIGN] = 3
    st[x["pos"]] = 3
    return {**planes, "status": st, "lang": lg}


def gold(ev, x):
    return ev["chunks"][x["chunk"]][2][0][x["pos"]].astype(np.int64)


# ------------------------------------------------------------------ predict (GPU)

def predict(ckpt, name, split):
    import torch
    from .evaluate import BATCH, Decoder, forward
    from .model import CunConfig, CunEncoder
    from .consts import READ_NONE
    from .paths import CKPT
    ck = torch.load(os.path.join(CKPT, ckpt), map_location="cuda", weights_only=False)
    ev = get_ev(split)
    if ck.get("corpus_version") != ev["version"]:
        raise SystemExit(f"checkpoint corpus {ck.get('corpus_version')} != eval set corpus {ev['version']}")
    model = CunEncoder(CunConfig(**ck["cfg"])).cuda().eval()
    model.load_state_dict(ck["model"])
    dec = Decoder(model, ev, "cuda")
    inst = ev["instances"]
    out = {"version": ev["version"], "ckpt": ckpt, "step": ck.get("step"), "split": split}
    order = np.argsort([len(x["pos"]) for x in inst], kind="stable")
    one, it = [None] * len(inst), [None] * len(inst)
    with torch.no_grad():
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            rows = [masked_input(ev, inst[i]) for i in idx]
            o = forward(model, rows, "cuda")
            for j, i in enumerate(idx):
                pos = torch.from_numpy(inst[i]["pos"].astype(np.int64)).cuda()
                sl = o["sign"][j, pos].masked_fill(~dec.allow_known, float("-inf"))
                one[i] = sl.argmax(-1).cpu().numpy()
            # iterative: fill the most confident slot of each instance, repeat
            todo = [np.ones(len(inst[i]["pos"]), bool) for i in idx]
            while any(t.any() for t in todo):
                act = [j for j, t in enumerate(todo) if t.any()]
                o = forward(model, [rows[j] for j in act], "cuda")
                for a_, j in enumerate(act):
                    x = inst[idx[j]]
                    pos = torch.from_numpy(x["pos"].astype(np.int64)).cuda()
                    p = torch.softmax(o["sign"][a_, pos].masked_fill(~dec.allow_known, float("-inf")), -1)
                    conf, arg = p.max(-1)
                    conf = conf.cpu().numpy()
                    conf[~todo[j]] = -1
                    k = int(conf.argmax())
                    rows[j]["sign"][x["pos"][k]] = int(arg[k])
                    rows[j]["read"][x["pos"][k]] = READ_NONE
                    todo[j][k] = False
            for j, i in enumerate(idx):
                it[i] = rows[j]["sign"][inst[i]["pos"]].copy()
    out["known/indep"], out["known/iter"] = one, it
    os.makedirs(os.path.join(RESULTS, "eval3"), exist_ok=True)
    with gzip.open(os.path.join(RESULTS, "eval3", f"{name}__{split}.pkl.gz"), "wb") as fh:
        pickle.dump(out, fh, protocol=5)
    log("saved", name, split)


# ------------------------------------------------------------------ n-gram baseline (CPU)

def ngram(split):
    from .devset import load_chunks
    from .ngram import WB, fill_runs
    corpus = load_corpus()
    ev = get_ev(split, corpus["version"])
    train = load_chunks(corpus, "train")
    V = len(corpus["sign_vocab"])
    cands = np.arange(FIRST_REAL_SIGN, V, dtype=np.int64)
    lms = {}
    for gk in ("all", "sux", "akk"):
        sub = train if gk == "all" else [c for c in train if corpus["langs"][c[4]] == gk]
        lms[gk] = WB([c[0] for c in sub], 4, 4096, V)
    preds = []
    for x in ev["instances"]:
        r = masked_input(ev, x)
        lang = ev["langs"][r["lang"]]
        filled, _ = fill_runs(lms.get(lang, lms["all"]), cands, r["sign"], beam=8)
        preds.append(filled[x["pos"]])
    os.makedirs(os.path.join(RESULTS, "eval3"), exist_ok=True)
    with gzip.open(os.path.join(RESULTS, "eval3", f"ngram__{split}.pkl.gz"), "wb") as fh:
        pickle.dump({"version": ev["version"], "split": split, "known/4gram": preds}, fh, protocol=5)
    log("saved ngram", split)


# ------------------------------------------------------------------ score

def run_rows(ev, pred):
    """One row per run: text, stratum, regime, length, SER, exact; and per-slot correctness by class."""
    from rapidfuzz.distance import Levenshtein
    rows, slots = [], collections.defaultdict(lambda: [0, 0])
    for x, p in zip(ev["instances"], pred):
        g = gold(ev, x)
        m = ev["meta"][x["chunk"]]
        k = 0
        for (s, l), reg in zip(x["runs"], x["regime"]):
            gg, pp = g[k:k + l].tolist(), [int(v) for v in p[k:k + l]]
            k += l
            rows.append((x["text"], m["stratum"], reg, l, Levenshtein.distance(pp, gg) / l, float(pp == gg),
                         m["period_bin"], x["type"], len(x["pos"])))
        for c, ok in zip(x["cls"], np.asarray(p) == g):
            slots[c][0] += int(ok)
            slots[c][1] += 1
    return rows, slots


def selection(rows):
    """Novel-regime SER macro-averaged over the four strata; also the per-stratum values."""
    per = {}
    for st in STRATA:
        v = [r[4] for r in rows if r[1] == st and r[2] == "novel"]
        per[st] = (100 * float(np.mean(v)), len(v)) if v else (float("nan"), 0)
    have = [per[s][0] for s in STRATA if per[s][1]]      # strata absent from a set (witness sets) are skipped
    return (float(np.mean(have)) if have else float("nan")), per


def boot_selection(rows_a, rows_b=None, n=2000, seed=0):
    """95% interval of the selection metric (or of the paired difference a - b), texts resampled."""
    rng = np.random.default_rng(seed)
    sel = [(i, STRATA.index(r[1])) for i, r in enumerate(rows_a) if r[1] in STRATA and r[2] == "novel"]
    idx = np.array([i for i, _ in sel])
    st = np.array([s for _, s in sel])
    va = np.array([rows_a[i][4] for i in idx])
    vb = np.array([rows_b[i][4] for i in idx]) if rows_b is not None else None
    tx = np.array([rows_a[i][0] for i in idx])
    u, inv = np.unique(tx, return_inverse=True)
    draws = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)
        vals = []
        present = [s for s in range(len(STRATA)) if (st == s).any()]
        for s in present:
            m = st == s
            if w[m].sum() == 0:
                break
            a_ = (w[m] * va[m]).sum() / w[m].sum()
            vals.append(a_ - ((w[m] * vb[m]).sum() / w[m].sum() if vb is not None else 0.0))
        if present and len(vals) == len(present):
            draws.append(100 * np.mean(vals))
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def score(split, base=None):
    ev = get_ev(split)
    systems = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "eval3", f"*__{split}.pkl.gz"))):
        d = pickle.load(gzip.open(f, "rb"))
        if d.get("version") != ev["version"]:
            log("skipped (other corpus version):", f)
            continue
        nm = os.path.basename(f).split("__")[0]
        for k, v in d.items():
            if "/" in k:
                systems[f"{nm}/{k.split('/')[1]}"] = v
    res, R = {"split": split, "version": ev["version"], "systems": {}}, {}
    for nm, pred in systems.items():
        rows, slots = run_rows(ev, pred)
        R[nm] = rows
        sel, per = selection(rows)
        lo, hi = boot_selection(rows)
        d = {"selection": sel, "selection_ci": [lo, hi], "per_stratum": {k: {"ser": v[0], "n": v[1]} for k, v in per.items()}}
        for reg in ("parallel", "formulaic", "novel"):
            v = [r for r in rows if r[2] == reg]
            d[reg] = {"ser": 100 * float(np.mean([r[4] for r in v])), "exact": 100 * float(np.mean([r[5] for r in v])), "n": len(v)}
        d["all"] = {"ser": 100 * float(np.mean([r[4] for r in rows])), "exact": 100 * float(np.mean([r[5] for r in rows])), "n": len(rows)}
        d["by_type"] = {t: {"ser": 100 * float(np.mean([r[4] for r in rows if r[7] == t])), "n": sum(r[7] == t for r in rows)}
                        for t in sorted({r[7] for r in rows})}
        d["by_period"] = {str(t): {"ser": 100 * float(np.mean([r[4] for r in rows if r[6] == t])), "n": sum(r[6] == t for r in rows)}
                          for t in sorted({str(r[6]) for r in rows}) for t in [t]}
        d["slot_acc"] = {c: {"acc": 100 * a / max(b, 1), "n": b} for c, (a, b) in slots.items()}
        res["systems"][nm] = d
    if base:
        res["vs_base"] = {}
        for nm in systems:
            if nm.split("/")[0] != base and f"{base}/{nm.split('/')[1]}" in R:
                rb = R[f"{base}/{nm.split('/')[1]}"]
                lo, hi = boot_selection(R[nm], rb)
                res["vs_base"][nm] = {"diff": selection(R[nm])[0] - selection(rb)[0], "ci": [lo, hi]}
    os.makedirs(os.path.join(RESULTS, "eval3"), exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "eval3", f"scores__{split}.json"), "w"), indent=1)
    L = [f"# Realistic-damage gaps, {split} (corpus {ev['version']})", "",
         "Selection metric: SER on novel runs, macro over four strata (lower is better).", "",
         "| system | selection [95% CI] | " + " | ".join(STRATA) + " | parallel | formulaic | novel | all | numeral acc | name acc | other acc |",
         "|---" * (10 + len(STRATA)) + "|"]
    for nm, d in res["systems"].items():
        sa = d["slot_acc"]
        L.append(f"| {nm} | {d['selection']:.1f} [{d['selection_ci'][0]:.1f}, {d['selection_ci'][1]:.1f}] | "
                 + " | ".join(f"{d['per_stratum'][s]['ser']:.1f} ({d['per_stratum'][s]['n']})" for s in STRATA)
                 + f" | {d['parallel']['ser']:.1f} | {d['formulaic']['ser']:.1f} | {d['novel']['ser']:.1f} | {d['all']['ser']:.1f} | "
                 + " | ".join(f"{sa.get(c, {'acc': float('nan')})['acc']:.1f}" for c in ("numeral", "name", "other")) + " |")
    if base:
        L += ["", f"Difference to {base} in the selection metric (negative = better), paired bootstrap by text:", ""]
        for nm, v in res["vs_base"].items():
            L.append(f"- {nm}: {v['diff']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
    open(os.path.join(RESULTS, "eval3", f"scores__{split}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "predict", "ngram", "score"))
    ap.add_argument("--split", default="devA")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--base")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.split, a.n)
    elif a.cmd == "predict":
        predict(a.ckpt, a.name, a.split)
    elif a.cmd == "ngram":
        ngram(a.split)
    else:
        score(a.split, a.base)


if __name__ == "__main__":
    main()
