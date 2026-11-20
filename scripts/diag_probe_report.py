"""Tables for the diagnostic probes (torch-free): reads results/diag/probes/{gen_gap,gaps_best}.pkl.gz and
writes results/diag/probes/tables.md + tables.json.   python scripts/diag_probe_report.py"""
import gzip, json, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_probe_common as C

OUT = C.OUT
L = []
J = {}
W = np.array([0.30, 0.20, 0.15, 0.15, 0.20])
PN = ["span", "word", "iid", "elastic", "translit"]
HW = np.array([1, 1, 0.5, 0.5])


def P(s=""):
    L.append(s)


def boot_ci(stat, n, B=1000, seed=0):
    """Percentile CI of stat(index array) resampling n units."""
    rng = np.random.default_rng(seed)
    v = [stat(rng.integers(0, n, n)) for _ in range(B)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def cluster_boot(d, cl, B=2000, seed=0):
    """mean of d with a CI from resampling clusters (texts)."""
    d = np.asarray(d, float)
    u, inv = np.unique(cl, return_inverse=True)
    s, c = np.bincount(inv, d), np.bincount(inv)
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(u), (B, len(u)))
    v = s[k].sum(1) / c[k].sum(1)
    return d.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)


# =================================================================== probe 1
def probe1():
    g = pickle.load(gzip.open(os.path.join(OUT, "gen_gap.pkl.gz"), "rb"))
    R = g["res"]
    P("## Probe 1. Generalisation gap (same noising recipe on dev chunks and on a random sample of train chunks)\n")
    P(f"dev: all {len(g['chunk_len']['dev'])} dev chunks ({int(g['chunk_len']['dev'].sum())} signs); train: {len(g['chunk_len']['train'])} "
      f"random train chunks ({int(g['chunk_len']['train'].sum())} signs; seed 777). Loss = token-pooled cross-entropy per head; "
      "CI = 95% bootstrap over chunks. `devlog` = cun.devset.dev_noisings (600 chunks, as logged during training); "
      "`fix` = every chunk, fixed pattern and rate, language known; `nat` = every chunk, rate t ~ Beta(2,4), planes and language as in training.\n")
    P("Loss exactly as cun.train computes it (mean over batches of sign + read + 0.5 join + 0.5 det):\n")
    P("| checkpoint | dev | train sample | sign acc dev | sign acc train | translit acc dev | translit acc train |")
    P("|---|---|---|---|---|---|---|")
    for ck in ("pre_anneal", "best"):
        a, b = g["as_logged"][(ck, "dev")], g["as_logged"][(ck, "train")]
        P(f"| {ck} | {a['loss']:.4f} | {b['loss']:.4f} | {a['sign']:.4f} | {b['sign']:.4f} | {a['read']:.4f} | {b['read']:.4f} |")
    J["as_logged"] = {f"{k[0]}/{k[1]}": v for k, v in g["as_logged"].items()}

    def ce(r, h, idx=None):
        r = r if idx is None else r[idx]
        return r[:, h, 0].sum() / max(r[:, h, 1].sum(), 1)

    def acc(r, h, idx=None):
        r = r if idx is None else r[idx]
        return r[:, h, 2].sum() / max(r[:, h, 1].sum(), 1)

    sets = [k for k in g["chunk_idx"]["dev"]]
    for ck in ("best", "pre_anneal"):
        P(f"\n### {ck}.pt: sign head (masked signs) and reading head\n")
        P("| noising | n sign targets dev / train | sign CE dev | sign CE train | dev - train [95% CI] | sign acc dev | sign acc train | read CE dev | read CE train | read acc dev | read acc train |")
        P("|---|---|---|---|---|---|---|---|---|---|---|")
        for k in sets:
            d, t = R[(ck, "dev", k)], R[(ck, "train", k)]
            row = [k, f"{int(d[:, 0, 1].sum())} / {int(t[:, 0, 1].sum())}"]
            if d[:, 0, 1].sum():
                lo1, hi1 = boot_ci(lambda i: ce(d, 0, i), len(d), 400, 1)
                lo2, hi2 = boot_ci(lambda i: ce(t, 0, i), len(t), 400, 2)
                diff = ce(d, 0) - ce(t, 0)
                se = np.hypot((hi1 - lo1) / 3.92, (hi2 - lo2) / 3.92)
                row += [f"{ce(d, 0):.3f}", f"{ce(t, 0):.3f}", f"{diff:+.3f} [{diff - 1.96 * se:+.3f}, {diff + 1.96 * se:+.3f}]",
                        f"{100 * acc(d, 0):.1f}", f"{100 * acc(t, 0):.1f}"]
                J[f"p1/{ck}/{k}"] = {"sign_ce_dev": ce(d, 0), "sign_ce_train": ce(t, 0), "diff": diff, "se": se,
                                     "acc_dev": acc(d, 0), "acc_train": acc(t, 0)}
            else:
                row += ["", "", "", "", ""]
            row += [f"{ce(d, 1):.3f}", f"{ce(t, 1):.3f}", f"{100 * acc(d, 1):.1f}", f"{100 * acc(t, 1):.1f}"]
            P("| " + " | ".join(row) + " |")
        P(f"\n{ck}.pt: joiner and determinative heads\n")
        P("| noising | join CE dev | join CE train | det CE dev | det CE train |")
        P("|---|---|---|---|---|")
        for k in sets:
            d, t = R[(ck, "dev", k)], R[(ck, "train", k)]
            P(f"| {k} | {ce(d, 2):.3f} | {ce(t, 2):.3f} | {ce(d, 3):.3f} | {ce(t, 3):.3f} |")

    # training-distribution loss: the five 'nat' sets mixed with the pattern weights (tokens pooled, as in a batch)
    P("\n### Loss under the training noise distribution (five `nat` sets mixed 0.30/0.20/0.15/0.15/0.20, tokens pooled as in a training batch)\n")
    P("| checkpoint | split | total | sign | read | join | det |")
    P("|---|---|---|---|---|---|---|")
    for ck in ("pre_anneal", "best"):
        for sp in ("dev", "train"):
            num = sum(W[p] * R[(ck, sp, f"nat_p{p}")][:, :, 0].sum(0) for p in range(5))
            den = sum(W[p] * R[(ck, sp, f"nat_p{p}")][:, :, 1].sum(0) for p in range(5))
            h = num / den
            J[f"p1/natmix/{ck}/{sp}"] = {"total": float((h * HW).sum()), "heads": h.tolist()}
            P(f"| {ck} | {sp} | {(h * HW).sum():.3f} | {h[0]:.3f} | {h[1]:.3f} | {h[2]:.3f} | {h[3]:.3f} |")
    lt = json.load(open(os.path.join(OUT, "log_trend.json")))
    w = {x["hi"]: x for x in lt["windows"]}
    P(f"\nTraining log for comparison (mean of the single-batch values logged every 100 steps): steps 44-48k total {w[48000]['loss']:.3f} "
      f"(sign {w[48000]['sign']:.3f}, read {w[48000]['read']:.3f}); steps 72-76k total {w[76000]['loss']:.3f} "
      f"(sign {w[76000]['sign']:.3f}, read {w[76000]['read']:.3f}, join {w[76000]['join']:.3f}, det {w[76000]['det']:.3f}).\n")

    # change over the anneal, paired by chunk
    P("### Change from pre_anneal.pt (step 46k) to best.pt (step 76k), sign CE, paired by chunk\n")
    P("| noising | dev: best - pre [95% CI] | train: best - pre [95% CI] |")
    P("|---|---|---|")
    for k in ("devlog_p0", "fix_p0_t15", "fix_p0_t40", "fix_p2_t15", "nat_p0", "nat_p1", "nat_p2", "nat_p3"):
        cells = []
        for sp in ("dev", "train"):
            a, b = R[("best", sp, k)], R[("pre_anneal", sp, k)]
            f = lambda i: ce(a, 0, i) - ce(b, 0, i)
            lo, hi = boot_ci(f, len(a), 400, 3)
            cells.append(f"{f(np.arange(len(a))):+.3f} [{lo:+.3f}, {hi:+.3f}]")
            J[f"p1/anneal/{k}/{sp}"] = [float(f(np.arange(len(a)))), float(lo), float(hi)]
        P(f"| {k} | {cells[0]} | {cells[1]} |")

    # gap by chunk length and language
    P("\n### best.pt, sign CE by chunk length and language (span + word + iid at t = 0.15 pooled: fix_p0/p1/p2_t15)\n")
    P("| group | chunks dev / train | sign targets dev / train | sign CE dev | sign CE train | gap | sign acc dev | sign acc train |")
    P("|---|---|---|---|---|---|---|---|")
    ln, lg = g["chunk_len"], g["chunk_lang"]
    li = {l: i for i, l in enumerate(g["langs"])}
    groups = [(f"length {lo}-{hi}", lambda sp, lo=lo, hi=hi: (ln[sp] >= lo) & (ln[sp] <= hi)) for lo, hi in ((2, 32), (33, 128), (129, 255), (256, 256))] + \
             [(l, lambda sp, l=l: lg[sp] == li[l]) for l in ("sux", "akk")]
    for lab, sel in groups:
        v = {}
        for sp in ("dev", "train"):
            r = sum(R[("best", sp, f"fix_p{p}_t15")] for p in range(3))[sel(sp)]
            v[sp] = (int(sel(sp).sum()), int(r[:, 0, 1].sum()), r[:, 0, 0].sum() / r[:, 0, 1].sum(), r[:, 0, 2].sum() / r[:, 0, 1].sum())
        P(f"| {lab} | {v['dev'][0]} / {v['train'][0]} | {v['dev'][1]} / {v['train'][1]} | {v['dev'][2]:.3f} | {v['train'][2]:.3f} | {v['dev'][2] - v['train'][2]:+.3f} | {100 * v['dev'][3]:.1f} | {100 * v['train'][3]:.1f} |")
        J[f"p1/group/{lab}"] = v

    # sampling noise of the logged dev loss
    P("\n### How large a dev-loss difference is meaningful?\n")
    def devloss(ck, i0, i4):
        a, b = R[(ck, "dev", "devlog_p0")][i0], R[(ck, "dev", "devlog_p4")][i4]
        l0 = sum(HW[h] * a[:, h, 0].sum() / max(a[:, h, 1].sum(), 1) for h in range(4))
        l4 = sum(HW[h] * b[:, h, 0].sum() / max(b[:, h, 1].sum(), 1) for h in range(4))
        return 0.5 * (l0 + l4)
    n = len(R[("best", "dev", "devlog_p0")])
    rng = np.random.default_rng(5)
    one, dif = [], []
    for _ in range(1000):
        i = rng.integers(0, n, n)                      # the same 600 chunks carry both patterns
        one.append(devloss("best", i, i)); dif.append(devloss("best", i, i) - devloss("pre_anneal", i, i))
    full = np.arange(n)
    J["p1/devnoise"] = {"dev_loss_best_pooled": devloss("best", full, full), "se_single": float(np.std(one)),
                        "diff_best_pre": devloss("best", full, full) - devloss("pre_anneal", full, full), "se_paired_diff": float(np.std(dif)),
                        "eval_to_eval_sd_peak_lr": lt["dev_noise_30000-46000"]["loss_sd"],
                        "eval_to_eval_sd_end": lt["dev_noise_60000-76000"]["loss_sd"]}
    d = J["p1/devnoise"]
    P(f"- Logged dev loss (600 dev chunks x 2 noisings; token-pooled recomputation {d['dev_loss_best_pooled']:.4f} for best.pt): "
      f"bootstrap SE over the 600 chunks of the value itself = {d['se_single']:.4f} (irrelevant between runs: all runs share the chunks and noisings).")
    P(f"- Paired difference best.pt - pre_anneal.pt = {d['diff_best_pre']:+.4f}, bootstrap SE {d['se_paired_diff']:.4f}: the sampling error of a "
      "difference between two models on this dev set.")
    P(f"- Eval-to-eval fluctuation of one run (SD of detrended successive differences / sqrt 2): {lt['dev_noise_30000-46000']['loss_sd']:.4f} at peak LR "
      f"(steps 30-46k, n = {lt['dev_noise_30000-46000']['n_evals']} evals; sign acc SD {lt['dev_noise_30000-46000']['sign_acc_sd']:.4f}), "
      f"{lt['dev_noise_60000-76000']['loss_sd']:.4f} late in the anneal (60-76k).")
    P("\n### Training log: loss by 4k-step window\n")
    P("| steps | train total (SE) | train sign | train read | dev loss | dev sign acc |")
    P("|---|---|---|---|---|---|")
    for x in lt["windows"]:
        P(f"| {x['lo'] // 1000}-{x['hi'] // 1000}k | {x['loss']:.3f} ({x['se']:.3f}) | {x['sign']:.3f} | {x['read']:.3f} | {x['dev_loss']:.4f} | {x['dev_sign_acc']:.4f} |")
    P("\nSlope per 10k steps (least squares; SE): " + "; ".join(
        f"{k}: train {v['train'][0]:+.3f} ({v['train'][1]:.3f}), dev {v['dev'][0]:+.4f} ({v['dev'][1]:.4f})" for k, v in lt["slopes"].items()) + "\n")

    # probe 4b
    P("## Probe 4b. Per-pattern loss under the training noise distribution (`nat`; best.pt)\n")
    P("| pattern | weight | share of sign targets in training | share of read targets | sign CE dev | sign CE train | sign acc dev | read CE dev | read CE train | weighted total loss dev (sign + read + 0.5 join + 0.5 det) | train |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    ns = np.array([W[p] * R[("best", "train", f"nat_p{p}")][:, 0, 1].sum() for p in range(5)])
    nr = np.array([W[p] * R[("best", "train", f"nat_p{p}")][:, 1, 1].sum() for p in range(5)])
    for p in range(5):
        d, t = R[("best", "dev", f"nat_p{p}")], R[("best", "train", f"nat_p{p}")]
        td = sum(HW[h] * ce(d, h) for h in range(4)); tt = sum(HW[h] * ce(t, h) for h in range(4))
        s = d[:, 0, 1].sum() > 0
        P(f"| {PN[p]} | {W[p]:.2f} | {100 * ns[p] / ns.sum():.1f}% | {100 * nr[p] / nr.sum():.1f}% | " +
          (f"{ce(d, 0):.3f} | {ce(t, 0):.3f} | {100 * acc(d, 0):.1f}" if s else " | | ") +
          f" | {ce(d, 1):.3f} | {ce(t, 1):.3f} | {td:.3f} | {tt:.3f} |")
        J[f"p4b/{PN[p]}"] = {"sign_ce_dev": ce(d, 0) if s else None, "sign_ce_train": ce(t, 0) if s else None, "read_ce_dev": ce(d, 1),
                             "share_sign_targets": ns[p] / ns.sum(), "share_read_targets": nr[p] / nr.sum()}
    P("\nElastic: sign targets include the BLANK slots. `fix` rows of the tables above give span / word / iid at equal rates (t = 0.15, 0.40).\n")


# =================================================================== probes 2-5
def probe_gaps():
    g = pickle.load(gzip.open(os.path.join(OUT, "gaps_best.pkl.gz"), "rb"))
    R, meta, gold = g["res"], g["meta"], g["gold"]

    def per_inst(key, name, it=False):
        """dict idx -> (slot acc, exact) for one task."""
        r = R[(key, name)]
        pr = r["it_pred"] if it else r["pred"]
        return {int(i): (float((p == gold[key][i]).mean()), float((p == gold[key][i]).all())) for i, p in zip(r["idx"], pr)}

    def changed(key, name, i):
        m = meta[key][i]
        if name.startswith("f_trunc"):
            k = int(name[7:])
            return m["start"] > k or m["n"] - m["start"] - m["L"] > k
        if name == "g_own_line":
            return m["line_lo"] > 0 or m["line_hi"] < m["n"]
        return True

    P("## Probe 2. Context ablations (best.pt, dev, known width)\n")
    P("`ext1` / `ext5`: one gap per eligible dev chunk (cun.evalset rule, seed 101). `ev1` / `ev5`: the 300 instances of evalset_dev. "
      "'affected' = instances whose input the ablation changes (e.g. the chunk is longer than the truncation window); delta = ablated - full "
      "context on the affected instances, 95% CI from a bootstrap over texts. Single-sign gaps: top-1 accuracy. 5-sign gaps: per-slot sign "
      "accuracy with the iterative decoder (independent argmax and exact-gap in tables.json).\n")
    names = ["base", "a_no_readings", "b_no_join_det", "c_signs_only", "d_lang_unknown", "e_lang_wrong"] + \
            [f"f_trunc{k}" for k in C.TRUNC] + ["g_own_line"] + [f"h_drop_first{k}" for k in C.DROP] + [f"h_drop_last{k}" for k in C.DROP] + \
            ["m_extra_mask15", "m_extra_mask40"]
    for key, it in (("ext1", False), ("ev1", False), ("ext5", True), ("ev5", True)):
        base = per_inst(key, "base", it)
        text = np.array([m["text"] for m in meta[key]])
        P(f"\n### {key} ({'iterative decoder, slot accuracy' if it else 'top-1 accuracy'}; n = {len(base)}, full context: "
          f"{100 * np.mean([v[0] for v in base.values()]):.1f}%" + (f", exact gap {100 * np.mean([v[1] for v in base.values()]):.1f}%" if it else "") + ")\n")
        P("| condition | n run | acc all | n affected | full context (affected) | ablated (affected) | delta [95% CI] |" + (" exact gap: full -> ablated (affected) |" if it else ""))
        P("|---|---|---|---|---|---|---|" + ("---|" if it else ""))
        for nm in names:
            if (key, nm) not in R or nm == "base":
                continue
            ab = per_inst(key, nm, it)
            ids = [i for i in ab if changed(key, nm, i)]
            if not ids:
                continue
            b = np.array([base[i][0] for i in ids]); a = np.array([ab[i][0] for i in ids])
            d, lo, hi = cluster_boot(a - b, text[ids])
            # accuracy over all instances: ablated where it applies, else full context (only for conditions run on every instance)
            allacc = np.mean([ab[i][0] if i in ab else base[i][0] for i in base]) if len(ab) == len(base) else float("nan")
            row = f"| {nm} | {len(ab)} | {100 * allacc:.1f} | {len(ids)} | {100 * b.mean():.1f} | {100 * a.mean():.1f} | {100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
            if it:
                row += f" {100 * np.mean([base[i][1] for i in ids]):.1f} -> {100 * np.mean([ab[i][1] for i in ids]):.1f} |"
            P(row)
            J[f"p2/{key}/{nm}"] = {"n": len(ab), "acc_all": allacc, "n_affected": len(ids), "base_aff": b.mean(), "abl_aff": a.mean(),
                                   "delta": d, "ci": [lo, hi]}
            if it:
                ab0, b0 = per_inst(key, nm, False), per_inst(key, "base", False)
                J[f"p2/{key}/{nm}"].update(indep_base_aff=np.mean([b0[i][0] for i in ids]), indep_abl_aff=np.mean([ab0[i][0] for i in ids]),
                                           exact_base_aff=np.mean([base[i][1] for i in ids]), exact_abl_aff=np.mean([ab[i][1] for i in ids]))
        # language conditions by language
        P(f"\n{key}, language label by text language (accuracy %, n):\n")
        P("| language | n | full | label unknown | wrong label (sux <-> akk; others -> sux) | no readings | signs only |")
        P("|---|---|---|---|---|---|---|")
        lang = np.array([m["lang"] for m in meta[key]])
        for lg in ("sux", "akk", "other"):
            ids = [i for i in base if (lang[i] == lg if lg != "other" else lang[i] not in ("sux", "akk"))]
            if not ids:
                continue
            cells = []
            for nm in ("base", "d_lang_unknown", "e_lang_wrong", "a_no_readings", "c_signs_only"):
                pi = per_inst(key, nm, it)
                cells.append(f"{100 * np.mean([pi[i][0] for i in ids]):.1f}")
            P(f"| {lg} | {len(ids)} | " + " | ".join(cells) + " |")
            J[f"p2/{key}/lang/{lg}"] = {"n": len(ids), "cells": cells}
        # by chunk length and novelty (full context)
        P(f"\n{key}, full context, by chunk length and by novelty (gap + 2 context signs each side absent from train):\n")
        P("| group | n | acc |")
        P("|---|---|---|")
        nlen = np.array([m["n"] for m in meta[key]])
        nov = np.array([bool(m["novel"]) for m in meta[key]])
        ids_all = np.array(sorted(base))
        accs = np.array([base[i][0] for i in ids_all])
        for lab, m in [(f"chunk length {lo}-{hi}", (nlen[ids_all] >= lo) & (nlen[ids_all] <= hi)) for lo, hi in ((5, 16), (17, 32), (33, 64), (65, 128), (129, 255), (256, 256))] + \
                      [("novel", nov[ids_all]), ("not novel", ~nov[ids_all])]:
            if m.sum():
                P(f"| {lab} | {int(m.sum())} | {100 * accs[m].mean():.1f} |")
                J[f"p2/{key}/group/{lab}"] = {"n": int(m.sum()), "acc": accs[m].mean()}

    # ---------------------------------------------------------------- probe 4a (already in the tables: m_extra_mask)
    P("\n## Probe 4a. Heavier damage elsewhere in the chunk (extra iid masking of other identified signs)\n")
    P("| set | metric | 0% | +15% | +40% | mean extra masked signs at 15% / 40% |")
    P("|---|---|---|---|---|---|")
    for key, it in (("ext1", False), ("ext5", False), ("ext5", True)):
        b = per_inst(key, "base", it)
        cells = [np.mean([v[0] for v in b.values()])]
        nm_ = []
        for p in (15, 40):
            cells.append(np.mean([v[0] for v in per_inst(key, f"m_extra_mask{p}", it).values()]))
            nm_.append(np.mean(R[(key, f"m_extra_mask{p}")]["info"]["n_masked"]))
        P(f"| {key} (n {len(b)}) | {'slot acc, iterative' if it else ('top-1' if key == 'ext1' else 'slot acc, independent')} | " +
          " | ".join(f"{100 * c:.1f}" for c in cells) + f" | {nm_[0]:.1f} / {nm_[1]:.1f} |")
        J[f"p4a/{key}/{'iter' if it else 'indep'}"] = cells
    # by chunk length
    P("\next1 top-1 by chunk length:\n")
    P("| chunk length | n | 0% | +15% | +40% |")
    P("|---|---|---|---|---|")
    nlen = np.array([m["n"] for m in meta["ext1"]])
    b = per_inst("ext1", "base"); m15 = per_inst("ext1", "m_extra_mask15"); m40 = per_inst("ext1", "m_extra_mask40")
    for lo, hi in ((5, 32), (33, 128), (129, 256)):
        ids = [i for i in b if lo <= nlen[i] <= hi]
        P(f"| {lo}-{hi} | {len(ids)} | " + " | ".join(f"{100 * np.mean([x[i][0] for i in ids]):.1f}" for x in (b, m15, m40)) + " |")

    # ---------------------------------------------------------------- probe 3
    P("\n## Probe 3. Calibration and confidence\n")
    def auroc(conf, ok):
        from scipy.stats import rankdata
        r = rankdata(conf)
        n1, n0 = ok.sum(), (~ok).sum()
        return (r[ok].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    def calib(conf, ok, title, cl=None):
        edges = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(conf, edges) - 1, 0, 9)
        ece = sum(abs(ok[b == k].mean() - conf[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any())
        P(f"\n{title}: n = {len(ok)}, accuracy {100 * ok.mean():.1f}%, mean confidence {100 * conf.mean():.1f}%, ECE (10 bins) {100 * ece:.1f} pp, "
          f"AUROC of confidence for right vs wrong {auroc(conf, ok):.3f}\n")
        P("| confidence bin | n | share | mean confidence | accuracy |")
        P("|---|---|---|---|---|")
        for k in range(10):
            if (b == k).any():
                P(f"| {edges[k]:.1f}-{edges[k + 1]:.1f} | {(b == k).sum()} | {100 * (b == k).mean():.1f}% | {100 * conf[b == k].mean():.1f} | {100 * ok[b == k].mean():.1f} |")
        J[f"p3/{title}"] = {"n": len(ok), "acc": ok.mean(), "conf": conf.mean(), "ece": ece, "auroc": auroc(conf, ok)}
        # selective accuracy
        o = np.argsort(-conf)
        P("\nSelective accuracy (keep the most confident share): " + ", ".join(
            f"top {int(100 * f)}%: {100 * ok[o[:max(1, int(f * len(o)))]].mean():.1f}" for f in (0.25, 0.5, 0.75, 1.0)))
        return ece

    r = R[("multi1", "base")]
    conf = np.array([c[0] for c in r["conf"]]); ok = np.array([p[0] == gold["multi1"][i][0] for i, p in zip(r["idx"], r["pred"])])
    calib(conf, ok, "single-sign gaps, multi1 (up to 12 per dev chunk)")
    nov = np.array([bool(meta["multi1"][i]["novel"]) for i in r["idx"]])
    calib(conf[nov], ok[nov], "single-sign gaps, multi1, novel only")
    r1 = R[("ext1", "base")]
    calib(np.array([c[0] for c in r1["conf"]]), np.array([p[0] == gold["ext1"][i][0] for i, p in zip(r1["idx"], r1["pred"])]),
          "single-sign gaps, ext1 (one per chunk)")
    rank = np.array([x[0] for x in r["rank"]])
    P(f"\nmulti1 top-k: top-1 {100 * (rank == 0).mean():.1f}, top-5 {100 * (rank < 5).mean():.1f}, top-10 {100 * (rank < 10).mean():.1f}, top-20 {100 * (rank < 20).mean():.1f} "
      f"(n = {len(rank)}; {len({meta['multi1'][i]['text'] for i in r['idx']})} texts)\n")
    J["p3/topk"] = [(rank < k).mean() for k in (1, 5, 10, 20)]

    # iterative decoding: accuracy by fill order
    P("### Iterative decoding: accuracy by fill order\n")
    for key in ("ext5", "evL"):
        r = R[(key, "base")]
        rows = []
        for i, pi, pit, od, cf, c0 in zip(r["idx"], r["pred"], r["it_pred"], r["it_order"], r["it_conf"], r["conf"]):
            gl = gold[key][i]
            Lg = len(gl)
            for k in range(Lg):
                rows.append((Lg, od[k], pit[k] == gl[k], pi[k] == gl[k], cf[k], c0[k], min(k, Lg - 1 - k), meta[key][i]["text"]))
        a = np.array([(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in rows], float)
        if key == "ext5":
            P(f"\n5-sign gaps (ext5, n = {len(r['idx'])} gaps):\n")
            P("| fill rank | n slots | accuracy (iterative) | accuracy of the same slots with independent argmax | mean confidence at fill | mean first-pass confidence |")
            P("|---|---|---|---|---|---|")
            for k in range(5):
                m = a[:, 1] == k
                P(f"| {k + 1} | {int(m.sum())} | {100 * a[m, 2].mean():.1f} | {100 * a[m, 3].mean():.1f} | {100 * a[m, 4].mean():.1f} | {100 * a[m, 5].mean():.1f} |")
                J[f"p3/order/ext5/{k + 1}"] = {"n": int(m.sum()), "it": a[m, 2].mean(), "indep": a[m, 3].mean(), "conf": a[m, 4].mean()}
            P("\nBy position in the gap (0 = edge slot, 2 = middle):\n")
            P("| distance to gap edge | n | iterative | independent | mean fill rank |")
            P("|---|---|---|---|---|")
            for k in range(3):
                m = a[:, 6] == k
                P(f"| {k} | {int(m.sum())} | {100 * a[m, 2].mean():.1f} | {100 * a[m, 3].mean():.1f} | {1 + a[m, 1].mean():.2f} |")
        else:
            P(f"\nEvaluation-set gaps L = 2..20 (evL, n = {len(r['idx'])} gaps, {len(a)} slots), fill order as a fraction of the gap:\n")
            P("| L bin | fill-order quintile | n slots | accuracy (iterative) | independent argmax, same slots | mean confidence at fill |")
            P("|---|---|---|---|---|---|")
            q = np.minimum((5 * a[:, 1] / a[:, 0]).astype(int), 4)
            for lo, hi in ((2, 5), (6, 10), (11, 20)):
                for k in range(5):
                    m = (a[:, 0] >= lo) & (a[:, 0] <= hi) & (q == k)
                    if m.sum():
                        P(f"| {lo}-{hi} | {k + 1} | {int(m.sum())} | {100 * a[m, 2].mean():.1f} | {100 * a[m, 3].mean():.1f} | {100 * a[m, 4].mean():.1f} |")
                        J[f"p3/order/evL/{lo}-{hi}/{k + 1}"] = {"n": int(m.sum()), "it": a[m, 2].mean(), "indep": a[m, 3].mean(), "conf": a[m, 4].mean()}
        conf_f, ok_f = a[:, 4], a[:, 2].astype(bool)
        calib(conf_f, ok_f, f"confidence at fill time, iterative decoder, {key}")
        first = a[:, 1] == 0
        P(f"\n{key}: first-filled slot right in {100 * a[first, 2].mean():.1f}% of gaps (n = {int(first.sum())}).")
        J[f"p3/order/{key}/first_right"] = a[first, 2].mean()

    # ---------------------------------------------------------------- probe 5
    P("\n## Probe 5. Repeated material and distance (single-sign gaps, best.pt)\n")
    P("Trigram = left neighbour, gold sign, right neighbour. `multi1` (up to 12 random gaps per dev chunk) gives the groups without a "
      "repeat and with a repeat nearer than 128 signs; `far1` is every dev position whose nearest repeat is >= 128 signs away (outside the "
      "local attention window; reachable only through the global blocks or by relaying). 'repeats masked': the centre sign of every other "
      "occurrence is masked too; control: the same number of random other signs masked. CI: bootstrap over texts.\n")
    P("| group | n gaps | texts | top-1 | repeats masked | random control | effect of masking repeats vs control [95% CI] |")
    P("|---|---|---|---|---|---|---|")

    def grp(key, lab, sel):
        r = R[(key, "base")]
        ok = {int(i): float(p[0] == gold[key][i][0]) for i, p in zip(r["idx"], r["pred"])}
        ids = [i for i in ok if sel(i)]
        if not ids:
            return
        txt = np.array([meta[key][i]["text"] for i in ids])
        row = f"| {lab} | {len(ids)} | {len(set(txt))} | {100 * np.mean([ok[i] for i in ids]):.1f} |"
        d = {"n": len(ids), "texts": len(set(txt)), "acc": np.mean([ok[i] for i in ids])}
        ra, rc = R[(key, "r_mask_repeats")], R[(key, "r_mask_random_control")]
        ma = {int(i): float(p[0] == gold[key][i][0]) for i, p in zip(ra["idx"], ra.get("pred", []))}
        mc = {int(i): float(p[0] == gold[key][i][0]) for i, p in zip(rc["idx"], rc.get("pred", []))}
        both = [i for i in ids if i in ma and i in mc]
        if both:
            t2 = np.array([meta[key][i]["text"] for i in both])
            e, lo, hi = cluster_boot(np.array([ma[i] - mc[i] for i in both]), t2)
            row += f" {100 * np.mean([ma[i] for i in both]):.1f} | {100 * np.mean([mc[i] for i in both]):.1f} | {100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |"
            d.update(masked=np.mean([ma[i] for i in both]), control=np.mean([mc[i] for i in both]), effect=e, ci=[lo, hi])
        else:
            row += " | | |"
        P(row)
        J[f"p5/{lab}"] = d

    rep = g["repeat"]
    M1, F1 = meta["multi1"], meta["far1"]
    dist = lambda key, i: (rep[key][i]["dist"] if rep[key][i] is not None else -2)
    grp("multi1", "all testable (both neighbours identified)", lambda i: dist("multi1", i) != -2)
    grp("multi1", "no repeat in chunk", lambda i: dist("multi1", i) == -1)
    grp("multi1", "no repeat, chunk >= 131 signs", lambda i: dist("multi1", i) == -1 and M1[i]["n"] >= 131)
    grp("multi1", "no repeat, but gold sign occurs elsewhere in chunk", lambda i: dist("multi1", i) == -1 and rep["multi1"][i]["uni_dist"] > 0)
    grp("multi1", "no repeat, gold sign nowhere else in chunk", lambda i: dist("multi1", i) == -1 and rep["multi1"][i]["uni_dist"] == -1)
    grp("multi1", "repeat at < 128", lambda i: 0 < dist("multi1", i) < 128)
    grp("multi1", "repeat at < 128, chunk >= 131 signs", lambda i: 0 < dist("multi1", i) < 128 and M1[i]["n"] >= 131)
    for lo, hi in ((1, 8), (9, 32), (33, 64), (65, 127)):
        grp("multi1", f"repeat at {lo}-{hi}", lambda i: lo <= dist("multi1", i) <= hi)
    grp("far1", "repeat only at >= 128 (far1)", lambda i: True)
    for lo, hi in ((128, 159), (160, 199), (200, 255)):
        grp("far1", f"repeat only at {lo}-{hi}", lambda i: lo <= dist("far1", i) <= hi)
    grp("far1", "repeat only at >= 128, earlier in the chunk", lambda i: rep["far1"][i]["dist_earlier"] >= 128)
    grp("far1", "repeat only at >= 128, later in the chunk", lambda i: rep["far1"][i]["dist_earlier"] == -1)
    grp("far1", "repeat only at >= 128, novel gaps", lambda i: bool(F1[i]["novel"]))
    grp("multi1", "repeat at < 128, novel gaps", lambda i: 0 < dist("multi1", i) < 128 and bool(M1[i]["novel"]))
    grp("multi1", "no repeat, novel gaps", lambda i: dist("multi1", i) == -1 and bool(M1[i]["novel"]))


which = sys.argv[1:] or ["1", "gaps"]
if "1" in which:
    probe1()
if "gaps" in which:
    probe_gaps()
open(os.path.join(OUT, "tables.md"), "w").write("\n".join(L) + "\n")
json.dump(J, open(os.path.join(OUT, "tables.json"), "w"), indent=1, default=float)
print("\n".join(L))
