"""Head-to-head, step 3 (torch-free): score system A (sign model), the sign 4-gram and system B (UR study
runs) on the matched positions. Writes results/headtohead/scores.json, scores.md, scores.tex and
predictions.tsv (one row per matched position).

  /nobackup/proj/disk/dionysus/shared/UR/envs/x86/bin/python scripts/h2h_score.py

Criteria. B: hit@1 of the masked WordPiece token (ranks['none'] == 1), mean over the runs of a family
(in-domain: the runs of the fold that holds the tablet out; OOD: all folds and seeds). A / 4-gram, reading
level (comparable with B): the reading chosen for the top-1 sign equals the gold reading (norm_read);
sign level: top-1 sign = gold sign; top-5 by sign.
Intervals: paired bootstrap over tablets, 2,000 draws, percentile 95% (OOD mean: tablets resampled within
each set, mean over the eight sets).
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UR = "/nobackup/proj/disk/dionysus/shared/UR"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(UR, "src"))
from cun.paths import PROC, RESULTS  # noqa: E402

OUT = os.path.join(RESULTS, "headtohead")
FAM = {"mbert/UrIII": "p3__mbert-UrIII__none__fold*__s0", "mbert/all": "p3__mbert-all__none__fold*__s0",
       "mbert/none+mlm15": "p3__mbert-none__mlm15__fold*__s*__lr2e-05__es30", "rand/all": "p3__rand-all__none__fold*__s0",
       "ngram-big (B)": "p2b__ngram-big__fold*"}
A_SYS = ["A line", "A tablet", "4-gram line", "4-gram tablet"]
DRAWS = 2000


def load_run(rid):
    """Read-only (ur.results.load would write a cache)."""
    c = os.path.join(UR, "results", "cache", rid + ".pkl")
    if os.path.exists(c):
        return pickle.load(open(c, "rb"))
    rec = json.load(gzip.open(os.path.join(UR, "results", rid + ".json.gz"), "rt"))
    return {"config": rec["config"], "sets": {s: {"lid": np.asarray(e["lid"]), "pos": np.asarray(e["pos"]),
            "ranks": {"none": np.asarray(e["ranks"]["none"])}} for s, e in rec["evals"].items()}}


def boot(diff, tab, strata, rng):
    """diff per position; tab: tablet id per position; strata: stratum per position (macro mean of strata)."""
    per = []
    for s in np.unique(strata):
        m = strata == s
        u, inv = np.unique(tab[m], return_inverse=True)
        sums = np.bincount(inv, diff[m])
        cnt = np.bincount(inv).astype(float)
        idx = rng.integers(0, len(u), (DRAWS, len(u)))
        per.append(sums[idx].sum(1) / cnt[idx].sum(1))
    d = np.mean(per, 0)
    point = np.mean([diff[strata == s].mean() for s in np.unique(strata)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(point), "lo": float(lo), "hi": float(hi), "n": int(len(diff)), "tablets": int(len(np.unique(tab)))}


def main():
    P = pickle.load(gzip.open(os.path.join(OUT, "positions.pkl.gz"), "rb"))
    R = pickle.load(gzip.open(os.path.join(OUT, "predictions.pkl.gz"), "rb"))
    corpus_rv = None
    pos = P["positions"]
    assert R["key"] == [(x["set"], x["lid"], x["pos"]) for x in pos] and R["version"] == P["version"]
    rv = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    names, rv = rv["sign_names"], rv["read_vocab"]
    n = len(pos)
    sets = P["sets"]
    sname = np.array([x["set"] for x in pos])
    gold_s = np.array([x["gold_sign"] for x in pos])
    gold_r = [x["gold_read"] for x in pos]
    tab = np.array([x["pid"] for x in pos])
    recur = np.array([x["recurring"] for x in pos])
    cat = np.array([x["cat"] for x in pos])
    indom = np.array([s.startswith("fold") for s in sname])
    group = np.where(indom, "in-domain", sname)

    # ---- A and 4-gram
    val, sign_ok, top5 = {}, {}, {}
    pred_txt = {}
    for name, src, cond in (("A line", "model", "line"), ("A tablet", "model", "tablet"),
                            ("4-gram line", "ngram", "line"), ("4-gram tablet", "ngram", "tablet")):
        o = R[src][cond]
        top = o["top"]
        rd = o["read"][:, 0] if o["read"].ndim == 2 else o["read"]
        val[name] = np.array([rv[int(r)] == g for r, g in zip(rd, gold_r)], float)
        sign_ok[name] = (top[:, 0] == gold_s).astype(float)
        top5[name] = (top[:, :5] == gold_s[:, None]).any(1).astype(float)
        pred_txt[name] = [(names[int(s)], rv[int(r)]) for s, r in zip(top[:, 0], rd)]
    strict = {k: val[k] * sign_ok[k] for k in val}

    # ---- B: per-position mean hit over the runs of a family; also on ALL of B's positions
    key = {(x["set"], x["lid"], x["pos"]): i for i, x in enumerate(pos)}
    b_all = {}
    runs_used = {}
    for fam, pat in FAM.items():
        rids = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(UR, "results", pat + ".json.gz")))
        runs_used[fam] = rids
        sm, ct = np.zeros(n), np.zeros(n)
        allh = collections.defaultdict(list)
        for rid in rids:
            r = load_run(rid)
            fold = r["config"]["split"]
            for s in sets:
                if s.startswith("fold") and s != fold + "/test":
                    continue
                e = r["sets"][s]
                h = (e["ranks"]["none"] == 1).astype(float)
                allh[s].append(h.mean())
                for l, p, v in zip(e["lid"].tolist(), e["pos"].tolist(), h.tolist()):
                    i = key.get((s, l, p))
                    if i is not None:
                        sm[i] += v
                        ct[i] += 1
        assert ct.min() >= 1, (fam, ct.min())
        val[fam] = sm / ct
        b_all[fam] = {s: float(np.mean(v)) for s, v in allh.items()}
    st = P["stats"]["by_set"]

    def nall(g):
        return sum(st[s]["all"] for s in sets if (s.startswith("fold") if g == "in-domain" else s == g))

    groups = ["in-domain"] + [s for s in sets if s.startswith("ood/")]
    systems = list(FAM) + A_SYS
    table = {}
    for g in groups:
        m = group == g
        table[g] = {"n_all": nall(g), "n": int(m.sum()), "tablets": int(len(np.unique(tab[m])))}
        for k in systems:
            table[g][k] = float(val[k][m].mean())
        for k in A_SYS:
            table[g][k + " (sign)"] = float(sign_ok[k][m].mean())
            table[g][k + " (sign top-5)"] = float(top5[k][m].mean())
            table[g][k + " (sign+reading)"] = float(strict[k][m].mean())
    ood = [g for g in groups if g != "in-domain"]
    table["OOD mean"] = {k: (int(sum(table[g][k] for g in ood)) if k in ("n", "n_all", "tablets")
                             else float(np.mean([table[g][k] for g in ood]))) for k in table[ood[0]]}

    # ---- B on all its positions vs the matched subset (sanity c)
    bias = {}
    for fam in FAM:
        ind_all = float(np.average([b_all[fam][s] for s in sets if s.startswith("fold")],
                                   weights=[st[s]["all"] for s in sets if s.startswith("fold")]))
        bias[fam] = {"in-domain": {"all": ind_all, "matched": table["in-domain"][fam]},
                     "OOD mean": {"all": float(np.mean([b_all[fam][s] for s in ood])), "matched": table["OOD mean"][fam]}}
        for s in ood:
            bias[fam][s] = {"all": b_all[fam][s], "matched": table[s][fam]}

    # ---- matched share by set and token category
    cats = ["init", "cont", "idx", "det", "num", "sep", "other"]
    share = {}
    for g in groups:
        ss = [s for s in sets if (s.startswith("fold") if g == "in-domain" else s == g)]
        share[g] = {c: [sum(st[s].get("matched/" + c, 0) for s in ss), sum(st[s].get("all/" + c, 0) for s in ss)] for c in cats}
        share[g]["total"] = [sum(st[s].get("matched", 0) for s in ss), sum(st[s]["all"] for s in ss)]
    share["by fold"] = {s: [st[s].get("matched", 0), st[s]["all"]] for s in sets if s.startswith("fold")}

    # ---- bootstrap
    rng = np.random.default_rng(20261120)
    bs = {}
    one = np.zeros(n, int)
    for a in ("A line", "A tablet", "4-gram line"):
        bs[f"{a} - mbert/UrIII | in-domain"] = boot((val[a] - val["mbert/UrIII"])[indom], tab[indom], one[indom], rng)
        bs[f"{a} - mbert/all | OOD mean"] = boot((val[a] - val["mbert/all"])[~indom], tab[~indom], sname[~indom], rng)
        bs[f"{a} - mbert/UrIII | OOD mean"] = boot((val[a] - val["mbert/UrIII"])[~indom], tab[~indom], sname[~indom], rng)
        bs[f"{a} - mbert/all | in-domain"] = boot((val[a] - val["mbert/all"])[indom], tab[indom], one[indom], rng)
    bs["A tablet - A line | in-domain"] = boot((val["A tablet"] - val["A line"])[indom], tab[indom], one[indom], rng)
    bs["A tablet - A line | OOD mean"] = boot((val["A tablet"] - val["A line"])[~indom], tab[~indom], sname[~indom], rng)
    bs["A line - 4-gram line | in-domain"] = boot((val["A line"] - val["4-gram line"])[indom], tab[indom], one[indom], rng)
    bs["A line - 4-gram line | OOD mean"] = boot((val["A line"] - val["4-gram line"])[~indom], tab[~indom], sname[~indom], rng)

    # ---- novel vs recurring lines
    nov = {}
    for lab, mm in (("recurring", recur), ("novel", ~recur)):
        nov[lab] = {}
        for g, gm in (("in-domain", indom), ("OOD (pooled positions)", ~indom)):
            m = mm & gm
            nov[lab][g] = {"n": int(m.sum())} | {k: float(val[k][m].mean()) for k in systems}
        per = [{k: val[k][mm & (sname == s)].mean() for k in systems} for s in ood if (mm & (sname == s)).sum() >= 20]
        nov[lab]["OOD mean"] = {"n": int((mm & ~indom).sum()), "sets": len(per)} | {k: float(np.mean([p[k] for p in per])) for k in systems}
        d = (val["A line"] - val["mbert/UrIII"])
        nov[lab]["boot A line - mbert/UrIII | in-domain"] = boot(d[mm & indom], tab[mm & indom], one[mm & indom], rng)
        d = (val["A tablet"] - val["mbert/UrIII"])
        nov[lab]["boot A tablet - mbert/UrIII | in-domain"] = boot(d[mm & indom], tab[mm & indom], one[mm & indom], rng)
        for a in ("A line", "A tablet"):
            d = val[a] - val["mbert/all"]
            keep = mm & ~indom & np.isin(sname, [s for s in ood if (mm & (sname == s)).sum() >= 20])
            nov[lab][f"boot {a} - mbert/all | OOD mean"] = boot(d[keep], tab[keep], sname[keep], rng)

    # ---- by token category (pooled)
    bycat = {}
    for g, gm in (("in-domain", indom), ("OOD (pooled positions)", ~indom)):
        bycat[g] = {c: {"n": int((gm & (cat == c)).sum())} | {k: float(val[k][gm & (cat == c)].mean()) for k in systems}
                    for c in cats if (gm & (cat == c)).sum() > 0}

    extra = {"gold reading not in the reading vocabulary": float(np.mean([not x["read_in_vocab"] for x in pos])),
             "gold reading not admissible for the gold sign (train)": float(np.mean([not x["read_admissible"] for x in pos])),
             "token has an upper-case letter": float(np.mean([x["token"] != x["token"].lower() for x in pos])),
             "tablet context longer than the line": float(np.mean([P["tablets"][x["pid"]].shape[1] > P["lines"][x["lid"]]["planes"].shape[1] for x in pos])),
             "tablet longer than 256 signs (window cut)": float(np.mean([P["tablets"][x["pid"]].shape[1] > 256 for x in pos]))}
    scores = {"corpus_version": P["version"], "ckpt": R["ckpt"], "model_step": R["model"].get("step"), "n_matched": n,
              "runs": runs_used, "table": table, "bootstrap": bs, "novel_recurring": nov, "by_category": bycat,
              "matched_share": share, "b_all_vs_matched": bias, "unmatched_reasons": P["stats"]["unmatched_reasons"],
              "position_stats": {k: v for k, v in P["stats"].items() if k != "by_set"}, "notes": extra}
    json.dump(scores, open(os.path.join(OUT, "scores.json"), "w"), indent=1)

    with open(os.path.join(OUT, "predictions.tsv"), "w") as fh:
        fh.write("\t".join(["set", "pid", "lid", "pos", "token", "cat", "recurring", "gold_sign", "gold_reading"]
                           + [f"B:{f}" for f in FAM] + [x for k in A_SYS for x in (k + ":sign", k + ":reading", k + ":correct")]) + "\n")
        for i, x in enumerate(pos):
            rowv = [x["set"], x["pid"], x["lid"], x["pos"], x["token"], x["cat"], int(x["recurring"]),
                    names[x["gold_sign"]], x["gold_read"]] + [f"{val[f][i]:.3f}" for f in FAM]
            for k in A_SYS:
                rowv += [pred_txt[k][i][0], pred_txt[k][i][1], int(val[k][i])]
            fh.write("\t".join(str(v) for v in rowv) + "\n")

    # ---- markdown
    def pc(v):
        return f"{100 * v:.1f}"
    cols = ["mbert/UrIII", "mbert/all", "mbert/none+mlm15", "rand/all", "ngram-big (B)", "A line", "A tablet", "4-gram line"]
    md = ["# Head-to-head: sign model (A) vs UR-study models (B) on identical positions", "",
          f"Corpus {P['version']}, checkpoint {R['ckpt']} (step {R['model'].get('step')}). {n} matched positions "
          f"(B's masked WordPiece token = exactly one whole sign with a sign id).", "",
          "B: hit@1 of the token (mean over the runs of the family). A and 4-gram: reading of the top-1 sign = gold "
          "reading (normalised). All numbers in %.", "",
          "## Main table (reading-level hit@1)", "",
          "| set | B positions | matched | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for g in groups + ["OOD mean"]:
        t = table[g]
        md.append(f"| {g.replace('ood/', 'OOD ')} | {t['n_all']} | {t['n']} | " + " | ".join(pc(t[k]) for k in cols) + " |")
    md += ["", "## Sign-level scores of A and the 4-gram", "",
           "| set | " + " | ".join(f"{k} {m}" for k in A_SYS for m in ("sign", "top-5", "sign+reading")) + " |",
           "|---|" + "---|" * (3 * len(A_SYS))]
    for g in groups + ["OOD mean"]:
        t = table[g]
        md.append(f"| {g.replace('ood/', 'OOD ')} | " + " | ".join(pc(t[k + s]) for k in A_SYS for s in (" (sign)", " (sign top-5)", " (sign+reading)")) + " |")
    md += ["", f"## Paired bootstrap (tablets resampled, {DRAWS} draws, 95% percentile interval; percentage points)", "",
           "| difference | set | diff | 95% interval | positions | tablets |", "|---|---|---|---|---|---|"]
    for k, v in bs.items():
        a, g = k.split(" | ")
        md.append(f"| {a} | {g} | {100 * v['diff']:+.2f} | [{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] | {v['n']} | {v['tablets']} |")
    md += ["", "## Novel vs recurring lines (line string occurs verbatim in adapt_UrIII)", "",
           "| lines | set | n | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for lab in ("recurring", "novel"):
        for g in ("in-domain", "OOD (pooled positions)", "OOD mean"):
            t = nov[lab][g]
            md.append(f"| {lab} | {g} | {t['n']} | " + " | ".join(pc(t[k]) for k in cols) + " |")
    md += ["", "| lines | difference | diff | 95% interval |", "|---|---|---|---|"]
    for lab in ("recurring", "novel"):
        for k, v in nov[lab].items():
            if k.startswith("boot "):
                md.append(f"| {lab} | {k[5:]} | {100 * v['diff']:+.2f} | [{100 * v['lo']:+.2f}, {100 * v['hi']:+.2f}] |")
    md += ["", "## How selective the matched subset is (matched / all B positions)", "",
           "| set | total | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
    for g in groups:
        s = share[g]
        md.append(f"| {g.replace('ood/', 'OOD ')} | {s['total'][0]}/{s['total'][1]} ({pc(s['total'][0] / s['total'][1])}) | "
                  + " | ".join(f"{s[c][0]}/{s[c][1]}" for c in cats) + " |")
    md += ["", "Folds: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in share["by fold"].items()), "",
           "Unmatched: " + "; ".join(f"{k}: {v}" for k, v in P["stats"]["unmatched_reasons"].items()), "",
           "## B's hit@1 on all its positions vs on the matched subset", "",
           "| family | in-domain all | in-domain matched | OOD mean all | OOD mean matched |", "|---|---|---|---|---|"]
    for f in FAM:
        b = bias[f]
        md.append(f"| {f} | {pc(b['in-domain']['all'])} | {pc(b['in-domain']['matched'])} | {pc(b['OOD mean']['all'])} | {pc(b['OOD mean']['matched'])} |")
    md += ["", "## By token category (pooled positions)", "", "| set | category | n | " + " | ".join(cols) + " |",
           "|---|---|---|" + "---|" * len(cols)]
    for g, d in bycat.items():
        for c, t in d.items():
            md.append(f"| {g} | {c} | {t['n']} | " + " | ".join(pc(t[k]) for k in cols) + " |")
    md += ["", "## Notes", ""] + [f"- {k}: {pc(v)}% of matched positions" for k, v in extra.items()]
    open(os.path.join(OUT, "scores.md"), "w").write("\n".join(md) + "\n")

    # ---- LaTeX
    tcols = ["mbert/UrIII", "mbert/all", "mbert/none+mlm15", "rand/all", "ngram-big (B)", "4-gram line", "A line", "A tablet"]
    head = ["mBERT/UrIII", "mBERT/all", "mBERT+MLM", "rand/all", "$n$-gram", "4-gram", "line", "tablet"]
    tex = ["% reading-level hit@1 (%) on matched positions; generated by scripts/h2h_score.py",
           "\\begin{tabular}{lr" + "r" * len(tcols) + "}", "\\toprule",
           " & & \\multicolumn{5}{c}{WordPiece models (UR study)} & \\multicolumn{3}{c}{Sign level} \\\\",
           "\\cmidrule(lr){3-7}\\cmidrule(lr){8-10}",
           "Set & $n$ & " + " & ".join(head) + " \\\\", "\\midrule"]
    for g in groups + ["OOD mean"]:
        t = table[g]
        if g == "OOD mean":
            tex.append("\\midrule")
        best = max(t[k] for k in tcols)
        cells = [("\\textbf{" + pc(t[k]) + "}") if t[k] == best else pc(t[k]) for k in tcols]
        lab = {"in-domain": "Ur III admin. (in-domain)"}.get(g, g.replace("ood/", ""))
        tex.append(f"{lab} & {t['n']:,} & ".replace(",", "{,}") + " & ".join(cells) + " \\\\")
        if g == "in-domain":
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(OUT, "scores.tex"), "w").write("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
