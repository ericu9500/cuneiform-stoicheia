"""Section 7 of results/decode/REPORT.md: the recommended configurations run end to end through the
drop-in decoders of cun.decode (scripts/decode_run.py --stage final), against the current decoders.
Imported by scripts/decode_score.py."""
from __future__ import annotations

import numpy as np

from cun import decode as D
from decode_score import BINS, boot, head2, mean, row, sys_from_preds

HD = ["exact", "exact@5", "exact@20", "SER", "CER"]


def sec_final(C, F, S, K, W, Wd, tuned):
    P = {}
    for name, preds in F["pred"].items():
        d = sys_from_preds(C, preds, cands=[[D.key(c) for c in p["cands"]] for p in preds])
        if max(len(p["cands"]) for p in preds) < 20:
            d.pop("ex20")
        kk = np.array([len(p["sign"]) for p in preds])
        d["wacc"], d["wmae"] = (kk == C.L).astype(float), np.abs(kk - C.L) / 100.0
        P[name] = d
    cost = {k: v["rows"] / C.N for k, v in F["cost"].items()}
    C.add("## 7. Recommended configurations, end to end through the drop-in decoders", "",
          "These rows come from cun.decode.decode_known / decode_unknown as they would be called from "
          "cun.evaluate (scripts/decode_run.py --stage final), not from the cached candidates of sections "
          "1-6. rows/gap = all forward rows of the decoder, including the final pass for readings, joiners "
          "and determinatives.")
    C.add("### Known width")
    rows = [row(C, "current indep", S["cur_indep"], extra=["1"]), row(C, "current iter", S["cur_iter"], extra=["L + 1 = 11.5"])]
    for name in F["pred"]:
        if name.startswith("known/"):
            rows.append(row(C, name.split("/")[1], P[name], extra=[f"{cost[name]:.0f}"]))
    C.table(head2("decoder", extra=["rows/gap"]), rows)
    cols = ("wacc", "ex1", "ex5", "ex20", "ser", "cer")
    hd = ["width acc"] + HD
    for c, title in (("unknown", "Unknown width (L + extra slots)"), ("r50", "Width known to +-50%"), ("r25", "Width known to +-25%")):
        rows = []
        if c == "unknown":
            for nm, k_ in (("current indep", "cur_unknown/indep"), ("current iter", "cur_unknown/iter"),
                           ("current width_first", "cur_unknown/iter-w")):
                if k_ in S:
                    rows.append(row(C, nm, S[k_], cols=cols, extra=[""]))
        for name in F["pred"]:
            if name.startswith(c + "/"):
                rows.append(row(C, name.split("/")[1], P[name], cols=cols, extra=[f"{cost[name]:.0f}"]))
        C.add(f"### {title}")
        C.table(head2("decoder", cols=hd, extra=["rows/gap"]), rows)
    C.add("### Gains on each half (paired differences, 95% CI by bootstrap over texts)")
    rows = []
    pairs = [("known: " + n.split("/")[1] + " - current iter", P[n], S["cur_iter"]) for n in F["pred"] if n.startswith("known/")]
    if "cur_unknown/iter-w" in S:
        pairs += [("unknown: " + n.split("/")[1] + " - current width_first", P[n], S["cur_unknown/iter-w"])
                  for n in F["pred"] if n.startswith("unknown/")]
    for lab, a, b in pairs:
        for hn, hm in (("A", C.A), ("B", C.B)):
            r = [f"{lab}, half {hn}"]
            for k in ("ex1", "ser", "cer"):
                d, lo, hi = boot((a[k] - b[k])[hm], C.text[hm])
                r.append(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
            rows.append(r)
    C.table(["difference", "exact", "SER", "CER"], rows)
    C.add("By gap length, half B: SER / CER (exact@1 / @5 / @20):")
    rows = []
    systems = [("known: current iter", S["cur_iter"])] + [("known: " + n.split("/")[1], P[n]) for n in F["pred"] if n.startswith("known/")]
    if "cur_unknown/iter-w" in S:
        systems.append(("unknown: current width_first", S["cur_unknown/iter-w"]))
    systems += [(n.split("/")[0] + ": " + n.split("/")[1], P[n]) for n in F["pred"] if not n.startswith("known/")]
    for nm, d in systems:
        r = [nm]
        for b, lo, hi in BINS:
            m = C.B & (C.L >= lo) & (C.L <= hi)
            ex = " / ".join(f"{mean(d[k], m):.0f}" for k in ("ex1", "ex5", "ex20") if k in d)
            r.append(f"{mean(d['ser'], m):.1f} / {mean(d['cer'], m):.1f} ({ex})")
        rows.append(r)
    C.table(["system"] + [f"L {b}" for b, _, _ in BINS], rows)
    C.js["final"] = {n: {h: {k: mean(d[k], hm) for k in d if k != "k"} for h, hm in (("A", C.A), ("B", C.B))} for n, d in P.items()}
