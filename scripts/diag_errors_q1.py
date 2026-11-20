"""Q1: accuracy by text / gap / context properties (DEV, known width, iterative decoding).
  python scripts/diag_errors_q1.py -> results/diag/errors/sec1.md"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_errors_lib import OUT, load, md, tables  # noqa: E402

ev, cache, P = load()
I, S = tables(ev, cache, P)
I.to_pickle(OUT + "/I.pkl")
S.to_pickle(OUT + "/S.pkl")
gp = os.path.join(OUT, "gpu_probs.pkl.gz")
if os.path.exists(gp):                                   # one-pass rank of the gold sign at every slot
    from diag_errors_lib import pk
    A = pk(gp)["A"]
    I["slot_top1"] = [float((a["rank"] == 0).mean()) for a in A]
    I["slot_top10"] = [float((a["rank"] < 10).mean()) for a in A]
    I["w_slots"] = I.L
mL = I.groupby("L").ser_iter.mean()
I["ser_adj"] = I.ser_iter - I.L.map(mL) + I.ser_iter.mean()      # SER with the gap-length mix held at the overall mix
out = []


def T(by, title, sub=None, order=None, min_n=1, note=""):
    d = I if sub is None else I[sub]
    g = d.groupby(by, observed=True, dropna=False)
    t = pd.DataFrame({"n": g.size(), "mean L": g.L.mean().round(1), "SER": 100 * g.ser_iter.mean(),
                      "SER (L-adj.)": 100 * g.ser_adj.mean(), "exact": 100 * g.exact_iter.mean(),
                      "CER": 100 * g.cer_iter.mean(), "novel %": 100 * g.novel.mean(),
                      "SER novel only": 100 * d[d.novel].groupby(by, observed=True, dropna=False).ser_adj.mean(),
                      "4-gram SER": 100 * g.ser_4gram.mean()})
    d1 = d[d.L == 1].groupby(by, observed=True, dropna=False)
    t["n(L=1)"] = d1.size()
    t["top1"] = 100 * d1.top1.mean()
    t["top10"] = 100 * d1.top10.mean()
    if "slot_top1" in d:
        w = d.assign(a=d.slot_top1 * d.L, b=d.slot_top10 * d.L).groupby(by, observed=True, dropna=False)
        t["slots"] = w.L.sum()
        t["slot top1 (1-pass)"] = 100 * w.a.sum() / w.L.sum()
        t["slot top10 (1-pass)"] = 100 * w.b.sum() / w.L.sum()
    t["n(L=1)"] = t["n(L=1)"].fillna(0).astype(int)
    t = t[t.n >= min_n]
    if order == "n":
        t = t.sort_values("n", ascending=False)
    out.append(f"\n### {title}\n\n{note}\n\n" + md(t) + "\n")


def bins(col, edges, labels):
    return pd.cut(I[col], edges, labels=labels, right=True, include_lowest=True)


T("lang", "1.1 Language", order="n")
T("period", "1.2 Period (normalised; text metadata)", order="n")
T(["lang", "period"], "1.2b Language x period (n >= 30)", order="n", min_n=30)
T("genre", "1.3 Genre (normalised)", order="n")
T(["lang", "genre"], "1.3b Language x genre (n >= 30)", order="n", min_n=30)
T("src", "1.4 Source (oracc:cdli = the Oracc project that mirrors CDLI texts)", order="n")
T(["lang", "src"], "1.4b Language x source", order="n")
I["text_len"] = bins("text_n", [0, 29, 49, 99, 255, 999, 1e9], ["<30", "30-49", "50-99", "100-255", "256-999", "1000+"])
T("text_len", "1.5 Text length (signs)")
T(["lang", "text_len"], "1.5b Language x text length", min_n=30)
T("L", "1.6 Gap length")
I["Lbin"] = pd.cut(I.L, [0, 1, 5, 10, 15, 20], labels=["1", "2-5", "6-10", "11-15", "16-20"])
T(["Lbin", "novel"], "1.7 Novel vs non-novel, by gap length")
T(["lang", "novel"], "1.7b Novel vs non-novel, by language", min_n=30)
lab = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51+"]
ed = [-1, 0, 2, 5, 10, 20, 50, 1e9]
I["dS"] = bins("d_start", ed, lab)
I["dE"] = bins("d_end", ed, lab)
T("dS", "1.8 Distance from the gap to the chunk start (signs of left context)")
T("dE", "1.8b Distance from the gap to the chunk end (signs of right context)")
I["side"] = np.select([(I.d_start == 0) & (I.d_end == 0), I.d_start == 0, I.d_end == 0, (I.d_start < 5) | (I.d_end < 5)],
                      ["no context", "no left context", "no right context", "<5 signs on one side"], "both sides >= 5")
T("side", "1.8c One-sided context")
I["leftkind"] = np.where(I.d_start > 10, "left ctx > 10", np.where(I.cut_left, "<=10, chunk is cut (text continues before)", "<=10, real text start"))
I["rightkind"] = np.where(I.d_end > 10, "right ctx > 10", np.where(I.cut_right, "<=10, chunk is cut (text continues after)", "<=10, real text end"))
T("leftkind", "1.9 Gap within 10 signs of the chunk start: real text start or cut piece")
T("rightkind", "1.9b Gap within 10 signs of the chunk end: real text end or cut piece")
I["multi"] = np.where(I.n_chunks == 1, "text fits one chunk", np.where(I.chunk_i == 0, "first chunk of a longer text",
                      np.where(I.chunk_i == I.n_chunks - 1, "last chunk", "middle chunk")))
T("multi", "1.9c Chunk is a piece of a longer text (text > 256 signs)")
T(["lang", "multi"], "1.9d ... by language", min_n=30)
I["chunk_len"] = bins("chunk_n", [0, 24, 49, 99, 199, 256], ["<25", "25-49", "50-99", "100-199", "200-256"])
T("chunk_len", "1.9e Chunk length (context available)")
I["dmg3b"] = bins("dmg3", [-1, 0, 1, 2, 6], ["0", "1", "2", "3-6"])
I["unk3b"] = bins("unk3", [-1, 0, 1, 6], ["0", "1", "2+"])
I["dmg10b"] = bins("dmg10", [-1, 0, 2, 5, 10, 20], ["0", "1-2", "3-5", "6-10", "11-20"])
I["unk10b"] = bins("unk10", [-1, 0, 1, 3, 20], ["0", "1", "2-3", "4+"])
I["dmgc"] = bins("dmg_chunk", [-1, 0, 0.05, 0.15, 0.3, 1], ["0", "0-5%", "5-15%", "15-30%", ">30%"])
I["unkc"] = bins("unk_chunk", [-1, 0, 0.02, 0.05, 0.15, 1], ["0", "0-2%", "2-5%", "5-15%", ">15%"])
T("dmg3b", "1.10 Damage near the gap: damaged / restored / unknown signs within 3 signs on each side")
T("unk3b", "1.10b Unknown tokens (x, gap, unk-sign; Oracc's 'X' sign included) within 3 signs on each side")
T("dmg10b", "1.10c Damaged / restored / unknown signs within 10 signs on each side")
T("unk10b", "1.10d Unknown tokens within 10 signs on each side")
T("dmgc", "1.10e Share of the chunk (outside the gap) damaged, restored or unknown")
T("unkc", "1.10f Share of the chunk that is unknown tokens (x, gap, unk-sign, X)")
T(["lang", "dmgc"], "1.10g ... by language", min_n=30)
m = I.L >= 2
T("x_line", "1.11 Gap crosses a line boundary (L >= 2)", sub=m)
T(["Lbin", "x_line"], "1.11b ... by gap length", sub=m)
T("x_word", "1.11c Gap crosses a word boundary (L >= 2)", sub=m)
T(["Lbin", "x_word"], "1.11d ... by gap length", sub=m)
I["align"] = np.select([I.starts_line & I.ends_line, I.starts_word & I.ends_word, I.starts_word, I.ends_word],
                       ["whole line(s)", "whole word(s)", "starts at a word start, ends inside a word",
                        "starts inside a word, ends at a word end"], "starts and ends inside words")
T("align", "1.11e Alignment of the gap with word / line edges")
T(["Lbin", "align"], "1.11f ... by gap length", min_n=30)
open(os.path.join(OUT, "sec1.md"), "w").write("## 1. Accuracy by text, gap and context\n\nKnown width, iterative decoder. SER = Levenshtein sign error rate (scorer's definition); "
     "'SER novel only' is the L-adjusted SER on novel gaps. 'SER (L-adj.)' subtracts the mean SER of the instance's gap length and adds the overall mean (49.4), i.e. the gap-length mix is held constant. "
     "top1/top10: single-sign gaps only (n(L=1)); 'slot top1/top10 (1-pass)': rank of the gold sign in the one-pass distribution at every slot of every gap. "
     "Cells with n < 100 are noise.\n" + "\n".join(out))
print("\n".join(out))
