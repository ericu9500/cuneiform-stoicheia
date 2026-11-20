"""Headline table: CER, top-1 and top-10 for gaps of 1, 1-5, 1-10 and 1-20 signs (uniform over lengths).

  python scripts/final_table.py --split devA --system R4-beam/beam [--baseline ngram/4gram]

Reads the scorer's JSON (scripts/score_gaps.py). CER is averaged per gap; top-1 = the whole gap exactly
right; top-10 = the right fill among the ten best whole-gap candidates (beam search). Each range averages
over gaps with the same number of gaps at every length. Given for all gaps and for novel gaps (gap + 2
signs of context on each side not found verbatim in the train texts).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cun.paths import RESULTS  # noqa: E402

COLS = ["1", "1-5", "1-10", "1-20"]
ROWS = [("CER", "cer"), ("top-1", "exact"), ("top-10", "ex@10")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="devA")
    ap.add_argument("--system", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--cond", default="known")
    a = ap.parse_args()
    r = json.load(open(os.path.join(RESULTS, "gaps", f"scores__{a.split}.json")))
    S = r["gaps"][a.cond]
    f = lambda v: "--" if v is None else f"{v:.1f}"
    L = [f"# {a.system}, {a.split}, {a.cond} width (corpus {r['version']})", ""]
    for nov, title in (("", "All gaps"), ("novel:", "Novel gaps")):
        L += [f"## {title}", "", "| | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 1) + "|"]
        for lab, key in ROWS:
            L.append(f"| {lab} | " + " | ".join(f(S[a.system][nov + c].get(key)) for c in COLS) + " |")
        if a.baseline:
            for lab, key in ROWS[:2]:
                L.append(f"| {lab}, {a.baseline} | " + " | ".join(f(S[a.baseline][nov + c].get(key)) for c in COLS) + " |")
        L.append("")
    out = os.path.join(RESULTS, "gaps", f"final__{a.system.replace('/', '_')}__{a.split}__{a.cond}.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
