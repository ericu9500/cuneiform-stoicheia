"""Audit the pretraining corpus (gate before a pretraining run). Prints and writes logs/audit_<tag>.json.

  python scripts/audit_corpus.py --tag v2
  CUN_PROC=data/proc_v3 python scripts/audit_corpus.py --tag v3 --compare data/proc

Checks: share of special signs overall and per source; readings missing; plane distributions; reading
vocabulary (case variants, ŋ, markup); readings per sign; top-reading agreement between Oracc and CDLI
for frequent signs; dev / test texts with a near-duplicate (line-set Jaccard >= 0.9, recomputed here
independently of the builder) among the train texts; split sizes; corpus version.
Corpus v3 (texts carry `composites`): the held-out labels are devA / devB / test1 / test2; in addition no
held-out text may share a composite (Q-number) or a museum number with a train text; devA and devB may not
near-duplicate or share a composite with each other; test2 has one text per composite; reserved_ids.json
lists exactly the held-out texts; split sizes per stratum and period bin are printed. `--compare DIR`
checks that vocabularies, texts and per-line arrays equal those of the corpus in DIR (v2) and that the
labels are consistent with it (test1 = its test, devA + devB = its dev, test2 and train from its train).
FAIL lines mark values outside the thresholds below.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import pickle
import re

import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun.composites import MUS_MAX  # noqa: E402
from cun.paths import PROC  # noqa: E402

HELD = ("dev", "test", "devA", "devB", "test1", "test2")
MAX_UNK = {"oracc": 0.05, "cdli": 0.05}
MAX_CASE_GROUPS = 0
MAX_DEV_DUP = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--compare", default=None, help="processed-data directory of the corpus to compare with (v2)")
    a = ap.parse_args()
    c = pickle.load(gzip.open(os.path.join(PROC, "corpus.pkl.gz"), "rb"))
    T, SV, RV = c["texts"], c["sign_vocab"], c["read_vocab"]
    out, fails = {}, []
    allv = np.concatenate([a2 for _, a2 in c["lines"]], 1)
    src = np.concatenate([np.full(a2.shape[1], T[ti]["src"].startswith("oracc")) for ti, a2 in c["lines"]])
    lang = np.concatenate([np.full(a2.shape[1], T[ti]["lang"]) for ti, a2 in c["lines"]])
    n = allv.shape[1]
    out["signs"] = n
    out["special"] = {SV[i]: round(100 * float((allv[0] == i).mean()), 2) for i in (3, 4, 5)}
    out["no_reading_pct"] = round(100 * float((allv[1] == 0).mean()), 2)
    for nm, m in (("oracc", src), ("cdli", ~src)):
        v = allv[:, m]
        d = {"signs": int(m.sum()), "unk_sign_pct": round(100 * float((v[0] == 3).mean()), 2),
             "x_pct": round(100 * float((v[0] == 4).mean()), 2),
             "no_reading_pct": round(100 * float(((v[1] == 0) & (v[0] >= 6)).mean()), 2),
             "missing_pct": round(100 * float((v[4] == 2).mean()), 1)}
        out[nm] = d
        if d["unk_sign_pct"] > 100 * MAX_UNK[nm]:
            fails.append(f"{nm} unk-sign {d['unk_sign_pct']}%")
    for lg in ("sux", "akk"):
        m = lang == lg
        out[f"unk_sign_pct_{lg}"] = round(100 * float((allv[0, m] == 3).mean()), 2)
    out["join"] = {c["joins"][k]: round(100 * v / n, 1) for k, v in collections.Counter(allv[2].tolist()).items()}
    out["det"] = {c["dets"][k] or "none": round(100 * v / n, 2) for k, v in collections.Counter(allv[3].tolist()).items()}
    low = collections.defaultdict(list)
    for r in RV[2:]:
        low[r.lower()].append(r)
    groups = {k: v for k, v in low.items() if len(v) > 1}
    out["read_vocab"] = len(RV)
    out["case_variant_groups"] = len(groups)
    out["readings_with_eng"] = sum("ŋ" in r for r in RV)
    out["readings_with_markup"] = [r for r in RV[2:] if re.search(r"[\[\]#!?<>|{}]", r)][:30]
    if len(groups) > MAX_CASE_GROUPS or out["readings_with_eng"]:
        fails.append(f"reading conventions: {len(groups)} case groups, {out['readings_with_eng']} with ŋ")
    k = np.array([len(v) for v in c["sign_reads"].values()])
    out["readings_per_sign"] = {"signs": len(k), "median": float(np.median(k)), "max": int(k.max())}
    cnt = {True: collections.defaultdict(collections.Counter), False: collections.defaultdict(collections.Counter)}
    for s, r, o in zip(allv[0], allv[1], src):
        if r > 1:
            cnt[bool(o)][int(s)][int(r)] += 1
    agree = tot = 0
    dis = []
    for s in cnt[True]:
        if s in cnt[False] and sum(cnt[True][s].values()) > 200 and sum(cnt[False][s].values()) > 200:
            x, y = cnt[True][s].most_common(1)[0][0], cnt[False][s].most_common(1)[0][0]
            tot += 1
            agree += x == y
            if x != y:
                dis.append((c["sign_names"][s], RV[x], RV[y]))
    out["top_reading_agree"] = f"{agree}/{tot}"
    out["top_reading_disagree_examples"] = dis[:25]
    # near-duplicates of dev / test texts among the texts used for training
    out["version"] = c["version"]
    out["splits"] = dict(collections.Counter(t["split"] for t in T))
    keys = collections.defaultdict(set)
    for ti, a2 in c["lines"]:
        if (a2[0] >= 6).any():                          # a line of only special tokens identifies nothing
            keys[ti].add(tuple(a2[0].tolist()))
    voc, rows, cols = {}, [], []
    for ti in range(len(T)):
        for kk in keys[ti]:
            rows.append(ti)
            cols.append(voc.setdefault(kk, len(voc)))
    X = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(T), len(voc)))
    size = np.asarray(X.sum(1)).ravel()
    XT = X.T.tocsr()
    used = np.array([t["split"] == "train" for t in T])
    def near_dups(q, target):
        """Texts of q with line-set Jaccard >= 0.9 to a text flagged in `target` (other than itself)."""
        dup = set()
        for s in range(0, len(q), 256):
            qq = q[s:s + 256]
            inter = (X[qq] @ XT).tocoo()
            qi = qq[inter.row]
            jac = inter.data / (size[qi] + size[inter.col] - inter.data)
            m = (jac >= 0.9) & target[inter.col] & (inter.col != qi)
            dup.update(qi[m].tolist())
        return dup

    labels = [nm for nm in HELD if any(t["split"] == nm for t in T)]
    for nm in labels:
        q = np.array([i for i, t in enumerate(T) if t["split"] == nm])
        dup = near_dups(q, used)
        out[f"{nm}_with_train_near_dup_pct"] = round(100 * len(dup) / len(q), 2)
        if len(dup) / len(q) > MAX_DEV_DUP:
            fails.append(f"{nm} texts with a train near-duplicate: {len(dup)}")
    if "composites" in T[0]:                            # corpus v3
        v3_checks(T, out, fails, near_dups, c["version"])
    if a.compare:
        compare(c, a.compare, out, fails)
    out["train_texts"] = int(used.sum())
    out["train_signs"] = int(sum(T[i]["n"] for i in np.flatnonzero(used)))
    out["fails"] = fails
    json.dump(out, open(os.path.join(ROOT, "logs", f"audit_{a.tag}.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("AUDIT", "FAIL" if fails else "PASS", fails)


def v3_checks(T, out, fails, near_dups, version):
    """Composite / object separation of the held-out sets, the dev halves, test2, reserved_ids.json, tables."""
    mus_n = collections.Counter(k for t in T for k in t["museum"])
    keys = [{"Q:" + q for q in t["composites"]} | {"M:" + k for k in t["museum"] if mus_n[k] <= MUS_MAX} for t in T]
    train_keys = set().union(*(keys[i] for i, t in enumerate(T) if t["split"] == "train"))
    held = [i for i, t in enumerate(T) if t["split"] in HELD]
    for nm in ("devA", "devB", "test1", "test2"):
        ix = [i for i in held if T[i]["split"] == nm]
        bad = [T[i]["id"] for i in ix if keys[i] & train_keys]
        out[f"{nm}_texts"] = len(ix)
        out[f"{nm}_with_composite"] = sum(bool(T[i]["composites"]) for i in ix)
        out[f"{nm}_sharing_composite_or_museum_no_with_train"] = len(bad)
        if bad:
            fails.append(f"{nm} texts sharing a composite / museum number with train: {len(bad)} {bad[:5]}")
    dev_a = np.array([i for i in held if T[i]["split"] == "devA"])
    is_b = np.array([t["split"] == "devB" for t in T])
    keys_b = set().union(*(keys[i] for i in np.flatnonzero(is_b)))
    cross = len(near_dups(dev_a, is_b)) + sum(bool(keys[i] & keys_b) for i in dev_a)
    out["devA_texts_linked_to_devB"] = cross
    if cross:
        fails.append(f"devA texts that near-duplicate or share a composite with devB: {cross}")
    t2 = collections.Counter(k for i in held if T[i]["split"] == "test2" for k in keys[i])
    out["test2_composites_with_several_texts"] = sum(v > 1 for v in t2.values())
    if out["test2_composites_with_several_texts"]:
        fails.append("test2 has several texts of one composite / object")
    if not all(t["dev"] == (t["split"] in ("devA", "devB")) for t in T):
        fails.append("flag dev != devA + devB")
    rp = os.path.join(PROC, "reserved_ids.json")
    res = json.load(open(rp)) if os.path.exists(rp) else {"texts": [], "version": None}
    ok = sorted((r["id"], r["split"]) for r in res["texts"]) == sorted((T[i]["id"], T[i]["split"]) for i in held) \
        and res["version"] == version
    out["reserved_ids"] = {"texts": len(res["texts"]), "p_numbers": len(res.get("p_numbers", ())),
                           "q_numbers": len(res.get("q_numbers", ())), "museum_keys": len(res.get("museum_keys", ())),
                           "matches_held_out": ok}
    if not ok:
        fails.append("reserved_ids.json does not list exactly the held-out texts of this corpus version")
    n_all = sum(t["n"] for t in T)
    cols = ("train", "excluded", "devA", "devB", "test1", "test2")
    for name in ("stratum", "period_bin"):
        tab = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for t in T:
            x = tab[t[name]][t["split"]]
            x[0] += 1
            x[1] += t["n"]
        out["by_" + name] = {k: dict(v) for k, v in tab.items()}
        print(f"\n== texts / signs per split and {name}")
        print(f"{name:<10}" + "".join(f"{s:>18}" for s in cols) + f"{'all':>18}")
        for k in sorted(tab, key=lambda k: -sum(v[1] for v in tab[k].values())):
            tot = [sum(v[0] for v in tab[k].values()), sum(v[1] for v in tab[k].values())]
            print(f"{k:<10}" + "".join(f"{tab[k][s][0]:>8,}/{tab[k][s][1]:>9,}" if s in tab[k] else f"{'-':>18}"
                                       for s in cols) + f"{tot[0]:>8,}/{tot[1]:>9,}")
    for name, key in (("stratum", "unknown"), ("period_bin", "unknown")):
        ix = [t for t in T if t[name] == key]
        out[f"{name}_unknown"] = {"texts": len(ix), "texts_pct": round(100 * len(ix) / len(T), 2),
                                  "signs": sum(t["n"] for t in ix),
                                  "signs_pct": round(100 * sum(t["n"] for t in ix) / n_all, 2)}
    out["dialect"] = dict(collections.Counter(str(t["dialect"]) for t in T if t["lang"] == "akk"))
    print()


def compare(c, other_dir, out, fails):
    """Same texts and lines as the corpus in other_dir (v2); labels consistent with its split. Vocabularies and
    line arrays must be equal, except where the v3 encoding fixes recorded in corpus_stats.json
    (round3.fixes: the illegible sign -> <x>; numeral readings -> OSL's sign) apply: every differing position
    is classified, and any difference outside the fixes fails."""
    if not os.path.isabs(other_dir):
        other_dir = os.path.join(ROOT, other_dir)
    o = pickle.load(gzip.open(os.path.join(other_dir, "corpus.pkl.gz"), "rb"))
    st = json.load(open(os.path.join(PROC, "corpus_stats.json")))
    fx = st.get("round3", {}).get("fixes") or {}
    T, O = c["texts"], o["texts"]
    r = {"other_version": o["version"],
         "sign_vocab_equal": c["sign_vocab"] == o["sign_vocab"], "read_vocab_equal": c["read_vocab"] == o["read_vocab"],
         "sign_vocab_only_here": sorted(set(c["sign_vocab"]) - set(o["sign_vocab"])),
         "sign_vocab_only_other": sorted(set(o["sign_vocab"]) - set(c["sign_vocab"])),
         "read_vocab_only_here": sorted(set(c["read_vocab"]) - set(o["read_vocab"])),
         "read_vocab_only_other": sorted(set(o["read_vocab"]) - set(c["read_vocab"])),
         "planes_equal": all(c[k] == o[k] for k in ("joins", "dets", "brks", "langs")),
         "texts_equal": len(T) == len(O) and all(all(t[k] == u[k] for k in ("id", "src", "lang", "period", "genre", "cdli", "n"))
                                                 for t, u in zip(T, O)),
         "n_lines": [len(c["lines"]), len(o["lines"])]}
    r["line_shapes_equal"] = len(c["lines"]) == len(o["lines"]) and all(
        ti == tj and x.shape == y.shape for (ti, x), (tj, y) in zip(c["lines"], o["lines"]))
    r["label_transitions"] = dict(collections.Counter(f"{u['split']}->{t['split']}" for t, u in zip(T, O)))
    allowed = {"test->test1", "dev->devA", "dev->devB", "train->train", "train->test2", "train->excluded",
               "excluded->excluded"}
    if "composites" not in T[0]:
        allowed = {"test->test", "dev->dev", "train->train", "excluded->excluded"}
    r["labels_consistent"] = set(r["label_transitions"]) <= allowed
    sr, so = c["sign_reads"], o["sign_reads"]
    out["compare"] = r
    hard = ["planes_equal", "texts_equal", "line_shapes_equal", "labels_consistent"]
    if r["line_shapes_equal"]:
        A = np.concatenate([x for _, x in c["lines"]], 1)
        B = np.concatenate([x for _, x in o["lines"]], 1)
        lab = np.concatenate([np.full(x.shape[1], T[ti]["split"]) for ti, x in c["lines"]])
        r["join_det_brk_equal"] = bool((A[2:] == B[2:]).all())
        hard.append("join_det_brk_equal")
        # compare signs and readings as strings (the vocabulary indices shift when a sign leaves the vocabulary)
        sv = {s: i for i, s in enumerate(dict.fromkeys(c["sign_vocab"] + o["sign_vocab"]))}
        rv = {s: i for i, s in enumerate(dict.fromkeys(c["read_vocab"] + o["read_vocab"]))}
        sa, sb = np.array([sv[s] for s in c["sign_vocab"]])[A[0]], np.array([sv[s] for s in o["sign_vocab"]])[B[0]]
        ra, rb = np.array([rv[s] for s in c["read_vocab"]])[A[1]], np.array([rv[s] for s in o["read_vocab"]])[B[1]]
        diff = (sa != sb) | (ra != rb)
        r["positions"] = int(diff.size)
        r["positions_differing"] = int(diff.sum())
        isx = diff & (sb == sv.get(fx.get("x_oid"), -1)) & (sa == sv["<x>"]) & (ra == rv["<none>"])
        canon_s = np.array(sorted(sv[x] for x in set(fx.get("numeral_canon", {}).values()) if x in sv), np.int64)
        canon_r = np.array(sorted(rv[x] for x in fx.get("numeral_canon", {}) if x in rv), np.int64)
        # a numeral fix: now one of the canonical numeral signs with one of the unified readings; before, another
        # sign with the same reading, or no sign / reading at all (ASCII numerals of some Oracc exports)
        isn = diff & ~isx & np.isin(sa, canon_s) & np.isin(ra, canon_r) & (sa != sb) & \
            ((rb == ra) | (rb == rv["<none>"]))
        other = diff & ~isx & ~isn
        r["differing_x_to_special"] = int(isx.sum())
        r["differing_numeral_sign"] = int(isn.sum())
        r["differing_other"] = int(other.sum())
        r["differing_by_split"] = {k: {"x": int((isx & (lab == k)).sum()), "numeral": int((isn & (lab == k)).sum())}
                                   for k in sorted(set(lab.tolist()))}
        names_a = dict(zip(c["sign_vocab"], c["sign_names"]))
        names_b = dict(zip(o["sign_vocab"], o["sign_names"]))
        inv_s, inv_r = list(sv), list(rv)
        tr = collections.Counter((inv_s[b], inv_r[y], inv_s[a2], inv_r[x]) for a2, b, x, y in
                                 zip(sa[isn | other].tolist(), sb[isn | other].tolist(), ra[isn | other].tolist(),
                                     rb[isn | other].tolist()))
        r["differing_numeral_and_other_table"] = [
            [f"{names_b.get(b, b)} '{y}' -> {names_a.get(a2, a2)} '{x}'", n] for (b, y, a2, x), n in tr.most_common(40)]
        tot_tr = (lab == "train").sum()
        r["x_special_share_of_train_before_after_pct"] = [
            round(100 * float(((sb == sv["<x>"]) & (lab == "train")).sum() / tot_tr), 2),
            round(100 * float(((sa == sv["<x>"]) & (lab == "train")).sum() / tot_tr), 2)]
        r["differences_confined_to_fixes"] = r["differing_other"] == 0
        hard.append("differences_confined_to_fixes")
        if not fx:                                       # no fixes recorded: everything must be equal
            r["lines_equal"] = r["positions_differing"] == 0
            hard += ["lines_equal", "sign_vocab_equal", "read_vocab_equal"]
    r["sign_reads_changed_signs"] = sum(sr.get(k) != so.get(k) for k in set(sr) | set(so)) \
        if r["sign_vocab_equal"] else "n/a (vocabulary differs)"
    for k in hard:
        if not r[k]:
            fails.append(f"compare with {other_dir}: {k} is False")


if __name__ == "__main__":
    main()
