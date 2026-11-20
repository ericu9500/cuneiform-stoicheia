"""Unit tests of the damage sampler (cun.damage) on edge cases.

  python scripts/damage_test.py [--stats PATH]      (statistics: cun.damage.stats_path() by default)"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun import damage as D  # noqa: E402

CASES = {
    "single-line chunk": [9],
    "two signs": [2],
    "one sign": [1],
    "one-sign lines": [1] * 40,
    "256 signs in one line": [256],
    "ordinary administrative chunk": [5, 4, 6, 7, 3, 5, 6, 6, 4, 5, 8, 5, 4, 6, 7, 3, 5, 6, 6, 4] * 2,
    "long literary lines": [22, 19, 25, 21, 18, 24, 20, 23, 22, 21, 20, 21],
    "mixed, with one-sign lines": [1, 12, 1, 3, 30, 1, 2, 7],
}


def check(mask, abr, n):
    assert mask.dtype == bool and abr.dtype == bool, "boolean arrays"
    assert mask.shape == (n,) and abr.shape == (n,), "one entry per sign"
    assert not (abr & ~mask).any(), "abraded signs are masked signs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=None)
    a = ap.parse_args()
    stats = D.load_stats(a.stats)
    strata = [None] + stats["strata"][1:] + ["no such stratum"]
    n_checks = 0
    for name, ll in CASES.items():
        n = sum(ll)
        for stratum in strata:
            for seed in range(40):
                kw_list = [{}, {"allow_clean": True}, {"text_start": True}, {"text_end": True},
                           {"text_start": True, "text_end": True}]
                for kw in kw_list:
                    m, ab = D.sample_mask(np.random.default_rng(seed), ll, stats, stratum=stratum, **kw)
                    check(m, ab, n)
                    if not kw.get("allow_clean"):
                        assert m.any(), f"{name}: empty mask without allow_clean"
                    m2, ab2 = D.sample_mask(np.random.default_rng(seed), ll, stats, stratum=stratum, **kw)
                    assert (m == m2).all() and (ab == ab2).all(), f"{name}: not deterministic given rng"
                    n_checks += 1
                for t in (0.05, 0.3, 0.8, 1.0):
                    m, ab = D.sample_mask(np.random.default_rng(seed), ll, stats, stratum=stratum, target_rate=t)
                    check(m, ab, n)
                    want = max(1, int(round(t * n)))
                    tol = max(1, int(0.01 * n))
                    assert abs(int(m.sum()) - want) <= tol, f"{name}: target {t}: {int(m.sum())} masked, wanted {want}"
                    m2, _ = D.sample_mask(np.random.default_rng(seed), ll, stats, stratum=stratum, target_rate=t)
                    assert (m == m2).all(), f"{name}: not deterministic given rng (target_rate)"
                    n_checks += 1
                for mx in (1, 5, 20, 1000):
                    for kind in (None,) + D.GAP_TYPES:
                        rng = np.random.default_rng(seed)
                        m, info = D.sample_gap(rng, ll, stats, mx, stratum=stratum, kind=kind)
                        if m is None:               # no admissible placement (e.g. a medial run on one-sign lines)
                            assert kind in ("hole", "medial"), f"{name}: no gap of type {kind}"
                            continue
                        assert m.dtype == bool and m.shape == (n,)
                        assert 1 <= m.sum() <= mx, f"{name}: gap of {m.sum()} signs, max {mx}"
                        lines = info["lines"]
                        assert (np.diff(lines) == 1).all(), "consecutive lines"
                        off = np.r_[0, np.cumsum(ll)]
                        m3 = np.zeros(n, bool)
                        for li, s, e in zip(lines, info["start"], info["end"]):
                            assert 0 <= s < e <= ll[li], "run inside its line"
                            m3[off[li] + s:off[li] + e] = True
                        assert (m3 == m).all(), "info describes the mask: one run per line"
                        if kind is not None:           # a lost line longer than max_signs becomes an edge loss
                            assert info["type"] == kind or (kind == "lines" and info["type"] == "left")
                        if info["type"] in ("hole", "medial"):
                            assert all(s >= 1 and e <= ll[li] - 1 for li, s, e in zip(lines, info["start"], info["end"])), "medial"
                        n_checks += 1
    # eligibility: gaps only on eligible signs
    ll = CASES["ordinary administrative chunk"]
    el = np.ones(sum(ll), bool)
    el[:60] = False
    rng = np.random.default_rng(0)
    got = 0
    for _ in range(300):
        m, info = D.sample_gap(rng, ll, stats, 10, eligible=el)
        if m is not None:
            assert el[m].all(), "gap on ineligible signs"
            got += 1
    assert got > 200, got
    # empty chunk
    m, ab = D.sample_mask(np.random.default_rng(0), [], stats)
    assert len(m) == 0 and len(ab) == 0
    assert D.sample_gap(np.random.default_rng(0), [], stats, 5) == (None, None)
    # the requested rate is met on average and line by line statistics react to it
    rng = np.random.default_rng(1)
    ll = CASES["ordinary administrative chunk"]
    for t in (0.05, 0.8):
        r = [D.sample_mask(rng, ll, stats, target_rate=t)[0].mean() for _ in range(300)]
        assert abs(np.mean(r) - t) < 0.01, (t, np.mean(r))
    print(f"ok: {n_checks} checks on {len(CASES)} line structures x {len(strata)} strata")


if __name__ == "__main__":
    main()
