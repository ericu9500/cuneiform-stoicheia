"""Noise audit: shared helpers (train cache, OSL value metadata, reading-difference classes)."""
from __future__ import annotations

import collections
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from cun.signmap import norm_read  # noqa: E402

OUT = os.path.join(ROOT, "results/noise_audit")
# corpus build: CUN_PROC (default here: data/proc_v3, corpus v3); only its split == 'train' texts are ever read
PROC = os.environ.get("CUN_PROC") or "data/proc_v3"
PROC = PROC if os.path.isabs(PROC) else os.path.join(ROOT, PROC)
VERSION = json.load(open(os.path.join(PROC, "corpus_stats.json")))["version"]
FIRST_REAL_SIGN = 6
IDX = str.maketrans("", "", "₀₁₂₃₄₅₆₇₈₉ₓ⁺⁻")
NUM = re.compile(r"^(\d|n\(|\d+/\d+)")
VOI = str.maketrans("pdtṭkqsṣe", "bdddgggzzi".replace("d", "d")[:0] + "bddd" + "gg" + "zz" + "i") if False else str.maketrans(
    {"p": "b", "t": "d", "ṭ": "d", "k": "g", "q": "g", "s": "z", "ṣ": "z", "e": "i"})
DET_READS = {"m", "d", "f", "diš", "dingir", "digir", "munus", "mi₂", "an", "1(diš)", "1"}


def period_norm(p):
    if p is None:
        return "(none)"
    p = re.sub(r"\s*\(.*\)", "", p).strip().rstrip("?").strip()
    p = p.replace("Lagaš II", "Lagash II")
    if p.startswith("ED ") or p == "Early Dynastic":
        return "Early Dynastic"
    return p or "(none)"


def load_train():
    """-> planes (5 x N), text index per sign, meta dict; plus per-text arrays src / project / lang / period."""
    m = json.load(open(os.path.join(OUT, "cache/meta.json")))
    assert m["version"] == VERSION
    z = np.load(os.path.join(OUT, "cache/train.npz"))
    tx = m["texts"]
    m["t_src"] = np.array([t["src"].split(":")[0] for t in tx])
    m["t_proj"] = np.array([t["src"] for t in tx])
    m["t_lang"] = np.array([t["lang"] for t in tx])
    m["t_period"] = np.array([period_norm(t["period"]) for t in tx])
    return z["planes"], z["text"], z["line_start"], m


def parse_osl(path=os.path.join(ROOT, "data/raw/osl.asl")):
    """oid -> {"name", "parent" (oid of the @sign for a @form, else None), "values": {norm value: set(flags)}}
    flags: 'dep' (@v-), 'unc' (value?), 'x' (ₓ index), 'lang:akk' (%akk ...); also the Attinger @sys mappings
    [(oid, from, to)] ('a => b': the value a is written b in the Attinger-Sallaberger system)."""
    signs, sysmap = {}, []
    cur = parent = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.startswith("@"):
            continue
        tag, _, rest = line.partition("\t") if "\t" in line else line.partition(" ")
        rest = rest.strip()
        if tag in ("@sign", "@sign-"):
            pend = ("sign", rest, tag.endswith("-"))
            cur = None
            parent = None
        elif tag in ("@form", "@form-", "@form+"):
            pend = ("form", rest, tag.endswith("-"))
            cur = None
        elif tag == "@oid":
            cur = rest
            signs[cur] = {"name": pend[1], "parent": parent if pend[0] == "form" else None, "dep": pend[2], "values": {}}
            if pend[0] == "sign":
                parent = cur
        elif tag in ("@v", "@v-") and cur:
            fl = set()
            if tag == "@v-":
                fl.add("dep")
            parts = rest.split()
            if not parts:
                continue
            if parts[0].startswith("%"):
                fl.add("lang:" + parts[0][1:].split("/")[0])
            v = parts[-1]
            if v.endswith("?"):
                fl.add("unc")
            v = v.strip("[]?")
            if "ₓ" in v:
                fl.add("x")
            signs[cur]["values"].setdefault(norm_read(v), set()).update(fl)
        elif tag == "@sys" and cur and rest.startswith("Attinger") and "=>" in rest:
            a, b = rest[len("Attinger"):].split("=>")
            sysmap.append((cur, norm_read(a.strip()), norm_read(b.strip())))
        elif tag == "@@":
            cur = parent
    return signs, sysmap


def letters(r):
    return r.translate(IDX)


def _eng(s):
    """skeleton for the nasal-velar convention: ŋ is written g (norm_read of ŋ), n, ng or m by other conventions;
    double consonants collapsed (inana / inanna, sagga / sanga)."""
    s = s.replace("ng", "g")
    s = re.sub(r"(.)\1", r"\1", s)
    return s


def read_class(a, b):
    """Formal class of two different readings of the SAME sign (strings after norm_read):
    numeral   both are number notations
    num/word  one is a number notation, the other a value (diš / 1(diš), u / 1(u))
    index     same letters, other index (šar₉ / šarₓ)
    eng       same after the ŋ / geminate skeleton (dingir / digir, sanga / sagga, pisan / pisag, inana / inanna)
    auslaut   long vs short form: one is the other plus one final consonant (gud / gu₄, kug / ku₃) or vowel (bala / bal)
    voicing   same after merging b/p, d/t/ṭ, g/k/q, z/s/ṣ, e/i
    other"""
    na, nb = bool(NUM.match(a)), bool(NUM.match(b))
    if na and nb:
        return "numeral"
    if na or nb:
        return "num/word"
    x, y = letters(a), letters(b)
    if x == y:
        return "index"
    s, l = sorted((x, y), key=len)
    if l.startswith(s) and len(l) - len(s) == 1 and len(s) >= 1:
        return "auslaut"
    ex, ey = _eng(x), _eng(y)
    if ex == ey:
        return "eng"
    if len(ex) == len(ey):
        d = [(p, q) for p, q in zip(ex, ey) if p != q]
        if len(d) == 1 and set(d[0]) in ({"n", "g"}, {"m", "g"}):
            return "eng"
    if x.translate(VOI) == y.translate(VOI):
        return "voicing"
    return "other"
