"""CDLI ASCII ATF -> sign-level lines in the same format as cun.oracc (one dict per sign position).

Uses the UR study's text/line parser for structure (language lines, skipped lines, sections) and
re-reads each line's raw content character by character to get sign-level damage:
  brk = 'missing' inside [...] (editor's restoration), 'damaged' for '#', '' otherwise.
Compounds |A.B| and qualified readings val(SIGN) are single signs; {det} is a determinative
(det='pre' or 'post' by position in the word); '{+...}' is a phonetic complement.
"""
from __future__ import annotations

import re

SEP = "-.:+"
HAND = {  # CDLI ASCII spellings that Oracc writes differently (checked against OSL / Oracc usage)
    "saga": "sa₆", "szunigin2": "šu-nigin₂", "muru13": "muru₁₃",
}


def tokenize_line(content: str):
    """Return list of (sign_text, det, join, brk, unk) for one ATF line content (inline comments removed)."""
    content = re.sub(r"\(\$.*?\$\)", " ", content)
    content = re.sub(r"\{\{.*?\}\}", "", content)                 # glosses {{...}}
    content = re.sub(r"<\(.*?\)>", "", content)                     # MIN<(lugal)>: gloss of a ditto sign
    content = re.sub(r"(?<!<)<(?!<)[^<>]*>(?!>)", "", content)        # <supplied> (keep <<...>>: on the tablet)
    content = re.sub(r"\*\([^)]*\)", " ", content)                  # lexical bullets *(disz)
    content = re.sub(r"(?<!\S)[=~\"&;](?!\S)", " ", content)         # lexical column markup
    content = re.sub(r",?!(?=[a-z])[a-z]+", "", content)              # lexical tags !sg ,!eq
    content = re.sub(r"(?<![|A-Za-z0-9])%[a-z]+\d*", " ", content)    # language switches %a %sux %e (not |a%b|)
    content = content.replace("–", "-")                               # en-dash separators
    content = content.replace("...", "\x00")                          # protect ellipses (lacunae)
    content = re.sub(r"[-.:+](?=\s|$)|(?:(?<=\s)|^)[-.:+]", "", content)   # separators left dangling
    content = content.replace("\x00", "...")
    out = []
    sq = 0
    i, n = 0, len(content)
    buf, buf_brk = [], set()
    det_type = "semantic"
    in_det = False
    in_comp = False
    in_logo = False
    paren = 0
    word_has_sign = False

    def flush(join):
        nonlocal buf, buf_brk
        s = "".join(buf).strip()
        if not s and join and out and out[-1][2] == "":
            out[-1][2] = join                       # separator right after a determinative or '...'
        if s:
            brk = "missing" if "missing" in buf_brk else ("damaged" if "damaged" in buf_brk else "")
            unk = "gap" if s == "..." else ("x" if s in ("x", "X") or s.startswith(("x(", "X(")) else "")
            out.append([s, det_kind(), join, brk, unk])
            if not in_det:
                mark_sign()
        buf, buf_brk = [], set()

    def mark_sign():
        nonlocal word_has_sign
        word_has_sign = True

    def det_kind():
        if not in_det:
            return ""
        return ("pre" if not word_has_sign else "post") + ":" + det_type

    while i < n:
        c = content[i]
        if content.startswith("...", i) and not in_comp:
            flush("")
            buf = list("...")
            if sq:
                buf_brk.add("missing")
            flush("")
            i += 3
            continue
        if c == "[":
            sq += 1; i += 1; continue
        if c == "]":
            sq = max(0, sq - 1); i += 1; continue
        if content.startswith("<<", i) or content.startswith(">>", i):
            i += 2; continue
        if c in "<>⸢⸣*?":
            i += 1; continue
        if c == "#":
            buf_brk.add("damaged"); i += 1; continue
        if c == "!":
            m = re.match(r"!\(([^()]*)\)", content[i:])
            i += m.end() if m else 1
            continue
        if c == "|":
            in_comp = not in_comp
            buf.append(c); i += 1; continue
        if c == "(":
            paren += 1
        if c == ")":
            paren = max(0, paren - 1)
        if c.isspace() and not in_comp and not paren:
            flush("w")
            word_has_sign = False
            in_det = False
            i += 1
            continue
        if c == "{" and not in_comp:
            flush("")
            in_det = True
            det_type = "semantic"
            if content.startswith("{+", i):
                det_type = "phonetic"
                i += 1
            i += 1
            continue
        if c == "}" and not in_comp:
            flush("")
            in_det = False
            i += 1
            continue
        if c in SEP and not in_comp and not paren:
            flush("." if (c == "-" and in_logo) else c)
            i += 1
            continue
        if c == "_":
            in_logo = not in_logo
            i += 1; continue
        if sq:
            buf_brk.add("missing")
        buf.append(c)
        i += 1
    flush("w")
    if out:
        out[-1][2] = "w"
    # determinative position: 'post' if the sign before it touches it (no separator: uri5{ki}),
    # 'pre' if it opens the word or follows a separator ({d}szu, szu-{d}suen, ur-{gesz}gigir)
    for k, o in enumerate(out):
        if not o[1]:
            continue
        j = k - 1
        while j >= 0 and out[j][1] and out[j][2] == "":       # skip over a run of determinatives
            j -= 1
        post = j >= 0 and not out[j][1] and out[j][2] == "" and not out[j][4]
        o[1] = ("post" if post else "pre") + ":" + o[1].split(":", 1)[-1]
    # an ellipsis or x followed by nothing inside the word ends the word
    return out


def convert_line(content, smap, lang):
    """-> list of sign dicts in the cun.oracc format."""
    res = []
    for s, det, join, brk, unk in tokenize_line(content):
        if unk:
            res.append({"oid": "", "sign": "", "read": "", "det": det, "join": join, "brk": brk, "lang": lang, "unk": unk})
            continue
        r = HAND.get(s, None)
        if r and "-" in r:                                  # one ASCII spelling = several Oracc signs
            parts = r.split("-")
            for k, p in enumerate(parts):
                oid = smap._get(p, lang)
                res.append({"oid": oid or "", "sign": smap.names.get(oid, ""), "read": p, "det": det,
                            "join": join if k == len(parts) - 1 else "-", "brk": brk, "lang": lang, "unk": ""})
            continue
        if r:
            oid = smap._get(r, lang)
            read = r
        else:
            oid, read = smap.lookup(s, lang)
        if not oid and s.startswith("|") and s.endswith("|") and "." in s:
            parts = s[1:-1].split(".")                   # compound unknown to OSL: |KI.MIN| -> KI . MIN
            got = [smap.lookup(p, lang) for p in parts]
            if all(o for o, _ in got):
                for k, (o, rd) in enumerate(got):
                    res.append({"oid": o, "sign": smap.names.get(o, ""), "read": rd, "det": det,
                                "join": join if k == len(parts) - 1 else ".", "brk": brk, "lang": lang, "unk": ""})
                continue
        res.append({"oid": oid or "", "sign": smap.names.get(oid, "") if oid else "", "read": read, "det": det,
                    "join": join, "brk": brk, "lang": lang, "unk": ""})
    return res
