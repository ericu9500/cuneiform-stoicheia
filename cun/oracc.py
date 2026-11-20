"""Read Oracc project zips (JSON export) into sign-level lines.

Each sign position is a dict:
  oid      OSL sign id ('' if Oracc gives none); numerals keep their own oid (e.g. 5(DIŠ))
  sign     OSL sign name (gdl_sign)
  read     reading in context: value ('du₃'), logogram name ('BAL' for role=logo), numeral form ('5(diš)')
  det      '' | 'pre' | 'post' (determinative / phonetic complement position) + ':' + kind
  join     separator after this sign inside a word ('-', '.', ':', '+', '') ; word end is marked by 'w'
  brk      '' | 'damaged' | 'missing'   (missing with a reading = restored by the editor)
  lang     word language (sux, akk, ...; Oracc's 'akk-x-oldbab' -> 'akk')
  unk      'x' for an unreadable sign, 'gap' for an ellipsis (lacuna of unknown length)
"""
from __future__ import annotations

import collections
import json
import zipfile

LEAF_KEYS = ("v", "s", "n", "c", "q")


def _lang(l):
    if not l:
        return ""
    return l.split("-")[0]


STATS = {"numeral_no_oid": 0, "numeral_split": 0, "mods_no_oid": 0, "empty_word": 0}


def _leaf(g, det, lang):
    read = g.get("v") or g.get("s") or g.get("sexified") or g.get("form") or g.get("q") or g.get("c") or ""
    if not (g.get("v") or g.get("s")) and g.get("mods") and isinstance(g["mods"][0], dict) and g["mods"][0].get("b"):
        read = g["mods"][0]["b"]                     # modified grapheme: its base (ti in ti@t)
    if "q" in g and g.get("qualified"):
        q = g["qualified"][0]
        read = q.get("v") or q.get("s") or read
    return {"oid": g.get("oid", ""), "sign": g.get("gdl_sign", ""), "read": read, "det": det,
            "join": g.get("delim", ""), "brk": g.get("break", ""), "lang": lang, "unk": ""}


def flatten_gdl(gs, lang, det="", out=None):
    out = [] if out is None else out
    for g in gs:
        if not isinstance(g, dict):
            continue
        if "x" in g:
            out.append({"oid": "", "sign": "", "read": "", "det": det, "join": g.get("delim", ""),
                        "brk": g.get("break", ""), "lang": lang, "unk": "gap" if g["x"] == "ellipsis" else "x"})
            continue
        if "det" in g and "seq" in g:
            flatten_gdl(g["seq"], lang, f"{g.get('pos', '')}:{g['det']}", out)
            if out and g.get("delim"):
                out[-1]["join"] = g["delim"]
            continue
        if "oid" in g and any(k in g for k in LEAF_KEYS + ("form",)):
            out.append(_leaf(g, det, lang))
            continue
        if "group" in g:
            n0 = len(out)
            flatten_gdl([h for h in g["group"] if isinstance(h, dict) and "t" not in h], lang, det, out)
            if len(out) > n0 and g.get("delim"):
                out[-1]["join"] = g["delim"]
            continue
        if any(k in g for k in ("v", "s")):            # grapheme without an oid: kept, mapped later
            out.append(_leaf(g, det, lang))
            continue
        if g.get("n") == "n" and (g.get("sexified") or g.get("form")):   # numeral without an oid: 5(diš)
            STATS["numeral_no_oid"] += 1
            parts = (g.get("sexified") or "").split()
            if len(parts) > 1:                         # '15' sexified '1(u) 5(diš)': one sign per part
                STATS["numeral_split"] += 1
                for k, part in enumerate(parts):
                    leaf = _leaf({**g, "sexified": part}, det, lang)
                    if k < len(parts) - 1:
                        leaf["join"] = ""
                    out.append(leaf)
            else:
                out.append(_leaf(g, det, lang))
            continue
        if g.get("mods") and "form" in g:              # modified grapheme without an oid (ti@t)
            STATS["mods_no_oid"] += 1
            out.append(_leaf(g, det, lang))
            continue
        if "seq" in g and "gloss" not in g:
            flatten_gdl(g["seq"], lang, det, out)
    return out


def read_text(d, tags=None):
    """Return list of lines; each line = dict(label, signs=[...]) with word ends marked join='w'.
    `tags` (a Counter), if given, receives the signs per FULL Oracc language tag ('akk-x-stdbab'); the
    sign dicts keep the truncated tag."""
    lines = []
    cur = None

    def walk(x):
        nonlocal cur
        if isinstance(x, dict):
            if x.get("node") == "d" and x.get("type") == "line-start":
                cur = {"label": x.get("label", ""), "signs": []}
                lines.append(cur)
            if "f" in x and isinstance(x["f"], dict) and "gdl" in x["f"]:
                if cur is None:
                    cur = {"label": "", "signs": []}
                    lines.append(cur)
                ws = flatten_gdl(x["f"]["gdl"], _lang(x["f"].get("lang")))
                if not ws:                             # a word that yields no sign: keep a placeholder
                    STATS["empty_word"] += 1
                    ws = [{"oid": "", "sign": "", "read": "", "det": "", "join": "", "brk": "",
                           "lang": _lang(x["f"].get("lang")), "unk": "x"}]
                if ws:
                    ws[-1]["join"] = "w"
                    cur["signs"].extend(ws)
                    if tags is not None and x["f"].get("lang"):
                        tags[x["f"]["lang"]] += len(ws)
            for k, v in x.items():
                if k != "f":
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(d.get("cdl", []))
    return [l for l in lines if l["signs"]]


def iter_project(zpath):
    """Yield (textid, project, catalogue entry, lines) for every corpus text in an Oracc project zip.
    The catalogue entry (a copy) also carries '_lang_tags': signs per full Oracc language tag."""
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        return
    cats = {}
    for n in zf.namelist():
        if n.endswith("/catalogue.json"):
            try:
                cats.update(json.loads(zf.read(n)).get("members", {}))
            except Exception:
                pass
    for n in zf.namelist():
        if "/corpusjson/" not in n or not n.endswith(".json"):
            continue
        try:
            d = json.loads(zf.read(n))
        except Exception:
            continue
        tid = d.get("textid") or n.rsplit("/", 1)[-1][:-5]
        tags = collections.Counter()
        lines = read_text(d, tags)
        yield tid, d.get("project", ""), {**cats.get(tid, {}), "_lang_tags": dict(tags)}, lines
