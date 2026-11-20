"""Composite and object identity of corpus texts (corpus v3: composite-level deduplication of held-out sets).

A text's `composites` are Q-numbers (Oracc / CDLI composite ids); texts that share one are exemplars of the
same composition. Sources of the links:
  * the text is itself a composite (Oracc id 'Q......');
  * Oracc catalogues (EVERY catalogue.json of every project zip, incl. umbrella catalogues such as armep and
    epsd2), witness -> composite: fields P2Q of the entry of a P- / X-text;
  * Oracc catalogues, composite -> witnesses: P-numbers in the fields Q2P of the entry of a Q-text;
  * Oracc catalogues, composite -> witnesses by museum number: the field 'exemplars' of a Q-text lists
    museum numbers, matched against the museum numbers of P-texts;
  * the CDLI catalogue: 'composite_id' of UR/data/proc/texts_meta.jsonl.gz.
A text's `museum` keys are its normalised museum numbers (Oracc catalogue fields MUSEUM); texts that share
a key are the same object (joins, double editions). X-numbers are local to a project, so an X-text only
uses the catalogue of its own project. The Oracc JSON exports contain no score files (only catalogues link
witnesses to composites).
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import os
import re
import zipfile

P2Q = ("related_comp_id", "qcat_2__id_composite", "Q_no", "composites", "related_compositions", "q_number",
       "composite_no", "composite_id", "id_composite")
Q2P = ("cdli_id", "cdli_composite_id", "cdli_id_numbers", "sources", "related_manu")     # + ex<N>id
MUSEUM = ("museum_no", "cdli_museum_no")
KEEP = set(P2Q + Q2P + MUSEUM + ("exemplars", "genre", "subgenre", "period", "accession_no", "excavation_no"))
RQ = re.compile(r"Q\d{6}")
RP = re.compile(r"P\d{6}")
MUS_MAX = 10          # a museum key carried by more texts than this is a collection or lot, not an object


def load_catalogues(oracc_dir):
    """{text id: [entry, ...]} over all Oracc catalogues; an entry keeps the fields used here and '_proj'."""
    out = collections.defaultdict(list)
    for z in sorted(glob.glob(os.path.join(oracc_dir, "*.zip"))):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:
            continue
        for n in zf.namelist():
            if not n.endswith("/catalogue.json"):
                continue
            try:
                members = json.loads(zf.read(n)).get("members", {})
            except Exception:
                continue
            proj = n[:-len("/catalogue.json")]
            for tid, c in members.items():
                e = {k: v for k, v in c.items() if (k in KEEP or re.fullmatch(r"ex\d+id", k)) and v}
                e["_proj"] = proj
                out[tid].append(e)
    return out


def load_cdli_composites(meta_path):
    out = {}
    with gzip.open(meta_path, "rt") as fh:
        for l in fh:
            d = json.loads(l)
            qs = RQ.findall(d.get("composite_id") or "")
            if qs:
                out[d["pid"]] = set(qs)
    return out


def museum_keys(value):
    """Normalised museum numbers of a catalogue string: 'BM 090184 (1979-12-20, 0100) + BM 90185' ->
    {'BM90184', 'BM90185'}. A key needs a collection siglum and a number (so 'BM', 'IM ---', 'unknown' and a
    bare number give nothing)."""
    out = set()
    if not isinstance(value, str):
        value = " ; ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
    value = re.sub(r"\([^)]*\)?", " ", value)
    for part in re.split(r"\+|;|&|=|,", value):
        k = re.sub(r"[^A-Z0-9]+", " ", part.upper()).strip()
        k = re.sub(r"\b0+(?=\d)", "", k).replace(" ", "")
        if re.match(r"[A-Z]+", k) and len(re.findall(r"\d", k)) >= 2:
            out.add(k)
    return out


def _s(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


class Identity:
    """Composite and museum identity for corpus texts (dicts with 'id', 'src', 'cdli')."""

    def __init__(self, oracc_dir, cdli_meta_path):
        self.cats = load_catalogues(oracc_dir)
        self.cdli_comp = load_cdli_composites(cdli_meta_path)
        self.q_of_p = collections.defaultdict(set)              # P-number -> Q-numbers (from Q entries)
        self.q_exemplar_keys = collections.defaultdict(set)     # museum key -> Q-numbers (field 'exemplars')
        self.n_links = collections.Counter()
        for tid, entries in self.cats.items():
            if not tid.startswith("Q"):
                continue
            for e in entries:
                for k, v in e.items():
                    if k in Q2P or re.fullmatch(r"ex\d+id", k):
                        for p in RP.findall(_s(v)):
                            self.q_of_p[p].add(tid)
                            self.n_links["Q->P:" + (k if k in Q2P else "exNid")] += 1
                for mk in museum_keys(e.get("exemplars", "")):
                    self.q_exemplar_keys[mk].add(tid)

    def entries(self, t):
        """Catalogue entries that describe text t (its id and its CDLI P-number; X-ids: own project only)."""
        proj = t["src"].partition(":")[2]
        out = []
        for key in dict.fromkeys(k for k in (t["id"], t.get("cdli")) if k):
            for e in self.cats.get(key, ()):
                if key.startswith("X") and e["_proj"] != proj:
                    continue
                out.append((key, e))
        return out

    def museum(self, t):
        keys = set()
        for _, e in self.entries(t):
            for f in MUSEUM:
                if e.get(f):
                    keys |= museum_keys(e[f])
        return keys

    def museum_raw(self, t):
        return sorted({_s(e[f]) for _, e in self.entries(t) for f in MUSEUM if e.get(f)})

    def composites(self, t, mus=None):
        """(set of Q-numbers, set of link kinds)."""
        qs, how = set(), set()
        if t["id"].startswith("Q"):
            qs.add(t["id"])
            how.add("self")
        for key, e in self.entries(t):
            if key.startswith("Q"):
                continue
            for f in P2Q:
                found = RQ.findall(_s(e[f])) if e.get(f) else ()
                if found:
                    qs.update(found)
                    how.add("oracc:P->Q")
        for key in (t["id"], t.get("cdli")):
            if key and key in self.q_of_p:
                qs |= self.q_of_p[key]
                how.add("oracc:Q->P")
            if key and key in self.cdli_comp:
                qs |= self.cdli_comp[key]
                how.add("cdli")
        for mk in (self.museum(t) if mus is None else mus):
            if mk in self.q_exemplar_keys:
                qs |= self.q_exemplar_keys[mk]
                how.add("oracc:exemplars")
        return qs, how
