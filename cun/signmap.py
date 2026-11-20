"""Map CDLI ASCII-ATF readings to OSL sign ids.

Sources, in order: (1) reading -> oid pairs observed in Oracc corpora (most frequent oid per reading),
(2) OSL @v / @sign / @form entries. CDLI ASCII is converted to Oracc conventions first.
"""
from __future__ import annotations

import collections
import re

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def norm_read(r: str) -> str:
    """Canonical reading for vocabularies and scoring: lower case (Oracc writes logograms and sign-name
    readings in capitals, CDLI mostly not), g for ŋ (Oracc writes ŋ, CDLI mostly g), h for ḫ (older Oracc
    projects write ḫ)."""
    return r.lower().replace("ŋ", "g").replace("ḫ", "h") if r else r


SIGN_LISTS = re.compile(r"^(LAK|REC|ZATU|ELLES|KWU|RSP|SLA|BAU|MZL|ABZL|HZL|LAGAB|N)(\d+)")


def ascii_to_oracc(r: str) -> str:
    """lugal -> lugal, sze3 -> še₃, kux -> kuₓ, s, -> ṣ, t, -> ṭ, j -> ŋ ; h stays h (current Oracc).
    Compounds |HIxASZ2| -> |HI×AŠ₂| (component by component); sign-list numbers (LAK384) keep their digits;
    modifiers after the index stay (gesz2@c -> geš₂@c)."""
    if len(r) > 2 and r.startswith("|") and r.endswith("|"):
        parts = re.split(r"([.×&%+()]|(?<=[A-Z0-9₀-₉])x(?=[A-Z|(]))", r[1:-1])
        out = []
        for p in parts:
            if p is None or p == "":
                continue
            out.append("×" if p == "x" else (p if p in ".×&%+()" else ascii_to_oracc(p)))
        return "|" + "".join(out) + "|"
    if SIGN_LISTS.match(r) and r.isupper():
        return r
    mod = ""
    mm = re.match(r"^(.*?)((?:[@~][a-z0-9]+)+)$", r)
    if mm and mm.group(1):
        r, mod = mm.group(1), mm.group(2)
    r = r.replace("sz", "š").replace("SZ", "Š").replace("s,", "ṣ").replace("S,", "Ṣ")
    r = r.replace("t,", "ṭ").replace("T,", "Ṭ").replace("j", "ŋ").replace("J", "Ŋ")
    r = r.replace("'", "ʾ") if r not in ("'",) else r
    m = re.match(r"^(.*?[^\dx])(\d+|x)$", r)
    if m:
        idx = m.group(2)
        r = m.group(1) + ("ₓ" if idx == "x" else idx.translate(SUB))
    return r + mod


def numeral_to_oracc(r: str) -> str:
    """5(disz) -> 5(diš); 1/3(disz) -> 1/3(diš)."""
    m = re.match(r"^(\d+(?:/\d+)?|n)\((.+)\)$", r)
    if not m:
        return r
    return f"{m.group(1)}({ascii_to_oracc(m.group(2))})"


NUMERAL = re.compile(r"^\d+(?:/\d+)?\([^ ]+\)$")          # one numeral sign: 5(diš), 1/2(diš), 1(aš@c)


def parse_osl(path):
    """reading/name (lowercased for values) -> set of oids; oid -> sign name."""
    r2o = collections.defaultdict(set)
    name = {}
    oid = None
    sname = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.startswith("@"):
            continue
        tag, _, rest = line.partition("\t") if "\t" in line else line.partition(" ")
        rest = rest.strip()
        if tag in ("@sign", "@form"):
            sname = rest
            oid = None
        elif tag == "@oid" and sname:
            oid = rest
            name.setdefault(oid, sname)
            r2o[sname].add(oid)
        elif tag == "@v" and oid:
            v = rest.split()[-1].strip("[]?") if rest else ""
            if v and not v.startswith("%"):
                r2o[v].add(oid)
        elif tag == "@end":
            pass
    return r2o, name


class SignMap:
    def __init__(self, oracc_pairs: dict, osl_path: str, lang_pairs: dict | None = None):
        """oracc_pairs: reading -> Counter(oid) from Oracc corpora; lang_pairs: (lang, reading) -> Counter(oid),
        consulted first when a language is given (limmu₂, uš₂ differ between Sumerian and Akkadian use)."""
        self.osl, self.names = parse_osl(osl_path)

        def top(c):                                     # only oids that the OSL sign list knows
            c = collections.Counter({o: n for o, n in c.items() if o in self.names})
            return c.most_common(1)[0][0] if c else None
        self.oracc = {r: o for r, o in ((r, top(c)) for r, c in oracc_pairs.items()) if o}
        self.oracc_lang = {k: o for k, o in ((k, top(c)) for k, c in (lang_pairs or {}).items()
                                             if sum(c.values()) >= 5) if o}
        self.miss = collections.Counter()
        self.miss_oracc = collections.Counter()
        # Corpus v3 (not used by lookup / lookup_oracc, so corpus v2 is unchanged): numeral readings for which
        # Oracc's own sign ids are split (the most frequent has < 90% of >= 20 tokens; e.g. 1/2(diš): DIŠ in
        # some projects, MAŠ in others) take the sign that OSL lists for that value, in both sources.
        self.oid_x = next((o for o, n in self.names.items() if n == "X"), None)      # the illegible sign
        agg = collections.defaultdict(collections.Counter)
        for r, c in oracc_pairs.items():
            if NUMERAL.match(r):
                agg[numeral_to_oracc(r).lower()].update({o: n for o, n in c.items() if o in self.names})
        self.num_canon, self.num_split = {}, {}
        for r, c in agg.items():
            tot = sum(c.values())
            if tot >= 20 and c.most_common(1)[0][1] < 0.9 * tot:
                o = self.osl.get(r)
                self.num_split[r] = {"oracc": [(self.names[x], n) for x, n in c.most_common(3)],
                                     "osl": sorted(self.names[x] for x in o) if o else []}
                if o and len(o) == 1:
                    self.num_canon[r] = next(iter(o))

    def numeral_v3(self, reading: str):
        """(canonical oid, reading in Oracc form) for a numeral reading whose sign v3 unifies, else (None, None).
        Accepts Oracc readings and ASCII forms ('1/2(disz)', which some Oracc exports give)."""
        if not reading or not NUMERAL.match(reading):
            return None, None
        r = numeral_to_oracc(reading).lower()
        return (self.num_canon[r], r) if r in self.num_canon else (None, None)

    def lookup(self, reading: str, lang: str | None = None):
        """CDLI ASCII reading (one sign, determinative braces removed) -> (oid, normalised reading)."""
        r = reading
        q = re.match(r"^(.+?)\((.+)\)$", r)
        if q and not re.match(r"^(\d+(?:/\d+)?|n)$", q.group(1)):      # qualified: val(SIGN)
            val, sgn = ascii_to_oracc(q.group(1)), ascii_to_oracc(q.group(2))
            o = self._get(sgn) or self._get(sgn.upper())
            return (o, val) if o else (None, val)
        if re.match(r"^(\d+(?:/\d+)?|n)\(", r):
            n = numeral_to_oracc(r)
            return self._get(n, lang) or self._get(n.upper(), lang), n
        o = ascii_to_oracc(r)
        oid = self._get(o, lang) or self._get(o.lower(), lang) or self._get(o.upper(), lang)
        if oid is None:
            self.miss[r] += 1
        return oid, o

    def lookup_oracc(self, reading: str, lang: str | None = None):
        """Oracc (Unicode) reading of a grapheme that came without an oid -> oid or None."""
        r = reading
        q = re.match(r"^(.+?)\((.+)\)$", r)
        if q and not re.match(r"^(\d+(?:/\d+)?|n)$", q.group(1)):      # qualified: val(SIGN)
            return self._get(q.group(2)) or self._get(q.group(2).upper())
        oid = self._get(r, lang) or self._get(r.lower(), lang) or self._get(r.upper(), lang)
        if oid is None:
            self.miss_oracc[r] += 1
        return oid

    def _get(self, r, lang=None):
        if lang and (lang, r) in self.oracc_lang:
            return self.oracc_lang[(lang, r)]
        if r in self.oracc:
            return self.oracc[r]
        s = self.osl.get(r)
        if s and len(s) == 1:
            return next(iter(s))
        if s:
            return sorted(s)[0]
        return None
