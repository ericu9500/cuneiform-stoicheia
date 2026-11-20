"""Per-text strata for corpus v3 (PLAN_v3.md, stage 0.1): genre class and stratum, period bin, Akkadian dialect.

The tables are explicit: every catalogue string that occurs in the corpus is listed; a string that is not
listed counts as unknown and is reported by the builder (stats['genre_unmapped'], ['period_unmapped']).

Stratum = language x genre class:
  sux-admin   Sumerian documentary: administrative, legal, letters (letter orders)
  sux-lit     Sumerian non-documentary: literary, royal / monumental, lexical, school, scholarly
  akk-doc     Akkadian documentary: letters, legal, administrative
  akk-lit     Akkadian non-documentary: literary, royal, scholarly (omens, astronomy, medicine, ...), lexical
  other       every other language (Eblaite, Urartian, Elamite, Old Persian, Hittite, ...)
  unknown     Sumerian / Akkadian text whose genre no source gives (last resort)
"""
from __future__ import annotations

import re

DOC = ("admin", "legal", "letter", "doc")                      # 'doc': documentary, kind not known (fallbacks)
LIT = ("royal", "literary", "lexical", "school", "scholarly")
STRATA = ("sux-admin", "sux-lit", "akk-doc", "akk-lit", "other", "unknown")
MAIN_STRATA = STRATA[:4]

# genre string (lower case, without '?', '(...)' and anything after ';' / '&') -> fine genre class
GENRE_CLASS = {
    # documentary
    "administrative": "admin", "administrative record": "admin", "receipt": "admin",
    "legal": "legal", "legal transaction": "legal", "contract": "legal", "grant": "legal", "decree": "legal",
    "appointment": "legal", "gift": "legal", "votive donation": "legal",           # SAA 12 grants and donations
    "letter": "letter", "administrative letter": "letter", "scholarly letter": "letter", "priestly letter": "letter",
    # royal / monumental / historiographic
    "royal inscription": "royal", "royal lnscription": "royal", "royal/monumental": "royal",
    "royal/momumental": "royal", "royal stone inscription": "royal", "royal metal inscription": "royal",
    "building inscription": "royal", "private/votive": "royal", "seal inscription": "royal", "divine seal": "royal",
    "official or display": "royal", "treaty": "royal", "historical": "royal", "eponym chronicle": "royal",
    "eponym list": "royal", "clay pithos inscription": "royal", "clay bulla inscription": "royal",
    "clay tablet inscription": "royal", "inscription on a bone bracelet": "royal", "inscription on a perle": "royal",
    "inscription on a ball of agate": "royal",                                     # the last six: Urartian (ecut)
    # literary, liturgical, magical
    "literary": "literary", "literary work": "literary", "liturgy": "literary", "hymn-prayer": "literary",
    "prayer/incantation": "literary", "prayer-incantation": "literary", "incantation-ritual": "literary",
    "ritual": "literary", "royal ritual": "literary", "prophecy": "literary", "magic": "literary",
    "colophon": "literary",
    # lexical, school
    "lexical": "lexical", "lexical text": "lexical", "signlists": "lexical",
    "school": "school",
    # scholarly / technical (incl. the reports and queries of SAA 4 and 8, which quote omens and rituals)
    "omen": "scholarly", "divination": "scholarly", "astrological report": "scholarly", "astrological": "scholarly",
    "astronomical diary": "scholarly", "astronomical": "scholarly", "goal-year text": "scholarly",
    "planetary text": "scholarly", "lunar text": "scholarly", "extispicy query": "scholarly",
    "extispicy report": "scholarly", "medical": "scholarly", "mathematical": "scholarly",
    "metrological": "scholarly", "scientific": "scholarly", "scholarly": "scholarly", "scholarly work": "scholarly",
    "scholarly or scientific": "scholarly", "hemerological": "scholarly", "technical procedure": "scholarly",
    "plan": "scholarly",
    # no information
    "uncertain": None, "unknown": None, "miscellanea": None, "other": None, "fake": None,
}

# CDLI subgenre (used only when no genre is given): substring -> class
SUBGENRE_CLASS = (("medical", "scholarly"), ("witness", "literary"), ("etcsl", "literary"))

# Oracc project (longest matching path prefix) -> class, when no catalogue gives a genre
PROJECT_CLASS = {
    "saao": "letter", "saao/saa02": "royal", "saao/saa03": "literary", "saao/saa04": "scholarly",
    "saao/saa06": "legal", "saao/saa07": "admin", "saao/saa08": "scholarly", "saao/saa09": "literary",
    "saao/saa11": "admin", "saao/saa12": "legal", "saao/saa14": "legal", "saao/saa20": "literary",
    "saao/saas2": "royal",
    "atae": "doc", "balt": "doc", "kish": "doc", "rimanum": "admin", "hbtin": "legal", "borsippa": "doc",
    "tcma": "doc", "obta": "admin", "urap": "doc", "aemw/amarna": "letter", "contrib/amarna": "letter",
    "rinap": "royal", "ribo": "royal", "riao": "royal", "etcsri": "royal", "suhu": "royal", "ario": "royal",
    "ecut": "royal", "cams/selbi": "royal", "aemw/alalakh": "royal",
    "dcclt": "lexical", "edlex": "lexical", "obmc": "school",
    "obel": "literary", "eisl": "literary", "blms": "literary", "caspo": "literary", "cmawro": "literary",
    "akklove": "literary", "dsst": "literary", "ckst": "literary", "nere": "literary", "cams/ludlul": "literary",
    "btto": "scholarly", "babcity": "scholarly", "cams": "scholarly", "adsd": "scholarly", "asbp": "scholarly",
    "ccpo": "scholarly", "dccmt": "scholarly", "glass": "scholarly",
}

PERIOD_BINS = ("ED", "OAkk", "LagashII", "UrIII", "OA", "OB", "MB", "MA", "NA", "NB", "Late", "FirstMill",
               "other", "unknown")

# period string (lower case, without '(...)', first of several) -> bin
PERIOD_BIN = {
    "ed i-ii": "ED", "ed iiia": "ED", "ed iiib": "ED", "early dynastic": "ED", "early dynastic i-ii": "ED",
    "early dynastic iiia": "ED", "early dynastic iiib": "ED", "early dynastic or old akkadian": "ED",
    "early dynastic or ur iii": "ED", "ebla": "ED",
    "old akkadian": "OAkk",
    "lagash ii": "LagashII", "lagaš ii": "LagashII",
    "ur iii": "UrIII",
    "old assyrian": "OA",
    "old babylonian": "OB", "early old babylonian": "OB", "sealand dynasty": "OB",
    "middle babylonian": "MB",
    "middle assyrian": "MA", "late middle assyrian or early neo-assyrian": "MA",
    "neo-assyrian": "NA",
    "neo-babylonian": "NB", "neo babylonian": "NB", "early neo-babylonian": "NB",
    "achaemenid": "Late", "persian": "Late", "hellenistic": "Late", "seleucid": "Late", "parthian": "Late",
    "late babylonian": "Late", "late antique": "Late",
    "first millennium": "FirstMill", "standard babylonian": "FirstMill",       # first millennium, not closer dated
    "urartian": "other", "urartian eastern anatolia": "other", "9th century": "other", "9th/8th century": "other",
    "8th century": "other", "8th/7th century": "other", "7th century": "other",   # centuries: Urartian (ecut)
    "hittite": "other", "middle hittite": "other", "middle hittit": "other", "middle elamite": "other",
    "uncertain": None, "unknown": None, "unclear date": None, "fake": None,
}

# Oracc project -> period bin, when no catalogue gives a period
PROJECT_PERIOD = {
    "atae": "NA", "saao": "NA", "rinap": "NA", "adsd": "Late", "hbtin": "Late", "balt": "FirstMill",
    "cmawro": "FirstMill", "ccpo": "FirstMill", "babcity": "FirstMill", "caspo": "FirstMill", "blms": "FirstMill",
    "eisl": "FirstMill", "btto": "FirstMill", "ecut": "other", "obel": "OB", "rimanum": "OB",
}

# Oracc Akkadian dialect tag -> period bin (last fallback for Oracc texts without a period)
DIALECT_PERIOD = {"akk-x-oldass": "OA", "akk-x-oldbab": "OB", "akk-x-midass": "MA", "akk-x-midbab": "MB",
                  "akk-x-mbperi": "MB", "akk-x-neoass": "NA", "akk-x-neobab": "NB", "akk-x-ltebab": "Late",
                  "akk-x-stdbab": "FirstMill", "akk-x-oldakk": "OAkk", "akk-x-earakk": "OAkk"}


def _norm(s):
    s = re.sub(r"\([^)]*\)?", " ", (s or "").lower())
    s = re.split(r"[;&,]", s)[0].replace("?", " ")
    return " ".join(s.split())


def _project(table, proj):
    while proj:
        if proj in table:
            return table[proj]
        proj = proj.rpartition("/")[0]
    return None


def genre_class(genres, subgenre=None, project=None, lang=None, period_bin=None, unmapped=None):
    """(fine class or None, source). `genres`: [(source name, genre string)] in order of trust; then the CDLI
    subgenre, the Oracc project, and one period rule (CDLI Old Assyrian Akkadian texts without a genre are the
    Kültepe merchant archives: documentary)."""
    for src, g in genres:
        k = _norm(g)
        if not k:
            continue
        if k not in GENRE_CLASS:
            if unmapped is not None:
                unmapped[k] += 1
            continue
        if GENRE_CLASS[k]:
            return GENRE_CLASS[k], src
    for sub, cls in SUBGENRE_CLASS:
        if sub in (subgenre or "").lower():
            return cls, "subgenre"
    cls = _project(PROJECT_CLASS, project)
    if cls:
        return cls, "project"
    if not project and lang == "akk" and period_bin == "OA":
        return "doc", "period"
    return None, "none"


def stratum(lang, cls):
    if lang not in ("sux", "akk"):
        return "other"
    if cls is None:
        return "unknown"
    doc = cls in DOC
    return ("sux-admin" if doc else "sux-lit") if lang == "sux" else ("akk-doc" if doc else "akk-lit")


def period_bin(periods, project=None, lang=None, dialect=None, unmapped=None):
    """(bin, source). `periods`: [(source name, period string)] in order of trust; then the Oracc project, the
    Akkadian dialect tag, and the language (Eblaite = Ebla archives)."""
    for src, p in periods:
        k = _norm(p)
        if not k:
            continue
        if k not in PERIOD_BIN:
            if unmapped is not None:
                unmapped[k] += 1
            continue
        if PERIOD_BIN[k]:
            return PERIOD_BIN[k], src
    b = _project(PROJECT_PERIOD, project)
    if b:
        return b, "project"
    if dialect in DIALECT_PERIOD:
        return DIALECT_PERIOD[dialect], "dialect"
    if lang == "qeb":
        return "ED", "language"
    return "unknown", "none"


def dialect(lang, tags):
    """Akkadian dialect of an Oracc text: the most frequent full language tag among its Akkadian words, if it
    names a dialect ('akk-x-neoass'); None for plain 'akk', other languages and CDLI texts."""
    if lang != "akk" or not tags:
        return None
    akk = sorted(((n, t) for t, n in tags.items() if t.lower().split("-")[0] == "akk"), reverse=True)
    if not akk or "-" not in akk[0][1]:
        return None
    return akk[0][1].lower()
