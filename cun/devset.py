"""Corpus chunks and the fixed dev noisings (torch-free; shared by cun.train and the baselines)."""
from __future__ import annotations

import numpy as np

from .noise import NoiseConfig, noise_chunk

CHUNK = 256
LINE_END = 6
DEV_SEED = 12345
DEV_N = 600
DEV_PATTERNS = ((0, 0.15), (4, None))       # span at t=0.15; full transliteration


SPLITS = ("train", "dev", "test", "devA", "devB", "test1", "test2")


def in_split(text, split):
    """Does a corpus text belong to `split`? `split` is a label or a tuple of labels. 'dev' means the texts
    used for the dev loss: the per-text flag `dev` (corpus v2: split == 'dev'; corpus v3: devA + devB)."""
    if not isinstance(split, str):
        return any(in_split(text, s) for s in split)
    assert split in SPLITS, split
    return bool(text["dev"]) if split == "dev" else text["split"] == split


def load_chunks(corpus, split, max_texts=None):
    """Return list of (sign, read, join, det, lang) chunks of the texts of one split (see in_split)."""
    texts = corpus["texts"]
    sel = [in_split(t, split) for t in texts]
    langs = {l: i for i, l in enumerate(corpus["langs"])}
    by_text = {}
    for ti, a in corpus["lines"]:
        if not sel[ti]:
            continue
        a = a.copy()
        a[2, -1] = LINE_END
        by_text.setdefault(ti, []).append(a)
    out = []
    for k, (ti, arrs) in enumerate(by_text.items()):
        if max_texts and k >= max_texts:
            break
        a = np.concatenate(arrs, 1)
        lg = langs[texts[ti]["lang"]]
        for s in range(0, a.shape[1], CHUNK):
            c = a[:, s:s + CHUNK]
            if c.shape[1] >= 2:
                out.append((c[0], c[1], c[2], c[3], lg, c[4]))
    return out


def load_texts(corpus, split, max_texts=None):
    """Texts of a split with their line structure, for random training windows:
    [(5 x n array: sign, read, join, det, brk (line ends marked), lang idx, line start offsets)]."""
    texts = corpus["texts"]
    sel = [in_split(t, split) for t in texts]
    langs = {l: i for i, l in enumerate(corpus["langs"])}
    by_text = {}
    for ti, a in corpus["lines"]:
        if not sel[ti]:
            continue
        a = a.copy()
        a[2, -1] = LINE_END
        by_text.setdefault(ti, []).append(a)
    out = []
    for k, (ti, arrs) in enumerate(by_text.items()):
        if max_texts and k >= max_texts:
            break
        a = np.concatenate(arrs, 1)
        if a.shape[1] < 2:
            continue
        starts = np.cumsum([0] + [x.shape[1] for x in arrs[:-1]])
        out.append((a, langs[texts[ti]["lang"]], starts))
    return out


class WindowSampler:
    """Random line-aligned training windows (round 3, R1) in place of fixed 256-sign chunks.
    A text is drawn with probability proportional to ceil(n / CHUNK) (as fixed chunking did). With
    probability p_short the window is short: one line (share p_single of the short ones) or 2..max_short
    lines from a random line; otherwise it starts at the line containing a random offset and takes as many
    whole lines as fit in CHUNK signs (a line longer than CHUNK is cut)."""

    def __init__(self, texts, p_short=0.25, p_single=0.4, max_short=8):
        self.texts = texts
        w = np.array([max(1, -(-t[0].shape[1] // CHUNK)) for t in texts], np.float64)
        self.cum = np.cumsum(w) / w.sum()
        self.p_short, self.p_single, self.max_short = p_short, p_single, max_short

    def __call__(self, rng):
        for _ in range(20):
            a, lg, starts = self.texts[int(np.searchsorted(self.cum, rng.random()))]
            n = a.shape[1]
            ends = np.r_[starts[1:], n]
            if rng.random() < self.p_short:
                k = 1 if rng.random() < self.p_single else int(rng.integers(2, self.max_short + 1))
                i = int(rng.integers(len(starts)))
                j = min(len(starts), i + k) - 1
            else:
                o = int(rng.integers(0, max(1, n - CHUNK + 1)))
                i = int(np.searchsorted(starts, o, side="right")) - 1
                j = int(np.searchsorted(ends, starts[i] + CHUNK, side="right")) - 1
                j = max(i, j)
            s, e = int(starts[i]), int(min(ends[j], starts[i] + CHUNK))
            if e - s >= 2:
                c = a[:, s:e]
                return c[0], c[1], c[2], c[3], lg, c[4]
        c = a[:, :CHUNK]
        return c[0], c[1], c[2], c[3], lg, c[4]


def dev_noisings(chunks, n_lang):
    """[(pattern, chunk index, noised dict)] -- the exact noisings cun.train evaluates on."""
    rng = np.random.default_rng(DEV_SEED)
    pick = rng.choice(len(chunks), min(DEV_N, len(chunks)), replace=False)
    out = []
    for pattern, t in DEV_PATTERNS:
        for i in pick:
            d = noise_chunk(rng, *chunks[i], n_lang=n_lang, cfg=NoiseConfig(p_lang_known=1.0), pattern=pattern, t=t)
            out.append((pattern, int(i), d))
    return out


def check_version(expected, found, what):
    """Derived files record the corpus version they were built from; refuse to mix versions."""
    if expected != found:
        raise SystemExit(f"{what}: corpus version {found!r} does not match the current corpus {expected!r}; rebuild it")
