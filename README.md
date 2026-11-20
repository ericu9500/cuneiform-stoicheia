# Cuneiform Stoicheia

A sign-level model for cuneiform text: it restores the signs lost in a break and transliterates bare signs.
A tablet is read as a sequence of signs (entries of the Oracc Sign List) with their readings, word
boundaries and determinatives; the model is a masked-diffusion encoder trained from scratch on 121,561
texts from Oracc and CDLI (Sumerian, Akkadian and smaller corpora).

- **Model and examples:** <https://huggingface.co/Ericu950/cuneiform-stoicheia>
- **Demonstration notebook:** [`notebooks/cuneiform_stoicheia_demo.ipynb`](notebooks/cuneiform_stoicheia_demo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ericu9500/cuneiform-stoicheia/blob/main/notebooks/cuneiform_stoicheia_demo.ipynb)
- **Technical report:** in preparation (corpus, model, training, all evaluations, comparison with a
  WordPiece model, suggestions for further work)

Author: Eric Cullhed.

## Using the model

The released model needs only `torch`, `numpy`, `safetensors` and `huggingface_hub`; the inference package
ships with the weights.

```python
import sys
from huggingface_hub import snapshot_download
path = snapshot_download("Ericu950/cuneiform-stoicheia")
sys.path.insert(0, path)
from cuneiform_stoicheia import Restorer

m = Restorer.from_pretrained(path)
t = m.parse("1(disz) udu [x x] lugal-e\nki ur-{d}en-lil2-la2-ta", lang="sux")
for c in m.restore(t, gap=0, topk=5):
    print(f"{c.prob:.3f}  {c.translit}")
print(m.transliterate("𒀭 𒂗 𒆤", lang="sux")["translit"])
```

The notebook walks through single signs, breaks of known and unknown width, real breaks checked against a
duplicate manuscript, transliteration, and the reliability of the model's confidence.

## What is in this repository

| Path | Content |
|---|---|
| `cun/` | the library: source readers (`oracc.py`, `cdli.py`, `signmap.py`), duplicate and composite handling (`composites.py`, `strata.py`), model (`model.py`, `layers.py`), damage patterns for training (`noise.py`, `damage.py`), training (`train.py`), decoding (`decode.py`), evaluation sets and scoring (`evalset.py`, `evaluate.py`, `eval3.py`, `witness.py`), n-gram baselines (`ngram.py`) |
| `scripts/` | corpus construction and audit, evaluation sets, baselines, scoring, analyses |
| `slurm/` | job templates used for training and evaluation |
| `release/` | the files published on the Hugging Face Hub (`hf/`), the tools that produce and check them (`tools/`) |
| `notebooks/` | the demonstration notebook |

Data, checkpoints and result files are not in the repository; they are rebuilt with the steps below.

## Rebuilding the corpus

Sources: the JSON zips of the Oracc projects (`scripts/fetch_oracc.sh` → `data/raw/oracc/`), the Oracc Sign
List (`data/raw/osl.asl`) and the CDLI transliteration dump (ATF).

```bash
export CUN_PROC=data/proc_v3
python scripts/build_corpus.py --round 3      # corpus.pkl.gz, corpus_stats.json, splits
python scripts/audit_corpus.py --tag v3       # consistency checks of the build
python scripts/build_evalset.py devA devB     # uniform-length gaps (test2 likewise)
python -m cun.eval3 build --split devA        # realistic damage (also devB, test2)
python scripts/witness_build.py               # real lacunae answered by a duplicate manuscript
```

Every text carries one split label. Texts that duplicate a held-out text (another exemplar of the same
composition, the same museum number, near-identical wording) are excluded from training. Each derived file
records the hash of the corpus it was built from, and the programs refuse to mix versions.

## Training

```bash
python -m cun.train --name F1 --seed 0 --dropout 0.1 --ema 0.999 --windows random \
    --elastic-flag --width-head --status-plane --compile \
    --noise w_translit=0.10,w_gap=0.20,w_span=0.15,w_word=0.10,w_iid=0.15,w_elastic=0.15,w_damage=0.20
```

This is the released configuration: 12 blocks of width 768, a constant learning rate until the development
loss stalls, then a cosine anneal; the averaged weights of the best checkpoint are saved. One run takes a
few hours on one GPU.

## Evaluation

```bash
python -m cun.evaluate --ckpt ckpt/v3/F1/best.pt --name F1 --split devA            # gaps, transliteration
python -m cun.evaluate --ckpt ckpt/v3/F1/best.pt --name F1 --split devA --beam 20  # ranked candidates
python scripts/eval_gaps_ngram.py --split devA                                     # 4-gram baseline
python scripts/score_gaps.py --split devA                                          # tables
python -m cun.eval3 predict --ckpt F1/best.pt --name F1 --split devA               # realistic damage
python -m cun.eval3 ngram --split devA
python -m cun.eval3 score --split devA
```

`devA-line` gives the line alone as context, `witness-dev` the real lacunae. The test sets (`test2`,
`witness-test`) are meant to be evaluated once, with the final model.

## The release files

```bash
python release/tools/export_data.py     # vocab.json, example tablets and lacunae
python release/tools/export_model.py    # model.safetensors, config.json
python release/tools/verify.py          # the package against the training code and stored predictions
python release/tools/verify_parser.py   # the transliteration parser against the corpus
```

## Licence

Code: MIT (see `LICENSE`). Model weights, vocabulary files and example tablets on the Hugging Face Hub:
CC BY-SA 4.0. The source texts remain under the terms of the Oracc projects and of CDLI. The transformer
primitives follow Stoicheia (Cullhed & Thörn Cleland, Apache-2.0).
