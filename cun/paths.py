"""Where the processed corpus lives. CUN_PROC selects a corpus build (default data/proc = corpus v2,
5da416adb1cc, which the model base-long and its evaluation files belong to; round 3 builds into data/proc_v3).

Results and checkpoints are namespaced by the build: with the default corpus they are results/ and ckpt/
(unchanged); with CUN_PROC=data/proc_v3 they are results/v3/ and ckpt/v3/ (TAG = directory name without
'proc_'), so files derived from different corpora cannot overwrite each other."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PROC = os.path.join(ROOT, "data", "proc")
PROC = os.environ.get("CUN_PROC") or DEFAULT_PROC
if not os.path.isabs(PROC):
    PROC = os.path.join(ROOT, PROC)
PROC = os.path.normpath(PROC)
_name = os.path.basename(PROC)
TAG = "" if PROC == DEFAULT_PROC else (_name[len("proc_"):] if _name.startswith("proc_") and len(_name) > 5 else _name)
RESULTS = os.path.join(ROOT, "results", TAG) if TAG else os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "ckpt", TAG) if TAG else os.path.join(ROOT, "ckpt")


def corpus_path():
    return os.path.join(PROC, "corpus.pkl.gz")
