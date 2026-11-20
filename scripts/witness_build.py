"""Build the duplicate-witness evaluation sets PROC/witness_dev.pkl.gz and PROC/witness_test.pkl.gz
(cun.witness; PLAN_v3 stage 0 item 5). Needs the corpus (~6 GB): run on a compute node.

  CUN_PROC=data/proc_v3 python scripts/witness_build.py [--workers 8]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from cun import witness  # noqa: E402

if __name__ == "__main__":
    sys.argv.insert(1, "build")
    witness.main()
