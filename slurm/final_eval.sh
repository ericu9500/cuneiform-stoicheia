#!/bin/bash
# Final evaluation of one trained model (run once per seed of the chosen configuration):
#   sbatch --job-name=cun-final-<name> --time=02:00:00 slurm/run.sbatch "bash slurm/final_eval.sh <name>"
# Dev (devA) and the untouched test sets (test2, witness-test): uniform-length gaps with one-pass, iterative
# and beam decoding (known width; unknown width by marginalisation), transliteration, realistic damage,
# line-only context, real lacunae.
set -u
export CUN_PROC=data/proc_v3
n=$1
for s in test2; do
  python -m cun.evaluate --ckpt ckpt/v3/$n/best.pt --name $n --split $s
  python -m cun.evaluate --ckpt ckpt/v3/$n/best.pt --name $n --split $s --beam 20
done
python -m cun.evaluate --ckpt ckpt/v3/$n/best.pt --name $n --split devA --beam 20
for s in test2 test2-line witness-test; do
  python -m cun.eval3 predict --ckpt $n/best.pt --name $n --split $s
done
if [ "${2:-}" = "h2h" ]; then python scripts/h2h_predict.py --ckpt ckpt/v3/$n/best.pt; fi
echo FINAL-EVAL-DONE $n
