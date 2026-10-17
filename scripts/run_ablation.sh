#!/bin/sh
# Run from the repo root. Real + synthetic ablation (augmentation on), 5 seeds. Eval is always on the real test set.
for s in 0 1 2 3 4; do
  python3 src/train.py --name real_plus_syn_all --seed $s --synthetic synthetic/raw
  python3 src/train.py --name real_plus_syn_filtered --seed $s --synthetic synthetic/accepted
done
