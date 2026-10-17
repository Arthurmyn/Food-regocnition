#!/bin/sh
# Real-data-only baselines: with / without augmentation, 5 seeds.
for s in 0 1 2 3 4; do
  python3 train.py --name real_aug --seed $s
  python3 train.py --name real_noaug --seed $s --no-augment
done
