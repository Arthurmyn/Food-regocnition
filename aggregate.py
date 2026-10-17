"""Aggregate results/runs/*.json into mean +/- std tables (results/summary.md, summary.json)."""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from common import CLASSES, RESULTS_DIR

runs = defaultdict(list)
for p in sorted(glob.glob(os.path.join(RESULTS_DIR, "runs", "*.json"))):
    r = json.load(open(p))
    runs[r["name"]].append(r)

def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"

lines = ["| Condition | n seeds | train real/syn | Val acc | Test acc | Test Top-5 | Test macro-F1 | "
         + " | ".join(f"R {c}" for c in CLASSES) + " |", "|" + "---|" * (7 + len(CLASSES))]
summary = {}
for name, rs in runs.items():
    t = [r["test"] for r in rs]
    row = {"n": len(rs), "test_acc": [x["accuracy"] for x in t], "f1": [x["f1_macro"] for x in t],
           "recall": {c: [x["per_class_recall"][c] for x in t] for c in CLASSES}}
    summary[name] = row
    lines.append(f"| {name} | {len(rs)} | {rs[0]['n_train_real']}/{rs[0]['n_train_synthetic']} | "
                 f"{ms([r['best_val_acc'] for r in rs])} | {ms(row['test_acc'])} | "
                 f"{ms([x['top5'] for x in t])} | {ms(row['f1'])} | "
                 + " | ".join(ms(row["recall"][c]) for c in CLASSES) + " |")
open(os.path.join(RESULTS_DIR, "summary.md"), "w").write("\n".join(lines) + "\n")
json.dump(summary, open(os.path.join(RESULTS_DIR, "summary.json"), "w"), indent=1)
print("\n".join(lines))
