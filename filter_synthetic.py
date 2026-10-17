"""Keep synthetic images that a real-data-only classifier labels as the intended class (conf >= thr).

    python filter_synthetic.py --thr 0.5
Writes synthetic/accepted/<class>/ and results/synthetic_filter.json.
"""
import argparse
import json
import os
import shutil

import torch
from PIL import Image

from common import CLASSES, DEVICE, RESULTS_DIR, load_model
from explain import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--root", default="synthetic")
    ap.add_argument("--model", default=os.path.join(RESULTS_DIR, "efficientnet_best.pt"))
    args = ap.parse_args()
    model = load_model(args.model)
    acc_dir = os.path.join(args.root, "accepted")
    shutil.rmtree(acc_dir, ignore_errors=True)
    stats = {}
    for cls_i, cls in enumerate(CLASSES):
        src = os.path.join(args.root, "raw", cls)
        os.makedirs(os.path.join(acc_dir, cls), exist_ok=True)
        files = sorted(os.listdir(src))
        kept, confusions = 0, {}
        for f in files:
            with torch.no_grad():
                p = torch.softmax(model(preprocess(Image.open(os.path.join(src, f))).to(DEVICE)), 1)[0]
            pred = int(p.argmax())
            if pred == cls_i and p[pred] >= args.thr:
                shutil.copy2(os.path.join(src, f), os.path.join(acc_dir, cls, f))
                kept += 1
            else:
                confusions[CLASSES[pred]] = confusions.get(CLASSES[pred], 0) + 1
        stats[cls] = {"generated": len(files), "accepted": kept, "rejected_as": confusions}
        print(cls, stats[cls])
    json.dump(stats, open(os.path.join(RESULTS_DIR, "synthetic_filter.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
