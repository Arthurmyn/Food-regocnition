"""Binary hotdog-vs-rest evaluation of the 5-class model (used by the joke app), real test set.

Rule used in the app: "Hotdog" iff P(hotdog) >= 0.4 (chosen on validation). Also probes a few non-photo inputs.
Writes results/a2/hotdog_binary.json.
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from common import CLASSES, DEPLOY_CHECKPOINT, DEVICE, RESULTS_DIR, SPLIT_DIR, load_model
from explain import preprocess

H = CLASSES.index("hotdog")
model = load_model(DEPLOY_CHECKPOINT)


def p_hotdog(img):
    with torch.no_grad():
        return torch.softmax(model(preprocess(img).to(DEVICE)), 1)[0, H].item()


ps, ys = [], []
for c in CLASSES:
    d = os.path.join(SPLIT_DIR, "test", c)
    for f in sorted(os.listdir(d)):
        ps.append(p_hotdog(Image.open(os.path.join(d, f))))
        ys.append(c == "hotdog")
ps, ys = np.array(ps), np.array(ys)

out = {"n_test": len(ys), "n_hotdog": int(ys.sum()), "thresholds": {}}
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    pred = ps >= t
    tp = int((pred & ys).sum())
    out["thresholds"][str(t)] = {"precision": tp / max(int(pred.sum()), 1), "recall": tp / int(ys.sum()),
                                 "accuracy": float((pred == ys).mean())}

rng = np.random.default_rng(0)
probes = {"noise": Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))}
for name, col in [("white", (255, 255, 255)), ("black", (0, 0, 0)), ("red", (200, 30, 30)), ("tan", (210, 170, 120))]:
    probes[name] = Image.new("RGB", (300, 300), col)
g = np.tile(np.linspace(0, 255, 300, dtype=np.uint8), (300, 1))
probes["gradient"] = Image.fromarray(np.stack([g, g // 2, 255 - g], 2))
txt = Image.new("RGB", (300, 300), "white")
ImageDraw.Draw(txt).text((60, 140), "NOT FOOD", fill="black")
probes["text"] = txt
out["non_photo_p_hotdog"] = {k: round(p_hotdog(v), 3) for k, v in probes.items()}

json.dump(out, open(os.path.join(RESULTS_DIR, "hotdog_binary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
