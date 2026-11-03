"""Figures for the write-up: Grad-CAM examples and mean training curves per condition."""
import glob, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from common import CLASSES, DEPLOY_CHECKPOINT, DEVICE, RESULTS_DIR, ROOT, embed, load_model
from explain import center_crop_rgb, gradcam_overlay
import torch
from explain import preprocess

m = load_model(DEPLOY_CHECKPOINT)
fig, ax = plt.subplots(2, 5, figsize=(14, 6))
for j, c in enumerate(CLASSES):
    d = f"dataset_split/test/{c}"; f = sorted(os.listdir(d))[0]
    img = ImageOps.exif_transpose(Image.open(os.path.join(d, f))).convert("RGB")
    with torch.no_grad():
        p = torch.softmax(m(preprocess(img).to(DEVICE)), 1)[0]
    k = int(p.argmax())
    ax[0, j].imshow(center_crop_rgb(img)); ax[0, j].set_title(f"true: {c}")
    ax[1, j].imshow(gradcam_overlay(m, img, k)[0]); ax[1, j].set_title(f"pred: {CLASSES[k]} ({p[k]:.0%})")
    for a in ax[:, j]: a.axis("off")
fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/gradcam_examples.png", dpi=120)

runs = defaultdict(list)
for p in glob.glob(f"{RESULTS_DIR}/runs/*.json"):
    r = json.load(open(p)); runs[r["name"]].append(r["history"])
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for name, hs in sorted(runs.items()):
    for a, key in zip(ax, ("train_loss", "val_acc")):
        a.plot(np.mean([h[key] for h in hs], 0), label=name)
ax[0].set_title("Train loss (mean of 5 seeds)"); ax[1].set_title("Val accuracy (mean of 5 seeds)")
for a in ax: a.axvline(9.5, ls=":", c="gray"); a.set_xlabel("epoch (dotted: head -> fine-tune)"); a.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{RESULTS_DIR}/a2_training_curves.png", dpi=120)

query_path = os.path.join(ROOT, "web", "assets", "example.jpg")
query = ImageOps.exif_transpose(Image.open(query_path)).convert("RGB")
with torch.no_grad():
    q = embed(m, preprocess(query).to(DEVICE)).cpu().numpy()[0]
q /= np.linalg.norm(q) + 1e-12

index = np.load(os.path.join(RESULTS_DIR, "index.npz"), allow_pickle=True)
sims = index["feats"] @ q
nearest = np.argsort(-sims)[:5]
fig, ax = plt.subplots(1, 6, figsize=(15, 3.4))
ax[0].imshow(center_crop_rgb(query))
ax[0].set_title("Query\nreal test example")
ax[0].axis("off")
for a, i in zip(ax[1:], nearest):
    img = ImageOps.exif_transpose(Image.open(str(index["paths"][i]))).convert("RGB")
    a.imshow(center_crop_rgb(img))
    a.set_title(f"{CLASSES[int(index['labels'][i])]}\ncosine {sims[i]:.3f}")
    a.axis("off")
fig.suptitle("Nearest training images (deployed frozen-feature model)")
fig.tight_layout()
fig.savefig(f"{RESULTS_DIR}/nearest_neighbors.png", dpi=140)
