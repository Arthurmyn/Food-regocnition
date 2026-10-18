"""Embedding index + retrieval precision@k.

Index = L2-normalised 1280-d EfficientNet-B0 features of the real TRAIN images.
Query = real TEST images. A neighbour is relevant if it has the same class as the query.

    python retrieval.py                     # deployed model = frozen ImageNet backbone + linear head
    python retrieval.py --model path.pt
"""
import argparse
import json
import os

import numpy as np
import torch
from torchvision import datasets

from common import A1_CHECKPOINT, DEPLOY_CHECKPOINT, DEVICE, RESULTS_DIR, SPLIT_DIR, build_model, build_transforms, embed, load_model

KS = (1, 5, 10)


@torch.no_grad()
def embed_folder(model, folder):
    _, tf = build_transforms(augment=False)
    ds = datasets.ImageFolder(folder, tf)
    feats = []
    for i in range(0, len(ds), 32):
        x = torch.stack([ds[j][0] for j in range(i, min(i + 32, len(ds)))]).to(DEVICE)
        feats.append(embed(model, x).cpu())
    f = torch.cat(feats).numpy()
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
    return f, np.array([y for _, y in ds.samples]), [p for p, _ in ds.samples]


def precision_at_k(train_f, train_y, test_f, test_y, ks=KS):
    sims = test_f @ train_f.T
    order = np.argsort(-sims, axis=1)
    return {f"P@{k}": float(np.mean(train_y[order[:, :k]] == test_y[:, None])) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEPLOY_CHECKPOINT)
    ap.add_argument("--index-out", default=os.path.join(RESULTS_DIR, "index.npz"))
    args = ap.parse_args()

    out = {}
    for name, model in [("a1_finetuned", load_model(A1_CHECKPOINT)),
                        ("deployed_frozen_probe", load_model(args.model))]:
        tr_f, tr_y, tr_p = embed_folder(model, os.path.join(SPLIT_DIR, "train"))
        te_f, te_y, _ = embed_folder(model, os.path.join(SPLIT_DIR, "test"))
        out[name] = precision_at_k(tr_f, tr_y, te_f, te_y)
        print(name, out[name])
        if name == "deployed_frozen_probe":
            np.savez(args.index_out, feats=tr_f, labels=tr_y, paths=np.array(tr_p))
    json.dump(out, open(os.path.join(RESULTS_DIR, "retrieval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
