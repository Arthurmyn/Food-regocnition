"""Parameterised EfficientNet-B0 training (same recipe as A1, plus optional synthetic data).

Synthetic images are added to the TRAIN set only. Val/test are always real images.

Example:
    python train.py --name real_aug --seed 0
    python train.py --name real_plus_syn --seed 0 --synthetic synthetic/accepted
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets

from common import (CLASSES, DEVICE, RESULTS_DIR, SPLIT_DIR, build_model,
                    build_transforms, set_seed)

HEAD_EPOCHS, FINE_TUNE_EPOCHS = 10, 8
HEAD_LR, FINE_TUNE_LR = 1e-3, 1e-5
BATCH_SIZE = 16


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
            preds += out.argmax(1).cpu().tolist()
            labels += y.cpu().tolist()
    return total / len(loader.dataset), accuracy_score(labels, preds)


def evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.softmax(model(x.to(DEVICE)), 1).cpu())
            labels += y.tolist()
    probs = torch.cat(probs).numpy()
    labels = np.array(labels)
    preds = probs.argmax(1)
    top5 = np.argsort(-probs, 1)[:, :min(5, len(CLASSES))]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "top5": float(np.mean([l in t for l, t in zip(labels, top5)])),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "per_class_recall": dict(zip(CLASSES, recall_score(labels, preds, average=None).round(4).tolist())),
        "preds": preds.tolist(),
        "labels": labels.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--synthetic", default=None, help="folder with <class>/ subfolders added to train")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "runs"))
    ap.add_argument("--save-model", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    train_tf, eval_tf = build_transforms(augment=not args.no_augment)

    train_sets = [datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), train_tf)]
    assert train_sets[0].classes == CLASSES
    n_real = len(train_sets[0])
    n_syn = 0
    if args.synthetic:
        syn = datasets.ImageFolder(args.synthetic, train_tf)
        assert syn.classes == CLASSES, f"synthetic classes {syn.classes} != {CLASSES}"
        train_sets.append(syn)
        n_syn = len(syn)
    train_loader = DataLoader(ConcatDataset(train_sets), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"), eval_tf), BATCH_SIZE)
    test_loader = DataLoader(datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), eval_tf), BATCH_SIZE)

    model = build_model().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    crit = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, None

    def phase(params, lr, epochs, tag):
        nonlocal best_acc, best_state
        opt = torch.optim.Adam(params, lr=lr)
        for ep in range(1, epochs + 1):
            tl, ta = run_epoch(model, train_loader, crit, opt)
            vl, va = run_epoch(model, val_loader, crit)
            for k, v in zip(hist, (tl, ta, vl, va)):
                hist[k].append(v)
            print(f"[{tag}] {ep}/{epochs} train {tl:.3f}/{ta:.3f} val {vl:.3f}/{va:.3f}", flush=True)
            if va > best_acc:
                best_acc, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}

    phase(model.classifier.parameters(), HEAD_LR, HEAD_EPOCHS, "head")
    for p in model.parameters():
        p.requires_grad = True
    phase(model.parameters(), FINE_TUNE_LR, FINE_TUNE_EPOCHS, "finetune")

    model.load_state_dict(best_state)
    result = {
        "name": args.name, "seed": args.seed, "augment": not args.no_augment,
        "n_train_real": n_real, "n_train_synthetic": n_syn, "best_val_acc": best_acc,
        "history": hist, "val": evaluate(model, val_loader), "test": evaluate(model, test_loader),
    }
    path = os.path.join(args.out, f"{args.name}_seed{args.seed}.json")
    json.dump(result, open(path, "w"), indent=1)
    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
    print(f"{args.name} seed{args.seed}: val {best_acc:.3f} test acc {result['test']['accuracy']:.3f} "
          f"F1 {result['test']['f1_macro']:.3f} -> {path}")


if __name__ == "__main__":
    main()
