"""Linear probe on FROZEN ImageNet EfficientNet-B0 features (no fine-tuning).

Why: the A1 fine-tuned model (79.7% test) generalises poorly to new photos (synthetic hot dogs: recall 0.38).
A logistic-regression head on frozen features is much simpler and generalises better.

Protocol (test set never used for any choice):
  1. extract frozen 1280-d features for train / val / test (+ synthetic images as an extra stress test)
  2. pick the regularisation C by VALIDATION accuracy
  3. report test metrics of the train-only model
  4. refit on train + val with that C -> deployed model (results/a2/linear_probe.pt, same architecture as A1,
     so the API / Grad-CAM / retrieval code is unchanged). Report its test metrics too.

    python src/train_linear_probe.py
"""
import json
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

from common import (A1_CHECKPOINT, CLASSES, DEVICE, RESULTS_DIR, ROOT, SPLIT_DIR, build_model, build_transforms,
                    load_model)

H = CLASSES.index("hotdog")
SYN_RAW = os.path.join(ROOT, "synthetic", "raw")
SYN_ACC = os.path.join(ROOT, "synthetic", "accepted")
OUT_PT = os.path.join(RESULTS_DIR, "linear_probe.pt")
_, EVAL_TF = build_transforms(augment=False)


def listing(root):
    items = []
    for i, c in enumerate(CLASSES):
        d = os.path.join(root, c)
        items += [(os.path.join(d, f), i) for f in sorted(os.listdir(d))]
    return items


backbone = build_model(pretrained=True).to(DEVICE).eval()


@torch.no_grad()
def features(items, tf=lambda im: im):
    out = []
    for p, _ in items:
        im = tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        x = EVAL_TF(im).unsqueeze(0).to(DEVICE)
        out.append(torch.flatten(backbone.avgpool(backbone.features(x)), 1)[0].cpu().numpy())
    return np.array(out)


def metrics(pred, y):
    return {"accuracy": round(float(accuracy_score(y, pred)), 4), "f1_macro": round(float(f1_score(y, pred, average="macro")), 4),
            "per_class_recall": dict(zip(CLASSES, recall_score(y, pred, average=None).round(3).tolist()))}


def hotdog_pr(prob, y, thresholds=(0.3, 0.4, 0.5, 0.6)):
    is_h, out = y == H, {}
    for t in thresholds:
        pred = prob[:, H] >= t
        tp = int((pred & is_h).sum())
        out[str(t)] = {"precision": round(tp / max(int(pred.sum()), 1), 3), "recall": round(tp / int(is_h.sum()), 3)}
    return out


def main():
    train, val, test = (listing(os.path.join(SPLIT_DIR, s)) for s in ("train", "val", "test"))
    syn_all, syn_acc = listing(SYN_RAW), listing(SYN_ACC)
    ytr, yva, yte, ysa = (np.array([l for _, l in s]) for s in (train, val, test, syn_all))
    print("extracting features...")
    Xtr, Xva, Xte, Xsa = features(train), features(val), features(test), features(syn_all)
    Xac = features(syn_acc)
    yac = np.array([l for _, l in syn_acc])
    Xrot = features(test, lambda im: im.rotate(90, expand=True))

    res = {"selection": {}}
    best_c, best_val = None, -1
    for c in (0.01, 0.03, 0.1, 0.3, 1, 3, 10):
        v = LogisticRegression(C=c, max_iter=5000).fit(Xtr, ytr).score(Xva, yva)
        res["selection"][str(c)] = round(float(v), 4)
        print(f"  C={c}: val {v:.3f}")
        if v > best_val:
            best_c, best_val = c, v
    res["chosen_C"], res["val_acc_at_chosen_C"] = best_c, round(float(best_val), 4)

    lr_tr = LogisticRegression(C=best_c, max_iter=5000).fit(Xtr, ytr)
    lr = LogisticRegression(C=best_c, max_iter=5000).fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]))
    res["probe_train_only"] = {"test": metrics(lr_tr.predict(Xte), yte), "synthetic_all": metrics(lr_tr.predict(Xsa), ysa)}
    res["probe_deployed_train_plus_val"] = {
        "test": metrics(lr.predict(Xte), yte), "synthetic_all": metrics(lr.predict(Xsa), ysa),
        "test_rotated_90": metrics(lr.predict(Xrot), yte),
        "hotdog_vs_rest_test": hotdog_pr(lr.predict_proba(Xte), yte),
        "hotdog_vs_rest_synthetic": hotdog_pr(lr.predict_proba(Xsa), ysa)}

    # synthetic data as extra TRAINING data for the probe (test stays real)
    res["probe_plus_synthetic"] = {}
    for name, (X, y) in {"real+all_synthetic": (Xsa, ysa), "real+filtered_synthetic": (Xac, yac)}.items():
        m = LogisticRegression(C=best_c, max_iter=5000).fit(np.vstack([Xtr, X]), np.concatenate([ytr, y]))
        res["probe_plus_synthetic"][name] = {"test": metrics(m.predict(Xte), yte)}
    res["probe_plus_synthetic"]["real_only (train)"] = {"test": metrics(lr_tr.predict(Xte), yte)}

    # the A1 fine-tuned model on the same data, for a like-for-like comparison
    a1 = load_model(A1_CHECKPOINT)
    def a1_prob(items, tf=lambda im: im):
        with torch.no_grad():
            return np.array([torch.softmax(a1(EVAL_TF(tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))).unsqueeze(0).to(DEVICE)), 1)[0].cpu().numpy() for p, _ in items])
    Pte, Psa, Prot = a1_prob(test), a1_prob(syn_all), a1_prob(test, lambda im: im.rotate(90, expand=True))
    res["a1_finetuned"] = {"test": metrics(Pte.argmax(1), yte), "synthetic_all": metrics(Psa.argmax(1), ysa),
                           "test_rotated_90": metrics(Prot.argmax(1), yte),
                           "hotdog_vs_rest_test": hotdog_pr(Pte, yte), "hotdog_vs_rest_synthetic": hotdog_pr(Psa, ysa)}
    pa, pb = Pte.argmax(1) == yte, lr.predict(Xte) == yte
    res["paired_test"] = {"probe_right_a1_wrong": int((pb & ~pa).sum()), "a1_right_probe_wrong": int((pa & ~pb).sum())}

    # deployable weights: same architecture as the A1 model, frozen ImageNet backbone + the LR head
    model = build_model(pretrained=True)
    with torch.no_grad():
        model.classifier[1].weight.copy_(torch.tensor(lr.coef_, dtype=torch.float32))
        model.classifier[1].bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32))
    torch.save(model.state_dict(), OUT_PT)
    chk = load_model(OUT_PT)
    with torch.no_grad():
        x = EVAL_TF(ImageOps.exif_transpose(Image.open(test[0][0])).convert("RGB")).unsqueeze(0).to(DEVICE)
        res["export_check_max_prob_diff"] = float(np.abs(torch.softmax(chk(x), 1)[0].cpu().numpy() - lr.predict_proba(Xte[:1])[0]).max())
    json.dump(res, open(os.path.join(RESULTS_DIR, "linear_probe_metrics.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("chosen_C", "val_acc_at_chosen_C", "paired_test", "export_check_max_prob_diff")}, indent=1))
    for k in ("a1_finetuned", "probe_train_only", "probe_deployed_train_plus_val"):
        print(k, {s: v["accuracy"] for s, v in res[k].items() if isinstance(v, dict) and "accuracy" in v})
    print("hotdog thr0.5 synthetic: A1", res["a1_finetuned"]["hotdog_vs_rest_synthetic"]["0.5"], "probe", res["probe_deployed_train_plus_val"]["hotdog_vs_rest_synthetic"]["0.5"])
    print("synthetic as training data:", {k: v["test"]["accuracy"] for k, v in res["probe_plus_synthetic"].items()})


if __name__ == "__main__":
    main()
