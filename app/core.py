"""Shared app logic: model, retrieval index, inference + latency, asset paths."""
import os
import sys
import time

import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from common import A1_CHECKPOINT, CLASSES, DEVICE, RESULTS_DIR, embed, load_model  # noqa: E402
from explain import gradcam_overlay, preprocess  # noqa: E402,F401

APP_MODEL = os.path.join(RESULTS_DIR, "app_model.pt")
MODEL_PATH = APP_MODEL if os.path.exists(APP_MODEL) else A1_CHECKPOINT
INDEX_PATH = os.path.join(RESULTS_DIR, "index.npz")
# images are shared with the plain-HTML frontend
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "assets")
SYN_DIR = os.path.join(ASSETS, "synthetic")
HOTDOG = CLASSES.index("hotdog")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def get_index():
    d = np.load(INDEX_PATH, allow_pickle=True)
    return d["feats"], d["labels"], d["paths"]


def thumb(path, label):
    """Thumbnail shipped with the app (the full dataset is not in git)."""
    name = os.path.splitext(os.path.basename(str(path)))[0] + ".jpg"
    return os.path.join(ASSETS, "train", CLASSES[label], name)


def predict(img, runs=10):
    """Class probabilities, feature vector and mean latency (ms) for a PIL image."""
    model = get_model()
    x = preprocess(img).to(DEVICE)
    with torch.no_grad():
        model(x)  # warm-up
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            logits = model(x)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        q = embed(model, x).cpu().numpy()[0]
    return probs, q / (np.linalg.norm(q) + 1e-12), float(np.mean(times))


def nearest(q, k=5):
    feats, labels, paths = get_index()
    sims = feats @ q
    return [(paths[i], int(labels[i]), float(sims[i])) for i in np.argsort(-sims)[:k]]
