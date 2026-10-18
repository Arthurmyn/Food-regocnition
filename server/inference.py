"""Model, prediction with latency, Grad-CAM overlay and nearest training photos for the API."""
import base64
import io
import os
import sys
import threading
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from common import A1_CHECKPOINT, CLASSES, DEVICE, embed, load_model  # noqa: E402
from explain import gradcam_overlay, preprocess  # noqa: E402

MODEL_NAME = "EfficientNet-B0"
INDEX_PATH = os.path.join(ROOT, "results", "a2", "index.npz")
HOTDOG = CLASSES.index("hotdog")

_lock = threading.Lock()  # one forward pass at a time (MPS/CPU safety, stable latency numbers)
_model = load_model(A1_CHECKPOINT)
_idx = np.load(INDEX_PATH, allow_pickle=True)
_feats, _labels, _paths = _idx["feats"], _idx["labels"], _idx["paths"]


def _thumb_url(path, label):
    name = os.path.splitext(os.path.basename(str(path)))[0] + ".jpg"
    return f"/assets/train/{CLASSES[label]}/{name}"


def predict(img, runs=10, k=5):
    """Class probabilities, mean latency (ms) and the k nearest training photos for a PIL image."""
    x = preprocess(img).to(DEVICE)
    with _lock, torch.no_grad():
        _model(x)  # warm-up
        times = []
        for _ in range(runs):
            t = time.perf_counter()
            logits = _model(x)
            if DEVICE.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        q = embed(_model, x).cpu().numpy()[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = _feats @ q
    neighbors = [{"label": CLASSES[int(_labels[i])], "sim": round(float(sims[i]), 3),
                  "url": _thumb_url(_paths[i], int(_labels[i]))} for i in np.argsort(-sims)[:k]]
    top = int(probs.argmax())
    return {"classes": CLASSES, "probs": [round(float(p), 4) for p in probs], "top": top,
            "hotdog_prob": round(float(probs[HOTDOG]), 4), "latency_ms": round(float(np.mean(times)), 1),
            "device": str(DEVICE), "model": MODEL_NAME, "neighbors": neighbors}


def explain(img, class_idx):
    """Grad-CAM overlay for one class as a JPEG data URL."""
    with _lock:
        overlay, _ = gradcam_overlay(_model, img, class_idx)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
