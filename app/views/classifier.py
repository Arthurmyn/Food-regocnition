"""Service 2 (real one): food classifier with Grad-CAM, nearest neighbours and latency."""
import streamlit as st
from PIL import Image

from core import CLASSES, DEVICE, gradcam_overlay, get_model, nearest, predict, thumb

st.title("Food classifier")
st.caption("burger / hotdog / pizza / sandwich / sushi, EfficientNet-B0 trained on 375 real photos.")

c1, c2 = st.columns(2)
up = c1.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png", "webp"])
cam = c2.camera_input("...or take a photo")
src = up or cam
if src is not None:
    img = Image.open(src).convert("RGB")
    probs, q, ms = predict(img)
    pred = int(probs.argmax())

    st.header(f"Prediction: {CLASSES[pred]} ({probs[pred]:.0%})")
    st.caption(f"Average inference time: {ms:.1f} ms on {DEVICE}")
    a, b, c = st.columns(3)
    a.image(img, caption="Input", width="stretch")
    b.image(gradcam_overlay(get_model(), img, pred)[0], caption=f"Grad-CAM for '{CLASSES[pred]}'",
            width="stretch")
    c.bar_chart({k: float(v) for k, v in zip(CLASSES, probs)})

    st.subheader("Nearest training images (cosine similarity in feature space)")
    for col, (path, label, sim) in zip(st.columns(5), nearest(q)):
        col.image(thumb(path, label), caption=f"{CLASSES[label]}  {sim:.2f}", width="stretch")
