"""Service 1 (joke): Hotdog or Not. Placeholder, redesigned in the next commits."""
import streamlit as st
from PIL import Image

from core import CLASSES, HOTDOG, predict

st.title("Hotdog or Not")
f = st.file_uploader("Upload anything", type=["jpg", "jpeg", "png", "webp"])
if f is not None:
    probs, _, _ = predict(Image.open(f).convert("RGB"))
    if int(probs.argmax()) == HOTDOG:
        st.success(f"Hotdog ({probs[HOTDOG]:.0%})")
    else:
        st.error(f"Not hotdog (looks like {CLASSES[int(probs.argmax())]})")
