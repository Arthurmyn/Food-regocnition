# Service 1 (joke): Hotdog or Not. Camera/upload -> "Evaluating..." -> green Hotdog / red Not hotdog.
import hashlib
import io
import time

import streamlit as st
from PIL import Image

import ui
from core import HOTDOG, predict

HOTDOG_THRESHOLD = 0.5  # P(hotdog) needed for a green check; anything else is "Not hotdog"

st.markdown("<h1 style='text-align:center;margin-bottom:0'>🌭 Hotdog or Not</h1>"
            "<p style='text-align:center;opacity:.7;margin-top:0'>The only app that tells you "
            "if something is a hotdog.</p>", unsafe_allow_html=True)

mode = st.segmented_control("Source", ["📷 Camera", "🖼 Upload"], default="📷 Camera",
                            label_visibility="collapsed", width="stretch")
if mode == "🖼 Upload":
    src = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"],
                           label_visibility="collapsed")
else:
    src = st.camera_input("Take a photo", label_visibility="collapsed")

screen = st.empty()
if src is None:
    screen.html(ui.screen_empty())
    st.stop()

data = src.getvalue()
key = hashlib.md5(data).hexdigest()
img = Image.open(io.BytesIO(data)).convert("RGB")
b64 = ui.to_b64(img)

if st.session_state.get("hd_key") != key:  # new photo: run the model behind the spinner
    screen.html(ui.screen_evaluating(b64))
    probs, q, ms = predict(img)
    time.sleep(1.2)
    st.session_state.update(hd_key=key, hd_probs=probs, hd_q=q, hd_ms=ms)

probs = st.session_state["hd_probs"]
screen.html(ui.screen_result(b64, bool(probs[HOTDOG] >= HOTDOG_THRESHOLD)))
