# Service 1 (joke): Hotdog or Not. Camera/upload -> "Evaluating..." -> green Hotdog / red Not hotdog.
import hashlib
import io
import time

import streamlit as st
from PIL import Image

import ui
from core import CLASSES, DEVICE, HOTDOG, gradcam_overlay, get_model, nearest, predict, thumb

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


# --- Additional information (button below the phone) ---
def _toggle_info():
    st.session_state.hd_info = not st.session_state.get("hd_info", False)


show_info = st.session_state.get("hd_info", False)
st.button("Hide information" if show_info else "Additional information", on_click=_toggle_info,
          icon="ℹ️", width="stretch")

if show_info:
    with st.container(border=True):
        best = int(probs.argmax())
        st.markdown(f"**Verdict:** P(hotdog) = **{probs[HOTDOG]:.0%}** "
                    f"(needs ≥ {HOTDOG_THRESHOLD:.0%} for a green check). "
                    f"Best guess among the 5 classes: **{CLASSES[best]}** ({probs[best]:.0%}).")
        st.bar_chart({k: float(v) for k, v in zip(CLASSES, probs)}, height=200)

        a, b = st.columns(2)
        a.image(img, caption="Your photo", width="stretch")
        b.image(gradcam_overlay(get_model(), img, HOTDOG)[0], caption="Where the model looks for 'hotdog' "
                "(Grad-CAM)", width="stretch")

        st.markdown("**Closest training photos**")
        for col, (path, label, sim) in zip(st.columns(5), nearest(st.session_state["hd_q"])):
            col.image(thumb(path, label), caption=f"{CLASSES[label]} {sim:.2f}", width="stretch")

        st.caption(f"Inference: {st.session_state['hd_ms']:.1f} ms on {DEVICE}. This joke mode reuses the "
                   "5-class food model (burger, hotdog, pizza, sandwich, sushi). It has never seen shoes, "
                   "bananas or people, so on non-food it can be confidently wrong. Inspired by the "
                   "'Not Hotdog' gag from the TV show Silicon Valley.")
