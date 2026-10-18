# Service 2 (the real one): food classifier with Grad-CAM, nearest neighbours and latency.
import io
import os

import streamlit as st
from PIL import Image

import theme
from core import ASSETS, CLASSES, DEVICE, gradcam_overlay, get_model, load_image, nearest, predict, thumb

theme.apply()

st.html('<h1 style="margin:0;font-size:28px">Classify a food image</h1>'
        '<p style="color:var(--muted);margin:.2rem 0 1.4rem">Upload a photo to identify burger, hotdog, '
        'pizza, sandwich, or sushi.</p>')

left, right = st.columns([7, 5], gap="large")

with left:
    up = st.file_uploader("Drop an image here or browse files", type=["jpg", "jpeg", "png", "webp"])
    with st.expander("Use camera"):
        cam = st.camera_input("Take a photo", label_visibility="collapsed")
    src = up or cam

if src is not None:
    img, name = load_image(src), getattr(src, "name", "camera.jpg")
else:  # example: a real test-set photo (never seen in training)
    img, name = load_image(os.path.join(ASSETS, "example.jpg")), "example.jpg"

probs, q, ms = predict(img)
top = int(probs.argmax())
cam_img = Image.fromarray(gradcam_overlay(get_model(), img, top)[0])

with left:
    st.html(f'<div class="fl-card" style="margin-top:1rem"><div style="position:relative">'
            f'{theme.img_tag(img, style="aspect-ratio:4/3")}'
            f'<span class="fl-tag mono" style="position:absolute;left:12px;top:12px">{name}</span></div>'
            f'<div class="fl-foot">Accepted: JPG, PNG, WEBP</div></div>')

with right:
    st.html(f"""<div class="fl-card">
      <div class="fl-pad"><p class="fl-eyebrow">Prediction</p>
        <div class="fl-pred"><b>{CLASSES[top].capitalize()}</b><span class="fl-pct">{probs[top]:.0%}</span></div>
        {theme.prob_bars(CLASSES, probs, top)}</div>
      <div class="fl-sep fl-pad"><p class="fl-eyebrow">Grad-CAM</p>
        {theme.img_tag(cam_img, style="aspect-ratio:1;border-radius:3px")}
        <p style="color:var(--muted);font-size:14px;margin:.6rem 0 0">Warm regions pushed the score
        for <b>{CLASSES[top]}</b> up; cool regions barely mattered.</p></div>
      {theme.footer(DEVICE, ms)}</div>""")

near = "".join(
    f'<figure>{theme.img_tag(Image.open(thumb(p, l)), max_side=256)}'
    f'<figcaption><span>{CLASSES[l].capitalize()}</span><span>{s:.2f}</span></figcaption></figure>'
    for p, l, s in nearest(q))
st.html(f'<div class="fl-sec"><h2>Closest training examples</h2>'
        f'<span>Nearest neighbours by cosine similarity in embedding space</span></div>'
        f'<div class="fl-grid">{near}</div>')
