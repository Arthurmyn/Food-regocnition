# Service 2 (the real one): food classifier. Same markup and stylesheet as the plain-HTML frontend (web/css/lens.css).
import html
import io
import os

import streamlit as st
from PIL import Image

import theme
import ui
from core import ASSETS, CLASSES, DEVICE, MODEL_PATH, HOTDOG, gradcam_overlay, get_model, load_image, nearest, predict, thumb
from rear_camera import rear_camera

theme.apply()
esc, cap = html.escape, str.capitalize


def uri(img, max_side=900):
    return "data:image/jpeg;base64," + ui.to_b64(img, max_side)


with st.container(key="lens-page"):
    st.html('<div class="lens"><h1>Classify a food image</h1><p class="lede">Drop in a photo and the model names the dish, '
            'shows how sure it is, and shows where it looked.</p></div>')

    up = st.file_uploader("Drop a photo here, or choose one", type=["jpg", "jpeg", "png", "webp"])
    cam = rear_camera(key="cls_cam") if st.toggle("Use camera") else None  # only mounted when switched on
    src = up or (io.BytesIO(cam) if cam else None)
    if src is not None:
        img, name = load_image(src), getattr(src, "name", "camera.jpg")
    else:  # example: a real test-set photo (never seen in training)
        img, name = load_image(os.path.join(ASSETS, "example.jpg")), "example.jpg"

    probs, q, ms = predict(img)
    order = sorted(range(len(CLASSES)), key=lambda i: -probs[i])
    top = order[0]
    heat = Image.fromarray(gradcam_overlay(get_model(), img, top)[0])
    pct = round(probs[top] * 100)
    sure = (f'<b class="num">{pct}%</b> sure.' if probs[top] >= 0.5
            else f'Only <b class="num">{pct}%</b> sure, so treat this as a guess.')

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.html(f"""<div class="lens"><div class="stage">
          <input class="vh" type="radio" name="view" id="v-photo" checked><input class="vh" type="radio" name="view" id="v-heat">
          <div class="stage-frame"><img src="{uri(img)}" alt="Photo being classified"><img class="heat" src="{uri(heat)}" alt="Grad-CAM heatmap over the photo"></div>
          <div class="seg"><label for="v-photo">Photo</label><label for="v-heat">Heatmap</label></div>
          <span class="src">{esc(name)}</span></div>
          <p class="stage-note">Heatmap: warm areas pushed the score for {esc(CLASSES[top])} up; cool areas barely mattered.</p></div>""")
    with right:
        rows = "".join(
            f'<li class="{"top" if k == 0 else ""}"><span>{esc(cap(CLASSES[i]))}</span><span class="bar"><i style="width:{probs[i] * 100:.1f}%"></i></span>'
            f'<span class="n">{round(probs[i] * 100)}%</span></li>' for k, i in enumerate(order))
        st.html(f"""<div class="lens readout"><h2>{esc(cap(CLASSES[top]))}</h2><p class="sure">{sure}</p><ul class="rank">{rows}</ul>
          <div class="facts"><span>{round(ms)} ms</span><span>{esc(str(DEVICE))}</span><span>EfficientNet-B0 (frozen ImageNet features + linear head)</span></div></div>""")

    near = "".join(
        f'<figure><img src="{uri(Image.open(thumb(p, l)), 300)}" alt="{esc(CLASSES[l])} training photo">'
        f'<figcaption><span>{esc(cap(CLASSES[l]))}</span><span>{s:.2f}</span></figcaption></figure>' for p, l, s in nearest(q))
    st.html(f'<div class="lens"><div class="sec2"><h2>Closest training photos</h2>'
            f"<p>Nearest neighbours by cosine similarity in the model's feature space.</p><div class=\"contact\">{near}</div></div></div>")
