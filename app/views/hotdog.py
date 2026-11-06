# Service 1 (joke): Hot Dog or Not.
# Opens straight on the camera. Photo -> "Evaluating..." -> green Hotdog / red Not hotdog.
import hashlib
import io
import logging
import time

import streamlit as st
from PIL import Image

import theme
import ui
from rear_camera import rear_camera
from core import CLASSES, DEVICE, HOTDOG, error_text, gradcam_overlay, get_model, load_image, nearest, predict, thumb

MIN_EVALUATING_S = 1.2  # how long the Evaluating... screen stays visible
HOTDOG_THRESHOLD = 0.4  # P(hotdog) needed for a green check (validation F1 is flat for 0.35-0.5)

theme.apply()
ss = st.session_state
ss.setdefault("hd_n", 0)      # camera widget generation: a new key gives a fresh live camera
ss.setdefault("hd_data", None)  # bytes of the current photo
ss.setdefault("hd_info", False)

st.html('<div style="text-align:center;margin-top:.4rem"><h1 style="margin:0;font-size:34px">'
        'Is it a hot dog?</h1><p style="color:var(--muted);margin:.3rem 0 1rem">'
        'A deeply unnecessary computer vision experiment.</p></div>')

_, mid, _ = st.columns([1, 1.25, 1])

with mid:
    if ss.hd_data is None:
        # No photo yet: show the camera right away.
        shot = rear_camera(key=f"hd_cam_{ss.hd_n}")
        with st.expander("No camera? Upload a photo instead"):
            up = st.file_uploader("Photo", type=["jpg", "jpeg", "png", "webp"], key=f"hd_up_{ss.hd_n}",
                                  label_visibility="collapsed")
        data = shot or (up.getvalue() if up is not None else None)
        if data is not None:
            ss.hd_data = data
            st.rerun()
        st.html('<p style="text-align:center;font-size:11px;color:var(--muted);margin-top:2rem">'
                'This model knows only five food classes. It may be confidently wrong about almost '
                'everything else.</p>')
        st.stop()

    try:
        img = load_image(io.BytesIO(ss.hd_data))
        b64 = ui.to_b64(img)
        screen = st.empty()
        key = hashlib.md5(ss.hd_data).hexdigest()
        if ss.get("hd_done") != key:  # not evaluated yet (or an earlier run was interrupted): run it behind the spinner
            screen.html(ui.screen_evaluating(b64))
            started = time.perf_counter()
            ss.hd_probs, ss.hd_q, ss.hd_ms = predict(img)
            time.sleep(max(0.0, MIN_EVALUATING_S - (time.perf_counter() - started)))  # keep the spinner at least this long
            ss.hd_done = key  # set last: only a finished evaluation counts
        probs = ss.hd_probs
    except Exception as e:  # show the real reason instead of a blank page (Streamlit Cloud redacts uncaught errors)
        logging.exception("hot dog evaluation failed")
        st.error(f"Could not analyse this photo: {error_text(e)}. Try a smaller JPG or PNG.")
        st.button("Try again", on_click=lambda: ss.update(hd_data=None, hd_n=ss.hd_n + 1))
        st.stop()
    screen.html(ui.screen_result(b64, bool(probs[HOTDOG] >= HOTDOG_THRESHOLD)))
    st.html(f'<p style="text-align:center;color:var(--muted);font-size:13px;margin:.8rem 0">'
            f'Hot dog probability: <span class="mono">{probs[HOTDOG]:.0%}</span></p>')

    def _again():
        ss.hd_data, ss.hd_n = None, ss.hd_n + 1

    def _toggle():
        ss.hd_info = not ss.hd_info

    st.button("Take another photo", on_click=_again, type="primary", width="stretch")
    st.button("Hide information" if ss.hd_info else "Additional information", on_click=_toggle,
              icon=":material/info:", width="stretch")

if ss.hd_info:
    best = int(probs.argmax())
    cam = gradcam_overlay(get_model(), img, HOTDOG)[0]
    near = "".join(
        f'<figure>{theme.img_tag(Image.open(thumb(p, l)), max_side=256)}'
        f'<figcaption><span>{CLASSES[l].capitalize()}</span><span>{s:.2f}</span></figcaption></figure>'
        for p, l, s in nearest(ss.hd_q))
    st.html(f"""
    <div class="fl-card" style="margin-top:1.5rem">
      <div class="fl-pad">
        <p class="fl-eyebrow">Verdict</p>
        <p style="margin:0">P(hotdog) = <b class="mono">{probs[HOTDOG]:.0%}</b>; it needs
        <span class="mono">{HOTDOG_THRESHOLD:.0%}</span> for a green check. Best guess among the five
        classes: <b>{CLASSES[best]}</b> (<span class="mono">{probs[best]:.0%}</span>).</p>
        {theme.prob_bars(CLASSES, probs, best)}
      </div>
      <div class="fl-sep fl-pad" style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
        <div><p class="fl-eyebrow">Your photo</p>{theme.img_tag(img, style="aspect-ratio:1;border-radius:3px")}</div>
        <div><p class="fl-eyebrow">What made it decide? (Grad-CAM)</p>
          {theme.img_tag(Image.fromarray(cam), style="aspect-ratio:1;border-radius:3px")}</div>
      </div>
      <div class="fl-sep fl-pad"><p class="fl-eyebrow">Closest training photos</p>
        <div class="fl-grid">{near}</div></div>
      {theme.footer(DEVICE, ss.hd_ms)}
    </div>
    <p style="font-size:11px;color:var(--muted);margin-top:.8rem">Joke mode: it reuses the 5-class food model
    (burger, hotdog, pizza, sandwich, sushi) and has never seen shoes, bananas or people. Inspired by the
    "Not Hotdog" gag from the TV show Silicon Valley.</p>""")
