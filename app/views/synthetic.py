# Synthetic training images, clearly labelled. Same markup and stylesheet as the plain-HTML frontend.
import html
import json
import os

import streamlit as st
from PIL import Image

import theme
import ui
from core import CLASSES, SYN_DIR

theme.apply()
esc, cap = html.escape, str.capitalize

stats = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "a2", "synthetic_filter.json")))
generated = sum(v["generated"] for v in stats.values())
accepted = sum(v["accepted"] for v in stats.values())

WARN_ICON = theme.svg_img('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#6F1448" stroke-width="2" '
                          'stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 2.5 20h19L12 3z"/><path d="M12 10v4.5M12 17.5v.01"/></svg>', 22)

sections = []
for cls in CLASSES:
    d = os.path.join(SYN_DIR, cls)
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    tiles = "".join(f'<div class="tile"><img src="data:image/jpeg;base64,{ui.to_b64(Image.open(os.path.join(d, f)), 256)}" '
                    f'alt="Synthetic {esc(cls)}"><span>Synthetic</span></div>' for f in files)
    v = stats[cls]
    sections.append(f"""<section class="syn-class"><div><h2>{esc(cap(cls))}</h2>
      <div class="accept" role="img" aria-label="{v['accepted']} of {v['generated']} accepted"><i style="width:{v['accepted'] / v['generated'] * 100:.0f}%"></i></div>
      <p class="of"><span class="num">{v['accepted']}</span> of <span class="num">{v['generated']}</span> accepted</p></div>
      <div class="tiles">{tiles}</div></section>""")

with st.container(key="lens-page"):
    st.html(f"""<div class="lens"><h1>Synthetic training images</h1>
      <div class="syn-warn" role="note">{WARN_ICON}<span><b>SYNTHETIC MEDIA.</b> Every image on this page was generated with SD-Turbo
      for an academic experiment. None is a real photograph.</span></div>
      <p class="lede" style="margin-top:30px">SD-Turbo generated <b class="num">{generated}</b> images and a classifier trained on real photos kept
      <b class="num">{accepted}</b> ({round(accepted / generated * 100)}%). Adding them to training did not improve accuracy on real test photos:
      the change was within noise.</p>{"".join(sections)}</div>""")
