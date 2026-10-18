# Synthetic training images, clearly labelled.
import json
import os

import streamlit as st
from PIL import Image

import theme
from core import CLASSES, SYN_DIR

theme.apply()

stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "a2",
                          "synthetic_filter.json")
stats = json.load(open(stats_path))
generated = sum(v["generated"] for v in stats.values())
accepted = sum(v["accepted"] for v in stats.values())

st.html('<h1 style="margin:0;font-size:28px">Synthetic training images</h1>'
        '<div class="fl-warn" style="margin-top:1.1rem"><b>SYNTHETIC MEDIA</b> — every image on this page was '
        'generated with SD-Turbo for an academic experiment. None is a real photograph.</div>')

side, main = st.columns([3, 9], gap="large")
side.html(f"""<div class="fl-card fl-stat" style="margin-top:1.4rem"><dl style="margin:0">
  <div><dt>Generated</dt><dd>{generated}</dd></div>
  <div><dt>Accepted after filtering</dt><dd style="color:var(--ok)">{accepted}</dd></div>
  <div><dt>Purpose</dt><dd style="font-family:inherit;font-size:14px;line-height:1.45">Test whether synthetic
  images improve classification. In our runs they did not: the gain was within noise.</dd></div>
</dl></div>""")

sections = []
for cls in CLASSES:
    d = os.path.join(SYN_DIR, cls)
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    cells = "".join(f'<div>{theme.img_tag(Image.open(os.path.join(d, f)), cls="", max_side=256)}'
                    f'<span>Synthetic</span></div>' for f in files)
    sections.append(f'<div class="fl-sec"><h2>{cls.capitalize()}</h2>'
                    f'<span>{stats[cls]["accepted"]} / {stats[cls]["generated"]} accepted</span></div>'
                    f'<div class="fl-syn-grid">{cells}</div>')
main.html("".join(sections).replace('class="fl-sec"', 'class="fl-sec" style="margin-top:0"', 1))
