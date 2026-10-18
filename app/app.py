"""Two services in one web app: a joke one (Hotdog or Not) and the real food classifier.

    streamlit run app/app.py
"""
import os
import platform
import sys

import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import DEVICE, MODEL_PATH  # noqa: E402

st.set_page_config(page_title="Hotdog or Not", page_icon="🌭", layout="centered")

pages = st.navigation([
    st.Page("views/hotdog.py", title="Hotdog or Not", icon="🌭", default=True),
    st.Page("views/classifier.py", title="Food classifier", icon="🔬"),
    st.Page("views/synthetic.py", title="Synthetic images", icon="🧪"),
])

with st.sidebar:
    st.caption(f"Device: **{DEVICE}** · Torch {torch.__version__}  \n{platform.platform()}  \n"
               f"Model: `{os.path.basename(MODEL_PATH)}`")
    st.caption("Uploads are processed in memory and not stored. Do not upload photos of people "
               "without their consent.")

pages.run()
