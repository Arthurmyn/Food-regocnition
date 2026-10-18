# Two services in one web app: the joke one (Hot Dog or Not) and the real food classifier.
#
#     streamlit run app/app.py
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import theme  # noqa: E402

st.set_page_config(page_title="Food Lens", page_icon="🌭", layout="centered")

pages = [
    st.Page("views/classifier.py", title="Classifier", url_path="classifier"),
    st.Page("views/hotdog.py", title="Hot Dog or Not", url_path="hotdog", default=True),
    st.Page("views/synthetic.py", title="Synthetic Data", url_path="synthetic"),
]
current = st.navigation(pages, position="hidden")
theme.header(pages)
current.run()
