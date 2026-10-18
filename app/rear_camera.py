# Camera widget that opens the REAR camera by default (st.camera_input cannot choose a camera).
# A tiny custom Streamlit component (plain HTML/JS in app/rear_camera/). Returns JPEG bytes once a photo is taken.
import base64
import os

import streamlit.components.v1 as components

_component = components.declare_component(
    "rear_camera", path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "rear_camera"))


def rear_camera(key, facing="environment"):
    value = _component(facing=facing, key=key, default=None)
    if not value:
        return None
    return base64.b64decode(value.split(",", 1)[1])
