# Make the project importable the same way the apps do (src/ for the ML code, app/ for the Streamlit helpers).
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "src", "app"):
    sys.path.insert(0, os.path.join(ROOT, sub))
os.chdir(ROOT)  # the API and scripts use paths relative to the repo root
