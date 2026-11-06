"""Publish all generated synthetic images as small JPEGs (the 150 MB of originals stay local and git-ignored).

Writes results/a2/synthetic_gallery/{accepted,rejected}/<class>/*.jpg (256 px) and copies the prompts and seeds to
results/a2/synthetic_gallery/metadata.csv, so every image's provenance can be inspected.

    python src/export_synthetic_gallery.py
"""
import os
import shutil

from PIL import Image

from common import CLASSES, ROOT

SRC = os.path.join(ROOT, "synthetic")
OUT = os.path.join(ROOT, "results", "a2", "synthetic_gallery")

shutil.rmtree(OUT, ignore_errors=True)
counts = {"accepted": 0, "rejected": 0}
for cls in CLASSES:
    accepted = set(os.listdir(os.path.join(SRC, "accepted", cls)))
    for name in sorted(os.listdir(os.path.join(SRC, "raw", cls))):
        kind = "accepted" if name in accepted else "rejected"
        dst = os.path.join(OUT, kind, cls)
        os.makedirs(dst, exist_ok=True)
        im = Image.open(os.path.join(SRC, "raw", cls, name)).convert("RGB")
        im.thumbnail((256, 256))
        im.save(os.path.join(dst, os.path.splitext(name)[0] + ".jpg"), quality=78, optimize=True)
        counts[kind] += 1
shutil.copy(os.path.join(SRC, "metadata.csv"), os.path.join(OUT, "metadata.csv"))
print(counts)
