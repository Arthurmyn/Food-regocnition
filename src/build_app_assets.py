"""Small image copies for the deployed app (the full dataset is not in git).

 - app/assets/train/<class>/<file>.jpg    : 256px thumbnails of every real train image (kNN gallery)
 - app/assets/synthetic/<class>/<file>.jpg: 5 accepted synthetic samples per class
"""
import os
import shutil

from PIL import Image

from common import CLASSES, SPLIT_DIR

OUT = os.path.join("app", "assets")
shutil.rmtree(OUT, ignore_errors=True)


def save(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.thumbnail((256, 256))
    im.save(os.path.splitext(dst)[0] + ".jpg", quality=80)


for c in CLASSES:
    d = os.path.join(SPLIT_DIR, "train", c)
    for f in sorted(os.listdir(d)):
        save(os.path.join(d, f), os.path.join(OUT, "train", c, f))
    d = os.path.join("synthetic", "accepted", c)
    for f in sorted(os.listdir(d))[:5]:
        save(os.path.join(d, f), os.path.join(OUT, "synthetic", c, f))
