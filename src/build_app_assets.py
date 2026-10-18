"""Small image copies for the deployed app (the full dataset is not in git).

 - web/assets/train/<class>/<file>.jpg    : 256px thumbnails of every real train image (kNN gallery)
 - web/assets/synthetic/<class>/<file>.jpg: 6 accepted synthetic samples per class
(web/assets/example.jpg is a separate, hand-picked real test photo and is left untouched.)
"""
import os
import shutil

from PIL import Image

from common import CLASSES, SPLIT_DIR

OUT = os.path.join("web", "assets")
for sub in ("train", "synthetic"):
    shutil.rmtree(os.path.join(OUT, sub), ignore_errors=True)


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
    for f in sorted(os.listdir(d))[:6]:
        save(os.path.join(d, f), os.path.join(OUT, "synthetic", c, f))
