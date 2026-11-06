import io

import numpy as np
import torch
from PIL import Image

from explain import center_crop_rgb, preprocess


def _photo():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))


def test_preprocess_shape_and_dtype():
    x = preprocess(_photo())
    assert x.shape == (1, 3, 224, 224) and x.dtype == torch.float32


def test_center_crop_matches_model_input_size():
    crop = center_crop_rgb(_photo())
    assert crop.shape == (224, 224, 3) and 0.0 <= crop.min() and crop.max() <= 1.0


def test_exif_orientation_is_applied():
    """Phone photos store rotated pixels plus an orientation tag; the model must see the upright image."""
    upright = _photo()
    stored = upright.rotate(90, expand=True)            # what a phone writes to disk
    exif = Image.Exif()
    exif[274] = 6                                       # "rotate 90 degrees clockwise to display"
    buf = io.BytesIO()
    stored.save(buf, "JPEG", quality=100, exif=exif.tobytes())
    tagged = Image.open(io.BytesIO(buf.getvalue()))
    buf2 = io.BytesIO()
    upright.save(buf2, "JPEG", quality=100)
    reference = Image.open(io.BytesIO(buf2.getvalue()))

    close = torch.allclose(preprocess(tagged), preprocess(reference), atol=0.15)
    untagged = Image.open(io.BytesIO(_save_without_exif(stored)))
    assert close
    assert not torch.allclose(preprocess(untagged), preprocess(reference), atol=0.15)  # sanity: the tag matters


def _save_without_exif(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=100)
    return buf.getvalue()
