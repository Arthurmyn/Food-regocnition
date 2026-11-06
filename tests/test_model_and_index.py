import os

import numpy as np
import pytest
import torch
from PIL import Image

from common import CLASSES, DEPLOY_CHECKPOINT, DEVICE, ROOT, load_model
from explain import gradcam_overlay, preprocess

EXAMPLE = os.path.join(ROOT, "web", "assets", "example.jpg")


@pytest.fixture(scope="module")
def model():
    return load_model(DEPLOY_CHECKPOINT)


def test_deployed_checkpoint_exists_and_is_small_enough_for_git():
    assert os.path.getsize(DEPLOY_CHECKPOINT) < 100 * 1024 * 1024


def test_probabilities_are_a_distribution(model):
    with torch.no_grad():
        p = torch.softmax(model(preprocess(Image.open(EXAMPLE)).to(DEVICE)), 1)[0]
    assert len(p) == len(CLASSES) == 5
    assert abs(float(p.sum()) - 1.0) < 1e-5


def test_example_photo_is_pizza(model):
    """Regression check on the example shown on the Classifier page (a real test-set pizza)."""
    with torch.no_grad():
        p = torch.softmax(model(preprocess(Image.open(EXAMPLE)).to(DEVICE)), 1)[0]
    assert CLASSES[int(p.argmax())] == "pizza" and float(p.max()) > 0.9


def test_gradcam_overlay_is_an_image_of_the_model_input_size(model):
    overlay, heat = gradcam_overlay(model, Image.open(EXAMPLE), CLASSES.index("pizza"))
    assert overlay.shape == (224, 224, 3) and overlay.dtype == np.uint8
    assert heat.shape == (224, 224) and 0.0 <= heat.min() and heat.max() <= 1.0


def test_retrieval_index_is_consistent():
    idx = np.load(os.path.join(ROOT, "results", "a2", "index.npz"), allow_pickle=True)
    feats, labels, paths = idx["feats"], idx["labels"], idx["paths"]
    assert feats.shape == (375, 1280) and len(labels) == len(paths) == 375   # real training images only
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0, atol=1e-4)        # L2-normalised for cosine similarity
    assert set(labels.tolist()) == set(range(len(CLASSES)))
    thumbs = os.path.join(ROOT, "web", "assets", "train")
    assert all(os.path.exists(os.path.join(thumbs, CLASSES[l], os.path.splitext(os.path.basename(str(p)))[0] + ".jpg"))
               for p, l in zip(paths, labels))                              # every neighbour thumbnail ships with the repo


def test_gradcam_without_backward_pass_equals_textbook_gradcam():
    """The app computes Grad-CAM analytically (GAP + Linear head), because the autograd backward pass crashed on
    Streamlit Cloud. It must agree with the textbook gradient version for every class and for both checkpoints."""
    from common import A1_CHECKPOINT
    from explain import gradcam_heatmap, gradcam_heatmap_autograd
    x = preprocess(Image.open(EXAMPLE)).to(DEVICE)
    for path in (DEPLOY_CHECKPOINT, A1_CHECKPOINT):
        m = load_model(path)
        for c in range(len(CLASSES)):
            fast, _ = gradcam_heatmap(m, x, c)
            ref, _ = gradcam_heatmap_autograd(m, x, c)
            assert np.abs(fast - ref).max() < 1e-3


def test_gradcam_does_not_need_autograd(model):
    from explain import gradcam_heatmap
    x = preprocess(Image.open(EXAMPLE)).to(DEVICE)
    with torch.inference_mode():                    # no autograd at all: the analytic version must still work
        heat, _ = gradcam_heatmap(model, x, 0)
    assert heat.shape == (224, 224)
