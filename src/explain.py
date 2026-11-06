"""Grad-CAM for EfficientNet-B0 (target layer: last conv block), pure PyTorch."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from matplotlib import colormaps

from common import build_transforms


def preprocess(pil_img):
    _, tf = build_transforms(augment=False)
    return tf(ImageOps.exif_transpose(pil_img).convert("RGB")).unsqueeze(0)


def center_crop_rgb(pil_img, size=224):
    """The exact 224x224 crop the model sees, as float RGB in [0,1] (for overlay)."""
    img = ImageOps.exif_transpose(pil_img).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)), dtype=np.float32) / 255.0


def _normalise(cam, size):
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def gradcam_heatmap_autograd(model, x, class_idx=None):
    """Textbook Grad-CAM with a backward pass. Kept as the reference implementation (see tests)."""
    acts = {}
    h = model.features[-1].register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            c = int(logits.argmax(1)) if class_idx is None else class_idx
            grad = torch.autograd.grad(logits[0, c], acts["a"])[0]
    finally:
        h.remove()
    w = grad.mean(dim=(2, 3), keepdim=True)
    return _normalise(F.relu((w * acts["a"]).sum(1, keepdim=True)), x.shape[-2:]), c


def gradcam_heatmap(model, x, class_idx=None):
    """Return (heatmap in [0,1] at input resolution, predicted/target class).

    Grad-CAM without a backward pass. The EfficientNet head is global-average-pooling -> Linear, so
    logit_c = sum_k W[c,k] * mean_hw A_k(h,w) + b_c and d logit_c / d A_k(h,w) = W[c,k] / (H*W): the same at every
    position. The Grad-CAM channel weights (spatial mean of the gradient) are therefore W[c,k] / (H*W), and
    ReLU(sum_k weight_k * A_k) equals ReLU(sum_k W[c,k] * A_k) up to a constant that the min-max normalisation
    removes. This is exact for this architecture and avoids autograd (faster, and it does not depend on the
    PyTorch version's backward engine; the backward pass crashed on Streamlit Cloud). Other heads fall back to autograd.
    """
    head = model.classifier[-1]
    if not isinstance(head, torch.nn.Linear):
        return gradcam_heatmap_autograd(model, x, class_idx)
    acts = {}
    h = model.features[-1].register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    try:
        with torch.no_grad():
            logits = model(x)
    finally:
        h.remove()
    c = int(logits.argmax(1)) if class_idx is None else class_idx
    with torch.no_grad():
        cam = F.relu(torch.einsum("k,bkhw->bhw", head.weight[c], acts["a"]))[:, None]
    return _normalise(cam, x.shape[-2:]), c


def gradcam_overlay(model, pil_img, class_idx=None, alpha=0.45):
    x = preprocess(pil_img).to(next(model.parameters()).device)
    heat, _ = gradcam_heatmap(model, x, class_idx)
    base = center_crop_rgb(pil_img)
    color = colormaps["jet"](heat)[..., :3].astype(np.float32)
    return (np.clip((1 - alpha) * base + alpha * color, 0, 1) * 255).astype(np.uint8), heat
