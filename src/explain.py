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


def gradcam_heatmap(model, x, class_idx=None):
    """Return (heatmap in [0,1] at input resolution, predicted/target class)."""
    acts = {}
    layer = model.features[-1]
    h = layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = model(x)
            c = int(logits.argmax(1)) if class_idx is None else class_idx
            grad = torch.autograd.grad(logits[0, c], acts["a"])[0]
    finally:
        h.remove()
    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), c


def gradcam_overlay(model, pil_img, class_idx=None, alpha=0.45):
    x = preprocess(pil_img).to(next(model.parameters()).device)
    heat, _ = gradcam_heatmap(model, x, class_idx)
    base = center_crop_rgb(pil_img)
    color = colormaps["jet"](heat)[..., :3].astype(np.float32)
    return (np.clip((1 - alpha) * base + alpha * color, 0, 1) * 255).astype(np.uint8), heat
