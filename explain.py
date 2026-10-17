"""Grad-CAM for EfficientNet-B0 (target layer: last conv block)."""
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from common import build_transforms


def preprocess(pil_img):
    _, tf = build_transforms(augment=False)
    return tf(pil_img.convert("RGB")).unsqueeze(0)


def center_crop_rgb(pil_img, size=224):
    """The exact 224x224 crop the model sees, as float RGB in [0,1] (for overlay)."""
    img = pil_img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)), dtype=np.float32) / 255.0


def gradcam_overlay(model, pil_img, class_idx=None):
    x = preprocess(pil_img).to(next(model.parameters()).device)
    # Grad-CAM runs on CPU-compatible ops; keep the model's device.
    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        targets = None if class_idx is None else [ClassifierOutputTarget(class_idx)]
        heat = cam(input_tensor=x, targets=targets)[0]
    return show_cam_on_image(center_crop_rgb(pil_img), heat, use_rgb=True), heat
