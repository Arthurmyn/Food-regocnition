"""Shared helpers for the A2 pipeline (training, retrieval, explainability, app)."""
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

SPLIT_DIR = "dataset_split"
RESULTS_DIR = "results"
CLASSES = ["burger", "hotdog", "pizza", "sandwich", "sushi"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(augment=True):
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(), norm])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), norm])
    return train_tf, eval_tf


def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    return model


def load_model(path=os.path.join(RESULTS_DIR, "efficientnet_best.pt"), device=DEVICE):
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def embed(model, x):
    """1280-d pooled feature vector (input of the classifier head)."""
    f = model.avgpool(model.features(x))
    return torch.flatten(f, 1)


def real_train_dataset(transform):
    return datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=transform)
