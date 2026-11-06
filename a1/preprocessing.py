import os
import json

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

SPLIT_DIR = "dataset_split"
RESULTS_DIR = "results/a1"
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_mean_std(loader):
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    n_pixels = 0

    for images, _ in loader:
        b, c, h, w = images.shape
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels += b * h * w

    mean = channel_sum / n_pixels
    std = (channel_sq_sum / n_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def build_transforms(mean, std):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def save_augmentation_preview(train_dir, train_transform_for_preview, out_path, n_samples=6):
    dataset = datasets.ImageFolder(train_dir)
    source_path, _ = dataset.samples[0]
    original = Image.open(source_path).convert("RGB")

    thumb_size = (IMAGE_SIZE, IMAGE_SIZE)
    padding = 8
    sheet_w = (n_samples + 1) * (thumb_size[0] + padding) + padding
    sheet_h = thumb_size[1] + padding * 2 + 20

    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)

    x = padding
    orig_resized = original.resize(thumb_size)
    sheet.paste(orig_resized, (x, padding + 20))
    draw.text((x, 2), "original", fill=(0, 0, 0))
    x += thumb_size[0] + padding

    for i in range(n_samples):
        augmented = train_transform_for_preview(original)
        sheet.paste(augmented, (x, padding + 20))
        draw.text((x, 2), f"aug {i+1}", fill=(0, 0, 0))
        x += thumb_size[0] + padding

    sheet.save(out_path, "JPEG", quality=90)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_dir = os.path.join(SPLIT_DIR, "train")

    if not os.path.isdir(train_dir):
        print(f"'{train_dir}' not found -- run split_dataset.py first.")
        return

    print("Computing dataset mean/std from train split...")
    stat_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    stat_dataset = datasets.ImageFolder(train_dir, transform=stat_transform)
    stat_loader = DataLoader(stat_dataset, batch_size=32, shuffle=False, num_workers=0)
    mean, std = compute_mean_std(stat_loader)

    print(f"Dataset mean: {[round(m, 4) for m in mean]}")
    print(f"Dataset std:  {[round(s, 4) for s in std]}")

    stats = {
        "dataset_mean": mean,
        "dataset_std": std,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "image_size": IMAGE_SIZE,
        "note": (
            "Use dataset_mean/std when training a CNN from scratch (baseline). "
            "Use imagenet_mean/std when fine-tuning a pretrained backbone "
            "(transfer learning)."
        ),
    }
    stats_path = os.path.join(RESULTS_DIR, "normalization_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Normalization stats written to: {stats_path}")

    train_transform, eval_transform = build_transforms(mean, std)

    print("\nTrain transform pipeline:")
    for t in train_transform.transforms:
        print(f"  - {t}")
    print("\nEval transform pipeline (val/test):")
    for t in eval_transform.transforms:
        print(f"  - {t}")

    preview_transform = transforms.Compose([
        t for t in train_transform.transforms
        if not isinstance(t, (transforms.ToTensor, transforms.Normalize))
    ])
    preview_path = os.path.join(RESULTS_DIR, "augmentation_preview.jpg")
    save_augmentation_preview(train_dir, preview_transform, preview_path)
    print(f"\nAugmentation preview written to: {preview_path}")

    full_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    loader = DataLoader(full_dataset, batch_size=8, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    print(f"\nSanity check batch: images.shape={tuple(images.shape)}, labels={labels.tolist()}")
    print(f"Classes: {full_dataset.classes}")


if __name__ == "__main__":
    main()
