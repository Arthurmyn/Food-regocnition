import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image, ImageDraw

RESULTS_DIR = "results/a1"
SPLIT_DIR = "dataset_split"
CLASSES = ["burger", "hotdog", "pizza", "sandwich", "sushi"]
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def plot_confusion_matrix(cm, title, out_path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def load_efficientnet_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(CLASSES))
    state = torch.load(os.path.join(RESULTS_DIR, "efficientnet_best.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def collect_misclassified(model, dataset, loader):
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    misclassified = []
    sample_idx = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            preds = outputs.argmax(dim=1).cpu()
            confidences = torch.softmax(outputs, dim=1).cpu()
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                if true_label != pred_label:
                    path, _ = dataset.samples[sample_idx]
                    confidence = confidences[i, pred_label].item()
                    misclassified.append((path, true_label, pred_label, confidence))
                sample_idx += 1
    return misclassified


def save_misclassified_sheet(misclassified, out_path, max_samples=12):
    samples = misclassified[:max_samples]
    if not samples:
        print("No misclassified samples to show.")
        return

    thumb = 150
    padding = 10
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    sheet_w = cols * (thumb + padding) + padding
    sheet_h = rows * (thumb + padding + 20) + padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, (path, true_label, pred_label, confidence) in enumerate(samples):
        row, col = divmod(idx, cols)
        x = padding + col * (thumb + padding)
        y = padding + row * (thumb + padding + 20)
        with Image.open(path) as img:
            img = img.convert("RGB").resize((thumb, thumb))
            sheet.paste(img, (x, y))
        label = f"true:{CLASSES[true_label]} pred:{CLASSES[pred_label]} ({confidence:.2f})"
        draw.text((x, y + thumb + 2), label, fill=(0, 0, 0))

    sheet.save(out_path, "JPEG", quality=90)


def main():
    with open(os.path.join(RESULTS_DIR, "baseline_svm_metrics.json")) as f:
        svm_metrics = json.load(f)
    with open(os.path.join(RESULTS_DIR, "efficientnet_metrics.json")) as f:
        effnet_metrics = json.load(f)

    plot_confusion_matrix(
        svm_metrics["test"]["confusion_matrix"],
        "SVM baseline - Test confusion matrix",
        os.path.join(RESULTS_DIR, "confusion_matrix_svm_test.png"),
    )
    plot_confusion_matrix(
        effnet_metrics["test"]["confusion_matrix"],
        "EfficientNet-B0 - Test confusion matrix",
        os.path.join(RESULTS_DIR, "confusion_matrix_efficientnet_test.png"),
    )
    print("Confusion matrix heatmaps saved.")

    comparison = {
        "svm_test": {
            "accuracy": svm_metrics["test"]["accuracy"],
            "precision_macro": svm_metrics["test"]["precision_macro"],
            "recall_macro": svm_metrics["test"]["recall_macro"],
            "f1_macro": svm_metrics["test"]["f1_macro"],
        },
        "efficientnet_test": {
            "accuracy": effnet_metrics["test"]["accuracy"],
            "precision_macro": effnet_metrics["test"]["precision_macro"],
            "recall_macro": effnet_metrics["test"]["recall_macro"],
            "f1_macro": effnet_metrics["test"]["f1_macro"],
        },
    }
    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("Model comparison written to results/a1/model_comparison.json")

    print("\nLoading EfficientNet model and test set to find misclassified examples...")
    model = load_efficientnet_model()
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    test_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    misclassified = collect_misclassified(model, test_dataset, test_loader)
    print(f"Misclassified on test: {len(misclassified)} / {len(test_dataset)}")

    sheet_path = os.path.join(RESULTS_DIR, "misclassified_examples.jpg")
    save_misclassified_sheet(misclassified, sheet_path)
    print(f"Misclassified examples sheet saved to: {sheet_path}")

    error_counts = {}
    for _, true_label, pred_label, _ in misclassified:
        key = f"{CLASSES[true_label]} -> {CLASSES[pred_label]}"
        error_counts[key] = error_counts.get(key, 0) + 1

    print("\nMost common confusion pairs (true -> predicted):")
    for pair, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  {pair}: {count}")

    with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w") as f:
        json.dump({
            "total_misclassified": len(misclassified),
            "total_test_samples": len(test_dataset),
            "confusion_pairs": error_counts,
        }, f, indent=2)
    print("Error analysis summary written to results/a1/error_analysis.json")


if __name__ == "__main__":
    main()
