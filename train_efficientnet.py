import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt

SPLIT_DIR = "dataset_split"
RESULTS_DIR = "results"
CLASSES = ["burger", "hotdog", "pizza", "sandwich", "sushi"]
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
BATCH_SIZE = 16
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
SEED = 42

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    return model.to(DEVICE)


def get_head_parameters(model):
    return model.classifier.parameters()


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_phase(model, train_loader, val_loader, criterion, optimizer, epochs, phase_name, history):
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["epoch_label"].append(f"{phase_name}-{epoch}")
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[{phase_name}] epoch {epoch}/{epochs} - "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return best_state, best_val_acc


def evaluate(model, loader, split_name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)

    print(f"\n{split_name.upper()} results:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision (macro): {precision:.4f}")
    print(f"  Recall (macro):    {recall:.4f}")
    print(f"  F1 (macro):        {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(report)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_training_curves(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main():
    set_seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_transform, eval_transform = build_transforms()

    train_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(SPLIT_DIR, "test"), transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model()
    criterion = nn.CrossEntropyLoss()

    history = {"epoch_label": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print("\n=== Phase 1: training classifier head (backbone frozen) ===")
    head_optimizer = torch.optim.Adam(get_head_parameters(model), lr=HEAD_LR)
    best_state_1, best_val_acc_1 = train_phase(
        model, train_loader, val_loader, criterion, head_optimizer, HEAD_EPOCHS, "head", history
    )

    print("\n=== Phase 2: fine-tuning full network ===")
    for param in model.parameters():
        param.requires_grad = True
    fine_tune_optimizer = torch.optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    best_state_2, best_val_acc_2 = train_phase(
        model, train_loader, val_loader, criterion, fine_tune_optimizer, FINE_TUNE_EPOCHS, "finetune", history
    )

    if best_val_acc_2 >= best_val_acc_1:
        model.load_state_dict(best_state_2)
        best_val_acc = best_val_acc_2
    else:
        model.load_state_dict(best_state_1)
        best_val_acc = best_val_acc_1

    print(f"\nBest val accuracy across both phases: {best_val_acc:.4f}")

    curves_path = os.path.join(RESULTS_DIR, "efficientnet_training_curves.png")
    save_training_curves(history, curves_path)
    print(f"Training curves saved to: {curves_path}")

    checkpoint_path = os.path.join(RESULTS_DIR, "efficientnet_best.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Best model weights saved to: {checkpoint_path}")

    train_eval_loader = DataLoader(
        datasets.ImageFolder(os.path.join(SPLIT_DIR, "train"), transform=eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    train_metrics = evaluate(model, train_eval_loader, "train")
    val_metrics = evaluate(model, val_loader, "val")
    test_metrics = evaluate(model, test_loader, "test")

    all_metrics = {
        "model": "EfficientNet-B0 (ImageNet pretrained, fine-tuned)",
        "classes": CLASSES,
        "history": history,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics_path = os.path.join(RESULTS_DIR, "efficientnet_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
