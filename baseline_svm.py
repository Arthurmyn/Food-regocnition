import os
import json
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

SPLIT_DIR = "dataset_split"
RESULTS_DIR = "results"
IMAGE_SIZE = 128
CLASSES = ["burger", "hotdog", "pizza", "sandwich", "sushi"]
SEED = 42


def load_split(split_name):
    X, y = [], []
    split_path = os.path.join(SPLIT_DIR, split_name)
    for label, cls in enumerate(CLASSES):
        cls_dir = os.path.join(split_path, cls)
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            with Image.open(fpath) as img:
                img = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
                arr = np.array(img)
            features = hog(
                arr,
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)


def evaluate(model, scaler, X, y, split_name):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=CLASSES, zero_division=0)

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


def main():
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading train split and extracting HOG features...")
    X_train, y_train = load_split("train")
    print(f"Train features: {X_train.shape}")

    print("Loading val split...")
    X_val, y_val = load_split("val")
    print("Loading test split...")
    X_test, y_test = load_split("test")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("\nTraining SVM (RBF kernel)...")
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)
    model.fit(X_train_scaled, y_train)

    train_metrics = evaluate(model, scaler, X_train, y_train, "train")
    val_metrics = evaluate(model, scaler, X_val, y_val, "val")
    test_metrics = evaluate(model, scaler, X_test, y_test, "test")

    all_metrics = {
        "model": "SVM (RBF kernel, HOG features)",
        "classes": CLASSES,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics_path = os.path.join(RESULTS_DIR, "baseline_svm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics written to: {metrics_path}")


if __name__ == "__main__":
    main()
