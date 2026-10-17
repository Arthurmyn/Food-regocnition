import os
import random
import shutil

CLASSES = ["burger", "hotdog", "pizza", "sandwich", "sushi"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SOURCE_DIR = "dataset"
SPLIT_DIR = "dataset_split"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

SEED = 42


def find_images(class_folder):
    if not os.path.isdir(class_folder):
        return []
    files = [
        f for f in os.listdir(class_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def split_list(files, train_ratio, val_ratio):
    n = len(files)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def main():
    random.seed(SEED)

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    summary = {}
    for cls in CLASSES:
        class_folder = os.path.join(SOURCE_DIR, cls)
        files = find_images(class_folder)
        random.shuffle(files)
        train, val, test = split_list(files, TRAIN_RATIO, VAL_RATIO)
        summary[cls] = (len(train), len(val), len(test))

        for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
            out_dir = os.path.join(SPLIT_DIR, split_name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(class_folder, f), os.path.join(out_dir, f))

    print(f"{'Class':<10} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>6}")
    for cls in CLASSES:
        tr, va, te = summary[cls]
        print(f"{cls:<10} {tr:>6} {va:>6} {te:>6} {tr+va+te:>6}")

    total_train = sum(v[0] for v in summary.values())
    total_val = sum(v[1] for v in summary.values())
    total_test = sum(v[2] for v in summary.values())
    print(f"\nTotal: train={total_train}, val={total_val}, test={total_test}")
    print(f"\nSplit written to: {SPLIT_DIR}/")


if __name__ == "__main__":
    main()
