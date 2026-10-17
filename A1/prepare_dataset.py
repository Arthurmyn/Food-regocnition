import os
import sys
import csv
import shutil
import hashlib
import random
from pathlib import Path

from PIL import Image, ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = False

try:
    import imagehash
    HAVE_IMAGEHASH = True
except ImportError:
    HAVE_IMAGEHASH = False


GDRIVE_FOLDER_URL = ""

DATASET_PATH = "dataset"
CLASSES = ["burger", "hotdog", "pizza", "sandwich", "sushi"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RESULTS_DIR = "results/a1"
PROBLEMS_DIR = os.path.join(DATASET_PATH, "_problems")

MIN_REASONABLE_DIMENSION = 200
PREVIEW_SAMPLES_PER_CLASS = 5


def download_dataset_from_drive(url, dest_dir):
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Run: pip3 install gdown")
        sys.exit(1)

    if "/file/d/" in url:
        import zipfile
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "dataset.zip")
            print(f"Downloading zip from Google Drive...")
            gdown.download(url=url, output=zip_path, quiet=False, fuzzy=True)
            print(f"Extracting into {dest_dir}/ ...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_dir)
        return

    print(f"Downloading dataset from Google Drive into {dest_dir}/ ...")
    try:
        gdown.download_folder(url=url, output=dest_dir, quiet=False, use_cookies=False)
    except Exception as e:
        if "50 files" in str(e) or "FolderContentsMaximumLimitError" in type(e).__name__:
            print()
            print("Google Drive won't list more than 50 files per folder without login.")
            print("Fix: on Google Drive, zip the whole dataset folder, share the .zip file")
            print("(not the folder) as 'Anyone with the link', and paste that file's link instead.")
            sys.exit(1)
        raise


def find_images(class_folder):
    if not os.path.isdir(class_folder):
        return []

    files = []
    for name in os.listdir(class_folder):
        path = os.path.join(class_folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            files.append(path)
    return sorted(files)


def file_hash(path, block_size=65536):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def inspect_image(path):
    info = {
        "path": path,
        "file_size": os.path.getsize(path),
        "is_valid": False,
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "md5_hash": None,
        "phash": None,
        "error": None,
    }

    try:
        info["md5_hash"] = file_hash(path)
    except OSError as e:
        info["error"] = f"Could not read file bytes: {e}"
        return info

    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            info["width"], info["height"] = img.size
            info["format"] = img.format
            info["mode"] = img.mode
            info["is_valid"] = True

            if HAVE_IMAGEHASH:
                try:
                    info["phash"] = str(imagehash.phash(img))
                except Exception as e:
                    info["error"] = f"phash failed: {e}"

    except Exception as e:
        info["error"] = str(e)
        info["is_valid"] = False

    return info


def build_contact_sheet(records_by_class, output_path):
    thumb_size = (150, 150)
    padding = 10

    rows = []
    for class_name in CLASSES:
        valid_records = [r for r in records_by_class[class_name] if r["is_valid"]]
        sample = random.sample(valid_records, min(PREVIEW_SAMPLES_PER_CLASS, len(valid_records)))
        rows.append((class_name, sample))

    if not any(sample for _, sample in rows):
        print("No valid images available to build a preview contact sheet.")
        return

    cols = max((len(sample) for _, sample in rows), default=1)
    cols = max(cols, 1)

    row_height = thumb_size[1] + padding * 2 + 25
    sheet_width = cols * (thumb_size[0] + padding) + padding
    sheet_height = len(rows) * row_height

    from PIL import ImageDraw
    sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    y = 0
    for class_name, sample in rows:
        draw.text((padding, y + 5), class_name.upper(), fill=(0, 0, 0))
        x = padding
        thumb_y = y + 25
        for record in sample:
            try:
                with Image.open(record["path"]) as img:
                    img = img.convert("RGB")
                    img.thumbnail(thumb_size)
                    sheet.paste(img, (x, thumb_y))
            except Exception:
                pass
            x += thumb_size[0] + padding
        y += row_height

    sheet.save(output_path, "JPEG", quality=90)


def build_eda_plots(records_by_class, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    counts = [sum(1 for r in records_by_class[c] if r["is_valid"]) for c in CLASSES]
    axes[0].bar(CLASSES, counts, color="steelblue")
    axes[0].set_title("Images per class")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=30)

    all_valid = [r for c in CLASSES for r in records_by_class[c] if r["is_valid"]]
    max_sides = [max(r["width"], r["height"]) for r in all_valid]
    axes[1].hist(max_sides, bins=30, color="darkorange")
    axes[1].set_title("Longest side (px)")
    axes[1].set_xlabel("pixels")
    axes[1].set_ylabel("count")

    aspect_ratios = [r["width"] / r["height"] for r in all_valid]
    axes[2].hist(aspect_ratios, bins=30, color="seagreen")
    axes[2].set_title("Aspect ratio (width / height)")
    axes[2].set_xlabel("ratio")
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def summarize_dimensions(records):
    valid = [r for r in records if r["is_valid"]]
    if not valid:
        return None

    widths = [r["width"] for r in valid]
    heights = [r["height"] for r in valid]
    unusual = sum(
        1 for r in valid
        if r["width"] < MIN_REASONABLE_DIMENSION or r["height"] < MIN_REASONABLE_DIMENSION
    )

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_width": sum(widths) / len(widths),
        "avg_height": sum(heights) / len(heights),
        "unusual_count": unusual,
    }


def find_exact_duplicates(all_records):
    by_hash = {}
    for r in all_records:
        if r["md5_hash"] is None:
            continue
        by_hash.setdefault(r["md5_hash"], []).append(r["path"])
    return {h: paths for h, paths in by_hash.items() if len(paths) > 1}


def find_similar_duplicates(all_records, max_distance=5):
    if not HAVE_IMAGEHASH:
        return []

    hashed = [r for r in all_records if r.get("phash")]
    pairs = []
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            h1 = imagehash.hex_to_hash(hashed[i]["phash"])
            h2 = imagehash.hex_to_hash(hashed[j]["phash"])
            distance = h1 - h2
            if distance <= max_distance:
                pairs.append((hashed[i]["path"], hashed[j]["path"], distance))
    return pairs


def main():
    random.seed(42)

    dataset_has_images = any(find_images(os.path.join(DATASET_PATH, c)) for c in CLASSES)
    if not dataset_has_images:
        local_zips = sorted(Path(".").glob("*.zip"))
        if local_zips:
            import zipfile
            for zip_path in local_zips:
                print(f"Found local zip: {zip_path} -- extracting into {DATASET_PATH}/ ...")
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(DATASET_PATH)
        else:
            folder_url = GDRIVE_FOLDER_URL
            if not folder_url:
                folder_url = input(
                    "No images found in dataset/. Paste a Google Drive link "
                    "to download it (or press Enter to skip): "
                ).strip()
            if folder_url:
                download_dataset_from_drive(folder_url, DATASET_PATH)

    print("=" * 40)
    print("DATASET VALIDATION")
    print("=" * 40)
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    records_by_class = {}
    all_records = []

    for class_name in CLASSES:
        class_folder = os.path.join(DATASET_PATH, class_name)
        image_paths = find_images(class_folder)

        records = [inspect_image(p) for p in image_paths]
        records_by_class[class_name] = records
        all_records.extend(records)

        valid_count = sum(1 for r in records if r["is_valid"])
        print(f"{class_name.capitalize()}: {len(records)} images ({valid_count} valid)")

    total_images = len(all_records)
    valid_images = [r for r in all_records if r["is_valid"]]
    invalid_images = [r for r in all_records if not r["is_valid"]]

    exact_dupes = find_exact_duplicates(all_records)
    similar_dupes = find_similar_duplicates(all_records)

    print()
    print(f"Valid images: {len(valid_images)}")
    print(f"Invalid images: {len(invalid_images)}")
    print(f"Exact duplicates (groups): {len(exact_dupes)}")
    if HAVE_IMAGEHASH:
        print(f"Possible visually-similar pairs: {len(similar_dupes)}")
    else:
        print("Possible visually-similar pairs: skipped (install 'imagehash' to enable)")

    print()
    print("=" * 40)
    print("DIMENSIONS")
    print("=" * 40)
    dim_stats_by_class = {}
    for class_name in CLASSES:
        stats = summarize_dimensions(records_by_class[class_name])
        dim_stats_by_class[class_name] = stats
        print()
        print(f"{class_name.capitalize()}:")
        if stats is None:
            print("  No valid images.")
            continue
        print(f"  Min:     {stats['min_width']}x{stats['min_height']}")
        print(f"  Max:     {stats['max_width']}x{stats['max_height']}")
        print(f"  Average: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}")
        print(f"  Unusual (< {MIN_REASONABLE_DIMENSION}px on a side): {stats['unusual_count']}")

    print()
    print("=" * 40)
    print("CLASS BALANCE")
    print("=" * 40)
    print()
    counts = {c: len(records_by_class[c]) for c in CLASSES}
    print(f"{'Class':<10} Number of images")
    for c in CLASSES:
        print(f"{c.capitalize():<10} {counts[c]}")
    min_count = min(counts.values())
    max_count = max(counts.values())
    print()
    print(f"Min class size: {min_count}")
    print(f"Max class size: {max_count}")
    print(f"Difference:     {max_count - min_count}")

    print()
    print("=" * 40)
    print("PROBLEM FILES (reported only -- not moved)")
    print("=" * 40)
    if invalid_images:
        for r in invalid_images:
            print(f"  {r['path']}  -- {r['error']}")
        print()
        print(f"To move these into {PROBLEMS_DIR}, review the list above and move them")
        print("manually, or re-run with a --move-problems flag if you add one.")
    else:
        print("  None found.")

    csv_path = os.path.join(RESULTS_DIR, "dataset_metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "class", "width", "height", "format",
            "mode", "file_size", "is_valid", "hash"
        ])
        for class_name in CLASSES:
            for r in records_by_class[class_name]:
                filename = os.path.basename(r["path"])
                writer.writerow([
                    filename,
                    class_name,
                    r["width"],
                    r["height"],
                    r["format"],
                    r["mode"],
                    r["file_size"],
                    r["is_valid"],
                    r["md5_hash"],
                ])
    print()
    print(f"Metadata CSV written to: {csv_path}")

    report_path = os.path.join(RESULTS_DIR, "dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("DATASET SUMMARY REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total images scanned: {total_images}\n")
        f.write(f"Valid images: {len(valid_images)}\n")
        f.write(f"Corrupted/invalid images: {len(invalid_images)}\n\n")

        f.write("Images per class:\n")
        for c in CLASSES:
            f.write(f"  {c.capitalize():<10} {counts[c]}\n")
        f.write(f"\nClass balance -- min: {min_count}, max: {max_count}, diff: {max_count - min_count}\n\n")

        f.write("Corrupted images:\n")
        if invalid_images:
            for r in invalid_images:
                f.write(f"  {r['path']} -- {r['error']}\n")
        else:
            f.write("  None\n")
        f.write("\n")

        f.write(f"Exact duplicate groups: {len(exact_dupes)}\n")
        for h, paths in exact_dupes.items():
            f.write(f"  hash {h}:\n")
            for p in paths:
                f.write(f"    {p}\n")
        f.write("\n")

        if HAVE_IMAGEHASH:
            f.write(f"Possible visually-similar pairs (Hamming distance <= 5): {len(similar_dupes)}\n")
            for p1, p2, dist in similar_dupes:
                f.write(f"  {p1}  <->  {p2}  (distance={dist})\n")
        else:
            f.write("Visually-similar duplicate check skipped (install 'imagehash' to enable)\n")
        f.write("\n")

        f.write("Dimension statistics per class:\n")
        for c in CLASSES:
            stats = dim_stats_by_class[c]
            f.write(f"  {c.capitalize()}:\n")
            if stats is None:
                f.write("    No valid images.\n")
                continue
            f.write(f"    Min:     {stats['min_width']}x{stats['min_height']}\n")
            f.write(f"    Max:     {stats['max_width']}x{stats['max_height']}\n")
            f.write(f"    Average: {stats['avg_width']:.0f}x{stats['avg_height']:.0f}\n")
            f.write(f"    Unusual (< {MIN_REASONABLE_DIMENSION}px on a side): {stats['unusual_count']}\n")
        f.write("\n")

        format_counts = {}
        mode_counts = {}
        for r in valid_images:
            format_counts[r["format"]] = format_counts.get(r["format"], 0) + 1
            mode_counts[r["mode"]] = mode_counts.get(r["mode"], 0) + 1

        f.write("File format statistics:\n")
        for fmt, count in sorted(format_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {fmt}: {count}\n")
        f.write("\n")

        f.write("Color mode statistics:\n")
        for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {mode}: {count}\n")

    print(f"Text report written to:  {report_path}")

    preview_path = os.path.join(RESULTS_DIR, "dataset_preview.jpg")
    build_contact_sheet(records_by_class, preview_path)
    print(f"Preview contact sheet:   {preview_path}")

    eda_path = os.path.join(RESULTS_DIR, "dataset_eda.png")
    build_eda_plots(records_by_class, eda_path)
    print(f"EDA plots:                {eda_path}")

    print()
    print("=" * 40)
    print("DONE")
    print("=" * 40)


if __name__ == "__main__":
    main()
