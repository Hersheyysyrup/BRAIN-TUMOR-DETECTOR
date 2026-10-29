import os
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from pathlib import Path

IMG_FORMATS = (".jpg", ".jpeg", ".png")
SEED = 42

### project root and dataset folders

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
TRAIN_DIR = DATASET / "Training"
TEST_DIR = DATASET / "Testing"


def load_paths(root, shuffle_data=True):
    """Return paths, labels, class names for a folder with one subfolder per class."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Folder not found: {root}\n"
            f"Check the dataset is extracted so that this path exists."
        )

    class_names = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"No class subfolders found inside {root}")

    paths, labels = [], []
    for label in class_names:
        for f in sorted((root / label).iterdir()):
            if f.suffix.lower() in IMG_FORMATS:
                paths.append(str(f))
                labels.append(label)

    if shuffle_data:
        paths, labels = shuffle(paths, labels, random_state=SEED)
    return (paths, labels), class_names


def split_train_val(paths, labels, val_size=0.15):
    return train_test_split(
        paths, labels, test_size=val_size, stratify=labels, random_state=SEED
    )


def show_samples(paths, labels, n=5):
    plt.figure(figsize=(15, 4))
    for i in range(min(n, len(paths))):
        plt.subplot(1, n, i + 1)
        with Image.open(paths[i]) as img:
            plt.imshow(img.convert("L"), cmap="gray")
        plt.title(labels[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    (train_img, train_label), class_names = load_paths(TRAIN_DIR)
    (test_img, test_label), _ = load_paths(TEST_DIR, shuffle_data=False)

    tr_x, val_x, tr_y, val_y = split_train_val(train_img, train_label)

    print("Classes:", class_names)
    print("Train:", len(tr_x), Counter(tr_y))
    print("Val:  ", len(val_x), Counter(val_y))
    print("Test: ", len(test_img), Counter(test_label))

    show_samples(tr_x, tr_y)