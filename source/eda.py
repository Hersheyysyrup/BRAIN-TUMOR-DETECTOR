### preforming eda here on class balance, image size and sample grids

import argparse
import json
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from config import METRICS_DIR, PLOTS_DIR, SEED, TEST_DIR, TRAIN_DIR, make_dirs
from data import load_paths

SAMPLE_SIZE = 600 

def class_distribution(train_label, test_label, class_names, show):
    train_counts = Counter(train_label)
    test_counts = Counter (test_label)
    tr = [train_counts[c] for c in class_names] ### Take the first class
    te = [test_counts[c] for c in class_names]  ##find the image count and add that count to this new list

    x = np.arange(len(class_names))
    fig, ax = plt.subplot(figsize = (8,4.5))
    b1 = ax.bar(x - 0.2, tr, 0.4, label="Training") ###creating bars
    b2 = ax.bar(x + 0.2, te, 0.4, label="Testing")
    ax.bar_label(b1)
    ax.bar_label(b2)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_class_distribution.png", dpi=150)
    plt.show() if show else plt.close(fig)

    ratio = max(tr) / min(tr) #ratio between the largest and smallest training classes.
    """An imbalance ratio of 1.21 means the largest class has approximately 1.21 times as many images as the smallest class."""
    print("\nClass counts (train): ", dict(zip(class_names, tr)))
    print("Class counts (test): ", dict(zip(class_names, te)))
    print(f"Imbalance ratio (largest/smallest train class): {ratio:.2f}")
    return {"train": dict(zip(class_names, tr)),
            "test": dict(zip(class_names, te)),
            "imbalance_ratio": round(ratio, 3)}

def image_stats(paths):
    rng = random.Random(SEED)
    sample = rng.sample(paths, min(SAMPLE_SIZE, len(paths)))
    sizes, modes, means = [], Counter(), []
    for p in sample:
        with Image.open(p) as img:
            sizes.append(img.size)  # (width, height)
            modes[img.mode] += 1
            means.append(np.asarray(img.convert("L"), dtype="float32").mean())
 
    widths = np.array([s[0] for s in sizes])
    heights = np.array([s[1] for s in sizes])
    size_counts = Counter(sizes).most_common(5)
 
    print(f"\nImage stats (sample of {len(sample)} training images):")
    print(f"  Width : min {widths.min()}, max {widths.max()}, mean {widths.mean():.0f}")
    print(f"  Height: min {heights.min()}, max {heights.max()}, mean {heights.mean():.0f}")
    print(f"  Most common sizes: {size_counts}")
    print(f"  Colour modes: {dict(modes)}")
    print(f"  Mean grey intensity: {np.mean(means):.1f} (0-255)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(widths, bins=20)
    axes[0].set_title("Image widths")
    axes[1].hist(heights, bins=20)
    axes[1].set_title("Image heights")
    axes[2].hist(means, bins=20)
    axes[2].set_title("Mean intensity per image")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_image_stats.png", dpi=150)
    plt.close(fig)
 
    return {
        "sampled": len(sample),
        "width": {"min": int(widths.min()), "max": int(widths.max()), "mean": float(widths.mean())},
        "height": {"min": int(heights.min()), "max": int(heights.max()), "mean": float(heights.mean())},
        "top_sizes": [{"size": list(s), "count": c} for s, c in size_counts],
        "modes": dict(modes),
        "mean_intensity": float(np.mean(means)),
    }
 
 
def sample_grid(paths, labels, class_names, per_class=5, show=False):
    rng = random.Random(SEED)
    fig, axes = plt.subplots(len(class_names), per_class,
                             figsize=(2.6 * per_class, 2.6 * len(class_names)))
    for r, cls in enumerate(class_names):
        cls_paths = [p for p, l in zip(paths, labels) if l == cls]
        for c, p in enumerate(rng.sample(cls_paths, min(per_class, len(cls_paths)))):
            with Image.open(p) as img:
                axes[r, c].imshow(img.convert("L"), cmap="gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(cls, loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_sample_grid.png", dpi=150)
    plt.show() if show else plt.close(fig)
 
 
def main(show=False):
    make_dirs()
    (train_img, train_label), class_names = load_paths(TRAIN_DIR)
    (test_img, test_label), _ = load_paths(TEST_DIR, shuffle_data=False)
 
    summary = {"classes": class_names}
    summary["distribution"] = class_distribution(train_label, test_label, class_names, show)
    summary["images"] = image_stats(train_img)
    sample_grid(train_img, train_label, class_names, show=show)
 
    out = METRICS_DIR / "eda_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved plots to {PLOTS_DIR}")
    print(f"Saved summary to {out}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="open plot windows")
    main(show=parser.parse_args().show)