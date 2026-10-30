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

