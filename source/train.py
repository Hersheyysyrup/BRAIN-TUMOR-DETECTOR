""" we will be training the classifier first with VGG16 frozen
then unfreeze the model and fine tune with lr = 0.0001"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import (CSVLogger, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau)
from config import (CHECKPOINT_DIR, FINE_TUNE, FINE_TUNE_LR, HEAD_EPOCHS, HEAD_LR,
                    METRICS_DIR, PLOTS_DIR, SEED, make_dirs)

from model import build_model, compile_model, unfreeze_block
from preprocess import make_generators

def get_class_weights(train_gen):
    counts = np.bincount(train_gen.y, minlength = train_gen.num_classes)
    weights = counts.sum() / (train_gen.num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}

def make_callbacks(phase):
    return [
        EarlyStopping(monitor ="val_loss", patience = 5, restore_best_weights = True),
        ModelCheckpoint(CHECKPOINT_DIR / f"phase{phase}_best.keras", monitor = "val_loss", save_best_only = True),
         ReduceLROnPlateau(monitor = "val_loss", factor = 0.3, patience = 2, min_lr = 1e-7),
         CSVLogger(METRICS_DIR / f"training_log_phase{phase}.csv")
    ]

def plot_history(h1,h2, out):
    def merged(key):
        return h1[key] + h2[key]

    split = len(h1["loss"]) - 0.5
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))
    for ax , metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(merged(metric), label="train")
        ax.plot(merged(f"val_{metric}"), label="validation")
        ax.axvline(split, color="gray", linestyle="--", label="fine-tuning starts")
        ax.set_title(plt.title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)