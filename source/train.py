""" we will be training the classifier first with VGG16 frozen
then unfreeze the model and fine tune with lr = 0.0001"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import (CSVLogger, EarlyStopping,ModelCheckpoint,ReduceLROnPlateau)
from config import (CHECKPOINT_DIR, FINE_TUNE_EPOCHS, FINE_TUNE_LR, EPOCHS, HEAD_LR,
                    METRICS_DIR,MODELS_DIR, PLOTS_DIR, SEED, make_dirs)

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

def main(args):
    make_dirs()
    tf.keras.utils.set_random_seed(SEED)

    train_gen, val_gen, test_gen, class_names = make_generators()
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names))
    print("Classes:", class_names)
    print(f"Images -> train: {len(train_gen.paths)}, val: {len(val_gen.paths)}, "
          f"test: {len(test_gen.paths)}")

    weights = None if args.weights == "none" else args.weights
    model, base = build_model(len(class_names), weights = weights)

    class_weight = get_class_weights(train_gen) if args.class_weights else None
    if class_weight:
        print("Class weights:", {class_names[i]: round(w, 3) for i, w in class_weight.items()})

    epochs = 1 if args.quick else args.head_epochs
    ft_epochs = 1 if args.quick else args.fine_tune_epochs


### this part covers head only 

    print(f"\n=== Phase 1: training the classifier head ({epochs} epochs) ===")
    compile_model(model, HEAD_LR)
    h1 = model.fit(train_gen, validation_data = val_gen, epochs = epochs, callbacks = make_callbacks(1)
                   , class_weight = class_weight).history

### this part covers to fine tune block 5
    unfrozen = unfreeze_block(base)
    print(f"\n=== Phase 2: fine-tuning {len(unfrozen)} VGG16 layers ({ft_epochs} epochs) ===")
    compile_model(model, FINE_TUNE_LR)  # must recompile after changing trainable layers
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=ft_epochs,
                   callbacks=make_callbacks(2), class_weight=class_weight).history

    final_path = MODELS_DIR / "brain_tumor_vgg16.keras"
    model.save(final_path)
    (METRICS_DIR / "history.json").write_text(json.dumps({"phase1": h1, "phase2": h2}, indent=2))
    plot_history(h1,h2,PLOTS_DIR / "training_curves.png")

    print(f"\nSaved model:  {final_path}")
    print(f"Saved curves: {PLOTS_DIR / 'training_curves.png'}")
    print(f"Best val accuracy: phase 1 {max(h1['val_accuracy']):.4f}, "
          f"phase 2 {max(h2['val_accuracy']):.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="1 epoch per phase (smoke test)")
    p.add_argument("--head-epochs", type=int, default=EPOCHS)
    p.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    p.add_argument("--class-weights", action="store_true")
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                   help="'none' skips the ImageNet download (testing only)")
    main(p.parse_args())