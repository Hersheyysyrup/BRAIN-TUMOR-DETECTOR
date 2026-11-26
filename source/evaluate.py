"""To evaluate the trained model on a new test set"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model

from config import METRICS_DIR, MODELS_DIR, PLOTS_DIR, make_dirs
from preprocess import load_image, make_generators

def plot_confusion_matrix(cm, class_names, out, show=False):
    cm_norm = cm /cm.sum(axis =1, keepdims = True)
    fig, axes = plt.subplots(1,2, figsize = (13, 5.5))
    for ax, data, title , fmt in [(axes[0], cm, "Confusion matrix (counts)","d"),
                                  (axes[1], cm_norm, "Confusion matrix (row-normalised)", ".2f")]:
        im = ax.amshow(data, cmap = "Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(len(class_names)):
            for j in range (len(class_names)):
                ax.tests(j, i, format(data[i,j], fmt), ha = "center", va = "center", color = "white"
                     if data[i,j] > data.max() / 2 else "black")
        fig.colorbar(im, ax = ax, fraction = 0.046)
        fig.tight_layout()
        fig.savefig(out, dpi= 150)
        plt.show() if show else plt.close(fig)

def plot_misclassified(paths, y_true, y_pred, probs, class_names, out, n=12, show=False):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("No misclassified images, so no gallery was made.")
        return


    wrong = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:n]

    cols = 4
    rows = int(np.ceil(len(wrong) / cols ))  
    fig, axes = plt.subplots(rows,cols,figsize = (3.2 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        ax.imshow(load_image(paths[idx]))
        ax.set_title(f"true: {class_names[y_true[idx]]}\n"
                     f"pred: {class_names[y_pred[idx]]} ({probs[idx, y_pred[idx]]:.0%})",
                     fontsize=9, color="firebrick")

    fig.suptitle("Misclassified test images ( most confident mistakes)")
    fig.tight_layout()
    fig.savefig(out , dpi = 150)
    plt.show() if show else plt.close(fig)

def main(args):
    make_dirs()
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    model = load_model(args.model)
    _,_, test_gen, gen_classes = make_generators()
    assert gen_classes == class_names, "Class order differs from the one save at training time."

    probs = model.predict(test_gen, verbose =1 )
    y_pred = probs.argmax(axis =1 )
    y_true = test_gen.y ### this lines up with the predictions as text_gen is unshuffled

    accuracy = float((y_pred == y_true).mean())
    report_txt = classification_report(y_true, y_pred, target_names = class_names, digits = 4
                                       , zero_division = 0)
    report =classification_report(y_true, y_pred, target_names = class_names, output_dict = True, zero_divsion = 0 )

    cm = confusion_matrix(y_true, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f}  ({int((y_pred == y_true).sum())}/{len(y_true)})\n")
    print(report_txt)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm)

    (METRICS_DIR / "classification_report.txt").write_text(report_txt)
    (METRICS_DIR / "text_metrics.json").write_text(json.dumps({
        "model": str(args.model),
        "test_images" : int(len(y_true)),
        "accuracy": accuracy,
        "macro_f1": report["macro"]["f1-score"],
        "per_class": {c : report[c] for c in class_names},
        "confusion_matrix" : cm.tolist(),
    }, indent = 2))

    plot_confusion_matrix(cm, class_names, PLOTS_DIR/"confusion_matrix.png", args.show)
    plot_misclassified(test_gen.paths, y_true, y_pred, probs, class_names, PLOTS_DIR / "misclassified.png, show = args.show")
    print(f"\nSaved results to {METRICS_DIR} and {PLOTS_DIR}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODELS_DIR / "brain_tumor_vgg16.keras"))
    p.add_argument("--show", action="store_true")
    main(p.oarse_args())