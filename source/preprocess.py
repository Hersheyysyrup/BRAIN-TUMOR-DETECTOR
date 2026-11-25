"""Preprocessing, augmentation and batch loading for VGG16
Image range from 150 to 1275 px needs to b resized 
 some images are RGB and some are Grayscale """

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from keras.applications.vgg16 import preprocess_input
from keras.utils import Sequence

from config import (BATCH_SIZE, IMG_SIZE, PLOTS_DIR, SEED, TEST_DIR, TRAIN_DIR, VAL_SIZE, make_dirs)
from data import load_paths, split_train_val

### to process a single image 

def augment_image(img):
    """Augment a single image with random transformations"""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img= ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    shift = IMG_SIZE * 0.05
    img = img.rotate(
        random.uniform(-10, 10),
        translate = (random.uniform(-shift, shift), random.uniform(-shift, shift)),
        resample = Image.BILINEAR,
    )

    return img

def load_image(path):
    with Image.open(path) as img:
        return img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)


def to_model_input(img):
    """ to convert imagfe into float 32 for VGG16 to accept"""
    return preprocess_input(np.asarray(img, dtype = "float32").copy())

def load_And_preprocess(path, augment = False):
    img = load_image(path)
    if augment:
        img = augment_image(img)
    return to_model_input(img)

### Batch Loading 

class MRISequence(Sequence):

    def __init__(self, paths, labels, class_names, augment = False,
                 shuffle = False, batch_size = BATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.paths = np.array(paths)
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.y = np.array([self.class_names.index(l) for l in labels])
        self.augment = augment
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.rng = np.random.default_rng(SEED)
        self.indices = np.arange(len(self.paths))
        if shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.paths)/ self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx+1) * self.batch_size]
        X = np.stack([load_And_preprocess(self.paths[i], self.augment) for i in batch_idx])
        Y = np.eye(self.num_classes, dtype = "float32")[self.y[batch_idx]]
        return X,Y


    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

def make_generators():
    """Make training and validation generators"""
    (all_train, all_labels), class_names = load_paths(TRAIN_DIR)
    (test_paths, test_labels), _ = load_paths(TEST_DIR, shuffle_data = False)
    tr_x, val_x, tr_y, val_y = split_train_val(all_train, all_labels, VAL_SIZE)

    train_gen = MRISequence(tr_x, tr_y, class_names, augment = True, shuffle = True)
    val_gen = MRISequence(val_x, val_y, class_names)
    test_gen = MRISequence(test_paths, test_labels, class_names)
    return train_gen, val_gen, test_gen, class_names

### Sanity check
def _augmentation_preview(path, n=5):
    make_dirs()
    original = load_image(path)
    fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 3.2))
    axes[0].imshow(original)
    axes[0].set_title("original")
    for i in range(1, n + 1):
        axes[i].imshow(augment_image(original))
        axes[i].set_title(f"augmented {i}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out = PLOTS_DIR / "augmentation_preview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("Saved augmentation preview:", out)
 
 
if __name__ == "__main__":
    train_gen, val_gen, test_gen, class_names = make_generators()
    print("Classes:", class_names)
    print(f"Batches -> train: {len(train_gen)}, val: {len(val_gen)}, test: {len(test_gen)}")
 
    X, y = train_gen[0]
    print("Batch X:", X.shape, X.dtype, f"| range [{X.min():.1f}, {X.max():.1f}]")
    print("Batch y:", y.shape, "| each row sums to", y.sum(axis=1)[:3])
 
    _augmentation_preview(train_gen.paths[0])