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
        resample = Image.BILENIAR,
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
