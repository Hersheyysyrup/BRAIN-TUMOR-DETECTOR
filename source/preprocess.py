"""Preprocessing, augmentation and batch loading for VGG16
Image range from 150 to 1275 px needs to b resized 
 some images are RGB and some are Grayscale """

import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from keras.applications.vgg16 import preprocess_input
from keras.utils import Sequence
