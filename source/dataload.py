import os 
import numpy as np
from PIL import Image, ImageEnhance
import random
import matplotlib.pyplot as plt

#keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from sklearn.utils import shuffle


#data set load 
train_path = "dataset/training"
labels = os.listdir(train_path)

print(labels)

train_img = []
train_label = []

for label in labels:

    for image_file in os.listdir(os.path.join(train_path,label)):

        train_img.append(os.path.join(train_path,label,image_file))
        train_label.append(label)

train_img, train_label = shuffle(train_img, train_label,)

for i in range(5):

    image = Image.open(train_img[i])

    plt.subplot(1, 5, i+1)
    plt.imshow(image)
    plt.title(f"Label: {train_label[i]}")
    plt.axis("off")

plt.show()