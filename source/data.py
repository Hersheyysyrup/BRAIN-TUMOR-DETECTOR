import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle

IMG_FORMATS = (".jpg", ".jpeg", ".png")


### access the root folders for types of tumor 
def load_paths(root):
    labels = sorted(d for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d)))


### creating empty lists for the class and shuffle them 
    paths , y = [], []
    for label in labels:
        folder = os.path.join(root, label)
        for f in os.listdir(folder):
            if f.lower().endswith(IMG_FORMATS):
                paths.append(os.path.join(folder, f))
                y.append(label
                )

    return shuffle(paths, y, random_state = 42), labels

(train_img, train_label), class_names = load_paths("dataset/Training")
(test_img, test_label), _ = load_paths("dataset/Testing")

print("classes:", class_names)
print("Train:", len(train_img), Counter(train_label))
print("Test:", len(test_img), Counter(test_label))

###figure plot ###
plt.figure(figsize = (15,4))
for i in range (5):
    plt.subplot(1,5,i + 1)
    plt.imshow(Image.open(train_img[i], cmap = "gray"))
    plt.title(train_label[i])
    plt.axis("off")

plt.tight_layout()
plt.show()