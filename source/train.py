from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Sequential
from keras.optimizers import Adam

from config import IMG_SIZE

def build_model(num_models, img_size = IMG_SIZE, weights = "imagenet", dropout = 0.3):

    """downloads the VGG!^ models weights on the first use 
    and give as randomly initialized base """

    base = VGG16(include_top = False, weights = weights,
                 input_shape = (img_size, img_size, 3))

    base.trainable = False 
    """this freezes the VGG16 model weights and only allows the classifier to learn """

    model = Sequential([
        Input(shape=(img_size, img_size, 3)),
        base,
        Flatten(),
        Dropout(dropout),
        Dense(128, activation= "relu"),
        Dropout(dropout),
        Dense(num_models, activation = "softmax"),
    ], name = "brain_tumor_vgg16")

    return model, base

def compile_model(model, lr):
    model.compile(
        optimizer = Adam(learning_rate = lr),
        loss = "Categorical_crossentropy",
        metrics = ["accuracy"],
    )
    return model

def unfreeze_block(base, block= "block5"):
    """this makes only the 5th layer of vgg16 block trainable """
    base.trainable = True
    for layer in base.layers:
        layer.trainable = layer.name.startswith(block)
        return [l.name for l in base.layers if l.trainable]

if __name__ == "__main__":
    model, base = build_model(num_classes=4, weights=None)
    compile_model(model, 1e-3)
    model.summary()
    print("Trainable layers after unfreezing:", unfreeze_block(base))
