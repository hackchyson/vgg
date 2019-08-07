from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.applications.vgg16 import preprocess_input
import cv2 as cv
from utils.mix import mix_feature
from keras.models import Input
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import get_file
import keras.losses
import keras.backend as K
from config.config import BETA


def VGG16Tune():
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    head_model = base_model.output
    print(head_model.shape)
    head_model = Flatten(name="flatten")(head_model)
    head_model = Dense(4096, activation='relu', name='fc1')(head_model)
    head_model = Dense(4096, activation='relu', name='fc2')(head_model)
    head_model = Dense(6, activation=None, name='fc3')(head_model)

    model = Model(inputs=base_model.input, outputs=head_model)

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in base_model.layers:
        layer.trainable = False

    def local_loss(y_true, y_pred):
        position = K.sqrt(K.square(y_pred[:, :3] - y_true[:, :3]))
        orientation = BETA * K.sqrt(K.square(y_pred[:, 3:] - y_true[:, 3:] / K.abs(y_true[:, 3:])))
        return K.mean(position + orientation, axis=-1)

    opt = SGD(lr=1e-4, momentum=.9)
    model.compile(loss=local_loss, optimizer=opt, metrics=['accuracy'])

    return model
