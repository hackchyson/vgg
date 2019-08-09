from model.vgg_fine import VGG16Tune
import numpy as np
from config.config import *
from os import path
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# load numpy data
x = np.load(X_PATH)
y = np.load(Y_PATH)
print('x shape: {}; y shape: {}'.format(x.shape, y.shape))

if TRAINING:
    model = VGG16Tune()
    print("[INFO] training head...")
    check_point = ModelCheckpoint(MODEL_FILE, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [check_point]
    # history = model.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=10, callbacks=callback_list, verbose=1)
    history = model.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=10, verbose=1)

    # accuracy plot
    plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model.save(MODEL_FILE)
else:
    model = load_model(MODEL_FILE)
