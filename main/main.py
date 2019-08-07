from model.vgg_fine import VGG16Tune
import numpy as np
from config.config import *
from os import path
from keras.models import load_model

# load numpy data
x = np.load(X_PATH)
y = np.load(Y_PATH)
print('x shape: {}; y shape: {}'.format(x.shape, y.shape))

if TRAINING:
    model = VGG16Tune()
    print("[INFO] training head...")
    model.fit(x, y, batch_size=BATCH_SIZE, nb_epoch=1, verbose=1)

    model.save(path.join(MODEL_PATH, 'bim.h5'))
else:
    model = load_model(path.join(MODEL_PATH, 'bim.h5'))
