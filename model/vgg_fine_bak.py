from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import get_file
import keras.losses
import keras.backend as K

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
weights = 'imagenet'
localization = 6
beta = 600
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block1_conv1',
                 input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block2_conv1'))
model.add(Conv2D(128, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block3_conv1'))
model.add(Conv2D(256, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block3_conv2'))
model.add(Conv2D(256, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block4_conv1'))
model.add(Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block4_conv2'))
model.add(Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block5_conv1'))
model.add(Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block5_conv2'))
model.add(Conv2D(512, (3, 3),
                 activation='relu',
                 padding='same',
                 name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

#
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dense(localization, activation='none', name='predictions'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)


def local_loss(y_true, y_pred):
    position = K.sqrt(K.square(y_pred[:, :3] - y_true[:, :3]))
    orientation = beta * K.sqrt(K.square(y_pred[:, 3:] - y_true[:, 3:] / K.abs(y_true[:, 3:])))
    return K.mean(position + orientation, axis=-1)


model.compile(loss=local_loss, optimizer=sgd)

model.fit(x_train, y_train, batch_size=32, epochs=10)

# Load weights.
weights_path = get_file(
    'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
    WEIGHTS_PATH_NO_TOP,
    cache_subdir='models',
    file_hash='6d6bbae143d832006294945121d1f1fc')
model.load_weights(weights_path)
