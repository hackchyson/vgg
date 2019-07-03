from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.applications.vgg16 import preprocess_input


class ImageNetVGG16:
    def __init__(self, layer=4, target_size=(224, 224)):
        self.layer = layer
        self.target_size = target_size
        self.base_model = VGG16(weights='imagenet', include_top=False, pooling='max')
        self.model = Model(inputs=self.base_model.input,
                           outputs=self.base_model.get_layer('block' + str(layer) + '_pool').output)

    def predict_img(self, img_path):
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return self.model.predict(x)

    def predict_mat(self, mat):
        return self.model.predict(mat)


