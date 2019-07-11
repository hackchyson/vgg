from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.applications.vgg16 import preprocess_input
import cv2 as cv
from utils.mix import mix_feature


class ImageNetVGG16:
    def __init__(self, target_size=(224, 224)):
        """
        VGG16 model.

        :param target_size: The input image size.
        """
        self.target_size = target_size
        self.base_model = VGG16(weights='imagenet', include_top=False, pooling='max')

    def predict_img(self, img_path, layer=4):
        """
        Extract feature map from image read with Keras.

        :param img_path: Image path.
        :param layer: From which layer to extract feature map.

        :return: The feature map extracted.
        """
        model = Model(inputs=self.base_model.input,
                      outputs=self.base_model.get_layer('block' + str(layer) + '_pool').output)
        img = image.load_img(img_path, target_size=self.target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x)

    def extract_feature(self, img_path, layer=4):
        """
        Extract feature map from image read with cv2.

        :param img_path: Image path.
        :param layer: From which layer to extract feature map.

        :return: The feature map extracted.
        """
        model = Model(inputs=self.base_model.input,
                      outputs=self.base_model.get_layer('block' + str(layer) + '_pool').output)
        img = cv.imread(img_path)
        img = cv.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return model.predict(x)

    def extract_features(self, npy_path, layer=4):
        """

        :param npy_path:
        :param layer: From which layer to extract feature map.
        :return:
        """
        features = list()
        images = np.load(npy_path)
        model = Model(inputs=self.base_model.input,
                      outputs=self.base_model.get_layer('block' + str(layer) + '_pool').output)
        for i in images:
            x = image.img_to_array(i)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            feature = mix_feature(feature)
            features.append(feature)
        features = np.array(features)
        return features


