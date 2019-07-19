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
        self.WHOLE = 3 / 4
        self.LOCAL = 1 / 4

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

    def combined_features(self, npy_path, whole_feature=4, local_feature=2):
        features = list()
        images = np.load(npy_path)
        model_whole = Model(inputs=self.base_model.input,
                            outputs=self.base_model.get_layer('block' + str(whole_feature) + '_pool').output)
        model_local = Model(inputs=self.base_model.input,
                            outputs=self.base_model.get_layer('block' + str(local_feature) + '_pool').output)
        for i in images:
            x = image.img_to_array(i)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature_whole = model_whole.predict(x)
            feature_local = model_local.predict(x)

            feature_whole = mix_feature(feature_whole)
            feature_local = mix_feature(feature_local)
            feature_local = cv.resize(feature_local, feature_whole.shape)
            feature = self.WHOLE * feature_whole + self.LOCAL * feature_local
            features.append(feature)
        features = np.array(features)
        return features

    def extract_all_features(self, npy_path, layer=4):
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
            feature = np.squeeze(feature, axis=0)
            features.append(feature)
        features = np.array(features)
        return features


if __name__ == "__main__":
    vgg = ImageNetVGG16()
    feature = vgg.predict_img('/home/hack/PycharmProjects/vgg/data/paper/bim1.png', layer=2)
    import matplotlib.pyplot as plt

    print(feature.shape)
    feature = mix_feature(feature)
    print(feature.shape)
    import cv2 as cv

    plt.imshow(feature)
    plt.title("layer 2")

    feature = cv.resize(feature, (14, 14))
    plt.figure(2)
    plt.imshow(feature)
    plt.title("layer 4 resize to layer 2")

    plt.figure(3)
    feature2 = vgg.predict_img('/home/hack/PycharmProjects/vgg/data/paper/bim1.png', layer=4)
    feature2 = mix_feature(feature2)
    plt.imshow(feature2)
    plt.title('layer 4')

    plt.figure(4)
    feature3 = (1 / 4 * feature + 3 / 4 * feature2) / 4
    plt.imshow(feature3)
    plt.title('layer 4 and layer 2')

    plt.show()
