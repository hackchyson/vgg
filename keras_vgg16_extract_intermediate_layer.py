from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# load image
img_path = 'data/photo/b.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

img_path_bim = 'data/bim/1.png'
img_bim = image.load_img(img_path_bim, target_size=(224, 224))
b = image.img_to_array(img_bim)
b = np.expand_dims(b, axis=0)
b = preprocess_input(b)

# vgg16
base_model = VGG16(weights='imagenet', include_top=False, pooling='max')


# visualization
def visualize_feature_map(img_batch):
    feature_map = np.squeeze(img_batch, axis=0)
    # print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.show()


for i in range(1, 6):
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block' + str(i) + '_pool').output)
    block_pool_features = model.predict(x)
    block_pool_features_bim = model.predict(b)
    # print(block_pool_features.shape, block_pool_features_bim.shape)

    visualize_feature_map(block_pool_features)
    visualize_feature_map(block_pool_features_bim)

    # cosine similarity

    cosine_dis = cosine_similarity(block_pool_features.reshape(1, -1), block_pool_features_bim.reshape(1, -1))
    print(cosine_dis)
    # cosine_dis = cosine_similarity(block_pool_features[0, :, :, 1], block_pool_features_bim[0, :, :, 1])
    # print(block_pool_features[0, :, :, 1].shape, block_pool_features_bim[0, :, :, 1].shape)
    # print(cosine_dis.shape)
