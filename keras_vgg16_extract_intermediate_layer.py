from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from pylab import axis
from sklearn.metrics.pairwise import cosine_similarity

##################################
# load image
img_path = 'data/photo/c.jpg'
img = image.load_img(img_path, target_size=(224, 224))

plt.imshow(img)
plt.show()

x = image.img_to_array(img)
# print(x.shape)
x = np.expand_dims(x, axis=0)
# print(x.shape)
# print(x[0, 0, 0, :])

x = preprocess_input(x)
# print(x.shape)
# print(x[0, 0, 0, :])

img_path_bim = 'data/bim/1.png'
img_bim = image.load_img(img_path_bim, target_size=(224, 224))
b = image.img_to_array(img_bim)
b = np.expand_dims(b, axis=0)
b = preprocess_input(b)  # 图片矩阵中的值进行了更改


# [37. 30. 24.] --> [-79.939 -86.779 -86.68 ]

#####################################
# visualization
def visualize_feature_map(img_batch):
    # squeeze:     Remove single-dimensional entries from the shape of an array.
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)

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


def get_row_col(num_pic):
    root = num_pic ** 0.5
    row = round(root)
    col = row + 1 if root - row > 0 else row
    return row, col


def visualize_feature_maps(img_batch):
    # squeeze:     Remove single-dimensional entries from the shape of an array.
    feature_map = np.squeeze(img_batch, axis=0)
    print(feature_map.shape)

    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    for i in range(num_pic):
        feature_map_split = feature_map[:, :, i]
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        axis('off')
    plt.show()


###############################
# vgg16
base_model = VGG16(weights='imagenet', include_top=False, pooling='max')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
block_pool_features = model.predict(x)
# visualize_feature_maps(block_pool_features)
# visualize_feature_map(block_pool_features)

# for i in range(1, 6):
#     model = Model(inputs=base_model.input, outputs=base_model.get_layer('block' + str(i) + '_pool').output)
#     block_pool_features = model.predict(x)
#     block_pool_features_bim = model.predict(b)
#     # print(block_pool_features.shape, block_pool_features_bim.shape)
#
#     visualize_feature_map(block_pool_features)
#     visualize_feature_map(block_pool_features_bim)
#
#     # cosine similarity
#
#     cosine_dis = cosine_similarity(block_pool_features.reshape(1, -1), block_pool_features_bim.reshape(1, -1))
#     print(cosine_dis)
#     # cosine_dis = cosine_similarity(block_pool_features[0, :, :, 1], block_pool_features_bim[0, :, :, 1])
#     # print(block_pool_features[0, :, :, 1].shape, block_pool_features_bim[0, :, :, 1].shape)
#     # print(cosine_dis.shape)


##################################
# similarity
sim = cosine_similarity(block_pool_features.reshape(1, -1), block_pool_features.reshape(1, -1))
print(sim)
