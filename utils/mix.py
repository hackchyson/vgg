import numpy as np


def mix_feature(img_batch):
    # squeeze:     Remove single-dimensional entries from the shape of an array.
    feature_map = np.squeeze(img_batch, axis=0)
    # print(feature_map.shape)

    feature_map_combination = []  # a list to contain all the sub feature maps

    num_pic = feature_map.shape[2]  # number of feature maps

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]  # get each feature map
        feature_map_combination.append(feature_map_split)

    # Just sum up all the feature maps in 1:1 ratio
    feature_map_sum = sum(ele for ele in feature_map_combination)
    return feature_map_sum
