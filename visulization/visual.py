import matplotlib.pyplot as plt
import numpy as np
from pylab import axis


def vis_into_one(img_batch):
    # squeeze:     Remove single-dimensional entries from the shape of an array.
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


def get_row_col(num_pic):
    root = num_pic ** 0.5
    row = round(root)
    col = row + 1 if root - row > 0 else row
    return row, col


def vis_all(img_batch):
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
