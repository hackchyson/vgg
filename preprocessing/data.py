import os
import cv2 as cv
import logging
import numpy as np
import pickle
from config.config import IMG_SIZE, LEVEL, SEPARATOR
import logging
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

logging.basicConfig(level=LEVEL)


def save(path, np_path, label_path, filename_path, dsize=(224, 224)):
    """
    Scan the path, convert all images into a numpy object and save it in npy format.
    Save labels that is a list in pickle format.

    :param path: Path is scan.
    :param np_path: numpy object path that will be saved.
    :param label_path: pickle file path.
    :param filename_path: A file that contains a list to store all the BIM model capture files.
    :return:
    """
    imgs = list()
    labels = list()
    files = list()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png'):
                filepath = os.path.join(dirpath, filename)
                img = cv.imread(filepath)
                img = cv.resize(img, dsize)
                pos_dir = filename.rstrip('.png').split('--')
                position = pos_dir[0]
                direction = pos_dir[1]
                label = "position: {};   direction: {};".format(position, direction)
                imgs.append(img)
                labels.append(label)
                logging.info(label)
                files.append(filepath)

    imgs = np.array(imgs)
    np.save(np_path, imgs)
    with open(label_path, 'wb') as file:
        pickle.dump(labels, file)
    with open(filename_path, 'wb') as file:
        pickle.dump(files, file)


def save_for_train(path, np_path, label_path, dsize=IMG_SIZE, separator=SEPARATOR):
    """
    Scan the path, convert all images into a numpy object and save it in npy format.
    Save labels that is a list in pickle format.

    :param path: Path is scan.
    :param np_path: numpy object path that will be saved.
    :param label_path: pickle file path.
    :param dsize: image size.
    :param separator: separator between position and direction
    :return:
    """
    imgs = list()
    labels = list()
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png'):
                filepath = os.path.join(dirpath, filename)
                img = cv.imread(filepath)
                img = cv.resize(img, dsize)
                arr = image.img_to_array(img)
                imgs.append(arr)

                pos_dir = filename.rstrip('.png').split(separator)
                logging.debug(pos_dir)
                position = eval(pos_dir[0])  # convert tuple representation string into tuple
                direction = eval(pos_dir[1])
                logging.debug('position: {}; direction: {}.'.format(position, direction))
                label = list(position) + list(direction)
                logging.debug(label)
                labels.append(label)

    imgs = np.array(imgs)
    imgs = preprocess_input(imgs)
    labels = np.array(labels)
    np.save(np_path, imgs)
    np.save(label_path, labels)


def load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    # save('/home/hack/PycharmProjects/vgg/data/bim/paper',
    #      '/home/hack/PycharmProjects/vgg/data/bim/paper/data.npy',
    #      '/home/hack/PycharmProjects/vgg/data/bim/paper/label.pickle',
    #      '/home/hack/PycharmProjects/vgg/data/bim/paper/bims.pickle')
    save_for_train('/home/hack/bim/photo',
                   '/home/hack/PycharmProjects/vgg/data/bim/1008/data',
                   '/home/hack/PycharmProjects/vgg/data/bim/1008/label')
