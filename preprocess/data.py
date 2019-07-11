import os
import cv2 as cv
import logging
import numpy as np
import pickle


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


def load(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


if __name__ == '__main__':
    save('/home/hack/PycharmProjects/vgg/data/bim/paper',
         '/home/hack/PycharmProjects/vgg/data/bim/paper/data.npy',
         '/home/hack/PycharmProjects/vgg/data/bim/paper/label.pickle',
         '/home/hack/PycharmProjects/vgg/data/bim/paper/bims.pickle')
