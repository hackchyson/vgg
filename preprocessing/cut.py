import cv2 as cv
import logging

logging.basicConfig(level=logging.INFO)


def cut_center(image, x, y):
    shape_x = image.shape[0]
    shape_y = image.shape[1]
    new_x = (shape_x - x) // 2
    new_y = (shape_y - y) // 2
    logging.info("new_x: {}; new_y: {}".format(new_x, new_y))
    logging.info(image.shape)
    dst = image[new_x:new_x + x, new_y:new_y + y, :]
    return dst


if __name__ == "__main__":
    img = cv.imread('/home/hack/PycharmProjects/vgg/data/test.png')
    dst = cut_center(img, 1000, 1000)
    cv.imshow("dst", dst)
    cv.waitKey()
