import cv2 as cv


# img = cv.imread('/home/hack/PycharmProjects/vgg/data/test.png')
# shape = img.shape
# print(shape)
# point = (shape[0] // 2, shape[1] // 2)
# print(point)
# mat = cv.getRotationMatrix2D(point, 3, 1.0)
# print(img.shape[:2])
#
# dst = cv.warpAffine(img, mat, img.shape[:2])
# cv.imshow("src", img)
# cv.imshow('dst', dst)
# cv.waitKey()


def rotate_center(image, angle, scale=1):
    shape = image.shape
    center = (shape[0] // 2, shape[1] // 2)
    mat = cv.getRotationMatrix2D(center, angle, scale)
    dst = cv.warpAffine(image, mat, shape[:2])
    return dst


if __name__ == "__main__":
    img = cv.imread('/home/hack/PycharmProjects/vgg/data/test.png')
    dst = rotate_center(img, 3)
    cv.imshow("dst", dst)
    cv.waitKey()
