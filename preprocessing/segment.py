import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('/home/hack/PycharmProjects/vgg/data/paper/bim1.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

plt.imshow(ret)
plt.show()
# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
