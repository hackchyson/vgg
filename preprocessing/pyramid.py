import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('/home/hack/PycharmProjects/vgg/data/paper/photo1.png')
lower_reso = cv.pyrDown(img)
higher_reso2 = cv.pyrUp(lower_reso)
plt.imshow(lower_reso)
plt.show()
