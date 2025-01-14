import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('/home/hack/PycharmProjects/vgg/data/paper/photo1.png',0)
edges = cv.Canny(img,100,100)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()