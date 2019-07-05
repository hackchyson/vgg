import numpy as np
from keras.preprocessing import image
import cv2

img = image.load_img('/home/hack/Pictures/a.png')
x = image.img_to_array(img)
print(type(x))
res = img - x
print(res.any())

cv_img = cv2.imread('/home/hack/Pictures/a.png')
print(type(cv_img))
res = cv_img - x
print(res.any())

# <class 'numpy.ndarray'>
# False
# <class 'numpy.ndarray'>
# True