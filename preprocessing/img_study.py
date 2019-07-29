import numpy as np
from keras.preprocessing import image
import cv2

# img = image.load_img('/home/hack/Pictures/a.png')
# x = image.img_to_array(img)
# print(type(x))
# res = img - x
# print(res.any())
#
# cv_img = cv2.imread('/home/hack/Pictures/a.png')
# print(type(cv_img))
# res = cv_img - x
# print(res.any())

# <class 'numpy.ndarray'>
# False
# <class 'numpy.ndarray'>
# True

# img = cv2.imread('/home/hack/PycharmProjects/vgg/data/paper/photo1.png')
# dst_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", dst_img)
# # cv2.waitKey(3)
# # dst_img = cv2.blur(dst_img, (3, 3))
# # cv2.imshow("blur", dst_img)
#
# dst_img = cv2.Canny(dst_img, 3, 9, 3)
# cv2.imshow("canny", dst_img)
# cv2.waitKey(0)
