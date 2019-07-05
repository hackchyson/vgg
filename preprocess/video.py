from keras.preprocessing import image
import cv2
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import pdb


# pdb.set_trace()
img = image.load_img('../data/indoor.jpg')
print(type(img))
arr = image.img_to_array(img)

print(type(arr))
print(arr.shape)
print(arr[0, :, 0])
cv2.imshow('bala', arr)

cimg = cv2.imread('../data/indoor.jpg')
print(cimg.shape)
print(type(cimg))
# print(cimg[0, :, 0], cimg[0, :, 1], cimg[0, :, 2], sep='\n')

# cv2.imshow('bala', cimg)
cv2.waitKey(5000)

plt.imshow(img)
plt.show()
