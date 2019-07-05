import cv2 as cv
import matplotlib.pyplot as  plt

img = cv.imread('../data/dog.jpg')
print(img.shape)
print(img[100, 100])
eye = img[56:76, 55:85]
for i in range(0, 190, 30):
    img[i:i + 20, i:i + 30] = eye
plt.imshow(img)
plt.show()
