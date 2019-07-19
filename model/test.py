from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

np.random.seed(1)
a = np.random.random([7, 7])
print(a)
plt.figure(1)
plt.imshow(a)

plt.figure(2)
a = cv.resize(a, (6, 6))
plt.imshow(a)
plt.show()
