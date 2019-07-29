import cv2 as cv
import numpy as np

#
# def custom_blur_demo(image):
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
#     dst = cv.filter2D(image, -1, kernel=kernel)
#     cv.imshow("custom_blur_demo", dst)
#
#
# src = cv.imread("/home/hack/PycharmProjects/vgg/data/paper/untitled.png")
#
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)
# custom_blur_demo(src)
# cv.waitKey(0)
# cv.destroyAllWindows()

img = cv.imread("/home/hack/PycharmProjects/vgg/data/paper/untitled.png")

kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
image = cv.filter2D(img, -1, kernel)
cv.imshow("1", image)

# image = cv.bilateralFilter(img, 9, 75, 75)
# cv.imshow("2", image)

sigma = 1;
threshold = 5;
amount = 1
blurred = cv.GaussianBlur(img, (0, 0), 1, None, 1)
lowContrastMask = abs(img - blurred) < threshold
sharpened = img * (1 + amount) + blurred * (-amount)
image = cv.bitwise_or(sharpened.astype(np.uint8), lowContrastMask.astype(np.uint8))
cv.imshow("3", image)

cv.waitKey(0)
cv.destroyAllWindows()
