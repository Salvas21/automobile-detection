import cv2
import numpy as np

img = cv2.imread('./assets/car.JPG', 0)
cv2.imshow('image', img)
cv2.waitKey(0)

blurImg = cv2.GaussianBlur(img, (5, 5), 0)
ret, otsuImg = cv2.threshold(blurImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow('otsu image', otsuImg)
cv2.waitKey(0)

kernel = np.ones((2, 2), np.uint8)
otsuImg = cv2.morphologyEx(otsuImg, cv2.MORPH_OPEN, kernel)
otsuImg = cv2.morphologyEx(otsuImg, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(otsuImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('contours image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()