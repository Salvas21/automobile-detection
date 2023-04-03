import os
import cv2
import numpy as np
import extract_cars as ec
from matplotlib import pyplot as plt


def filter_image(image):
    img = cv2.GaussianBlur(image, (7, 7), 0)

    gX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)
    gY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)
    AgXgY = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

    plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(gX, cmap='gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(gY, cmap='gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(AgXgY, cmap='gray')
    plt.title('Amplitude gX gY'), plt.xticks([]), plt.yticks([])

    plt.show()