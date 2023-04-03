import os
import cv2
import numpy as np
import extract_cars as ec
from matplotlib import pyplot as plt

import image_filters

images = os.listdir('./assets')

for imageName in images:
    img_bgr = cv2.imread(f'./assets/{imageName}')
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # image_filters.filter_image(gray)

    extracted_cars = ec.extract_cars(image)
    for car in extracted_cars:
        plt.imshow(car)
        plt.show()

# https://github.com/Spidy20/Car_Detection_System/blob/master/Car_detection.py
# https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/
# https://medium.com/analytics-vidhya/detecting-the-colors-of-the-vehicle-2904e8b669f8