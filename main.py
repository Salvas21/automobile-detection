import os
import cv2
import numpy as np
import extract_cars as ec
from matplotlib import pyplot as plt

import image_filters

images = os.listdir('./assets')

for imageName in images:
    if ".png" in imageName:
        img_bgr = cv2.imread(f'./assets/{imageName}')
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        image_filters.filter_image(gray)

        extracted_cars = ec.extract_cars(image)
        for car in extracted_cars:
            plt.imshow(car)
            plt.show()
