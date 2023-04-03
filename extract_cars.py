import os
import cv2
from matplotlib import pyplot as plt

car_cascade = cv2.CascadeClassifier('cars.xml')

def extract_cars(img):
    cars = car_cascade.detectMultiScale(img, 1.08, 12)
    extracted = []

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x +w, y +h) ,(51 ,51,255),2)
        extracted.append(img[y:y+h, x:x+w])

    plt.imshow(img)
    plt.show()

    return extracted