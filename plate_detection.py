import concurrent.futures
import threading
import time

import cv2
import pytesseract

tesseract_path = "/usr/local/Cellar/tesseract/5.3.1/bin/tesseract"
# languages: "eng", "osd", and "snum"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

big_car_img = None
lock = threading.Lock()
texts = ""


def detect_plate(car_img):
    print("Called from detect_plate")
    # https://www.makeuseof.com/python-car-license-plates-detect-and-recognize/?newsletter_popup=1
    global big_car_img
    big_car_img = car_img

    if not car_img.any():
        return ""

    gray_image = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Can possibly change the canny params to alter result
    edged_image = cv2.Canny(gray_image, 30, 200)
    contours, new = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # img1 = car_img.copy()
    # cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("img1", img1)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # stores the license plate contour
    screen_contour = None
    # img2 = car_img.copy()

    # draws top 30 contours
    # cv2.drawContours(img2, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("img2", img2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(find_text, contours)

    with lock:
        global texts
        print(texts)
        plate = texts
        texts = ""
    return plate


def find_text(contour):
    # approximate the license plate contour, maybe can useful later ?
    # contour_perimeter = cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, 0.018 * contour_perimeter, True)

    # find the coordinates of the license plate contour
    x, y, w, h = cv2.boundingRect(contour)
    new_img = big_car_img[y: y + h, x: x + w]
    text = pytesseract.image_to_string(new_img, lang='eng')

    with lock:
        global texts
        if len(text) > 0 and len(text) > len(texts):
            texts = text
