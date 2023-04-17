import concurrent.futures
import threading

import cv2
import pytesseract

tesseract_path = "/usr/local/Cellar/tesseract/5.3.1/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

big_car_img = None
lock = threading.Lock()
temp_plate = ""


def detect_plate(car_img):
    global big_car_img
    big_car_img = car_img

    if not car_img.any():
        return ""

    # grayscale image
    gray_image = cv2.cvtColor(car_img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # find all the contours from the image
    edged_image = cv2.Canny(gray_image, 30, 200)
    contours, new = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # get the 5 biggest contours from their area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # starts threads for each of the most important contours
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        executor.map(find_text, contours)

    with lock:
        global temp_plate
        plate = temp_plate
        temp_plate = ""
        plate = "".join(char for char in plate if char.isalnum())

    return plate


# find text inside the contour in the original image
def find_text(contour):
    x, y, w, h = cv2.boundingRect(contour)
    # creates new image from contour
    new_img = big_car_img[y: y + h, x: x + w]

    text = pytesseract.image_to_string(new_img, lang='eng')

    with lock:
        global temp_plate
        if len(text) > 0 and len(text) > len(temp_plate):
            temp_plate = text
