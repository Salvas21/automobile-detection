import collections
import math
import sys
import threading
import time

import cv2
import numpy as np

import colorDetection
import tracker as tracker_class
import plate_detection

# Initialize Tracker
tracker = tracker_class.EuclideanDistTracker()

# Initialize the videocapture object
# cap = cv2.VideoCapture('./assets/speedDetectionTest.mp4')
input_size = 320

# Detection confidence and non-max suppression threshold
conf_threshold = 0.2
nms_threshold = 0.2

# Printing variables
font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2

# Cross lines position
middle_line_position = 100

# Store Coco Names in a list
classes_file = "coco.names"
class_names = open(classes_file).read().strip().split('\n')

# class index for our required detection classes (car, motorbike, bus, truck as in pickup)
required_class_index = [2, 3, 5, 7]

lock = threading.Lock()
plates = {}
carColors = {}

# YoloV3 Files
model_configuration = 'yolov3-320.cfg'
model_weights = 'yolov3-320.weights'  # https://pjreddie.com/darknet/yolo/

# Deep Neural Network Configurations
net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Random Colors for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype='uint8')

car_color = colors[2].tolist()
bike_color = colors[3].tolist()
bus_color = colors[5].tolist()
truck_color = colors[7].tolist()


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# List for storing vehicle count information
temp_up_list = []
temp_down_list = []
count_list = [0, 0, 0, 0]


# Function for count vehicle
def count_vehicle(box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center

    # Find the current position of the vehicle and put it in the right list
    if iy > middle_line_position:
        if id not in temp_up_list:
            temp_up_list.append(id)
        if id in temp_down_list:
            temp_down_list.remove(id)
            count_list[index] = count_list[index] + 1

    if iy < middle_line_position:
        if id not in temp_down_list:
            temp_down_list.append(id)
        if id in temp_up_list:
            temp_up_list.remove(id)
            count_list[index] = count_list[index] + 1

    # Draw red circle at center
    cv2.circle(img, center, 2, (0, 0, 255), -1)


# Function for finding the detected objects from the network output
def post_process(outputs, img, original_img):
    height, width = img.shape[:2]
    boxes = []
    class_ids = []
    confidence_scores = []
    detection = []

    # TODO : What we do with this ???
    # Variables for speed
    distance_in_meters = 27
    time_in_seconds = 1
    prev_frame_time = 1
    fps = 30

    # selection of the outputs in relation to its confidence and the confidence threshold
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in required_class_index:
                if confidence > conf_threshold:
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, conf_threshold, nms_threshold)

    # Check each recognised object
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[class_ids[i]]]
            name = class_names[class_ids[i]]

            # TODO : clean this up
            # detect speed from this random equation with y value of the detected object
            d_total = math.sqrt(y ** 2)
            v = d_total / prev_frame_time * 3.6
            v = (600 - (y * 0.6)) / 6

            # draw km/h and class name and confidence
            cv2.putText(img, str(int(v)) + " km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw detection box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(class_ids[i])])

        # Update the tracker for each object
        boxes_ids = tracker.update(detection)

        # for each boxes (detected automobile)
        threads = []
        for box_id in boxes_ids:
            # count the vehicule if touches / is over top of the middle detection line
            count_vehicle(box_id, img)

            # get info from box
            x, y, w, h, id, index = box_id

            # if car color already found, draw it on top of box
            global carColors
            if id in carColors:
                cv2.putText(img, "Color: " + str(carColors[id]), (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (50, 180, 235), 1)

            # start a thread if color not found yet with current box position
            with lock:
                if id not in carColors and id in temp_up_list and id not in count_list:
                    thread = threading.Thread(target=handle_color_detection_of_box, args=(box_id, img))
                    thread.start()
                    threads.append(thread)

            # if car plate already found, draw it on top of box
            global plates
            cv2.putText(img, f"Plate: {str(plates[id]) if id in plates else ''}", (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

            # start a thread if car plate not found yet with current box position
            with lock:
                # if plate not found and currently traveling up
                if id not in plates and id in temp_up_list:
                    plates[id] = ""
                    thread = threading.Thread(target=handle_plate_detection_of_box, args=(box_id, original_img))
                    thread.start()
                    threads.append(thread)

        # TODO : might be better to not wait for threads to finish ?
        for thread in threads:
            thread.join()

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time


def handle_color_detection_of_box(box_id, img):
    x, y, w, h, id, index = box_id

    if x > 0 and y > 0:
        rect = img[y:y + h, x:x + w]
        color = colorDetection.find_most_common_pixel(rect)
        with lock:
            global plates
            carColors[id] = color


def handle_plate_detection_of_box(box_id, original_img):
    x, y, w, h, id, index = box_id
    # transform dimensions to original image size
    # (because we resize the 2160p video to a 1080p to simplify detection and tracking,
    # but we want the original quality to have the best chance at detecting the plates)
    big_x = x * 2
    big_y = y * 2
    big_w = w * 2
    big_h = h * 2

    img_h, img_w, c = original_img.shape
    # arbitrary ratio at which if the box of the detected object is smaller, we don't bother
    # looking for a plate since it is probably too small to even detect letters
    ratio_h = img_h * 0.12
    ratio_w = img_w * 0.12

    if big_h > ratio_h or big_w > ratio_w:
        rect = original_img[big_y:big_y + big_h, big_x:big_x + big_w]
        plate = plate_detection.detect_plate(rect)
        if len(plate) >= 4:
            with lock:
                global plates
                plates[id] = plate
        else:
            del plates[id]


def video_detection(video_name):
    cap = cv2.VideoCapture(video_name)
    while True:
        success, img = cap.read()
        # stops while true loop when the video is over
        if not success:
            break

        # resizes the video to ease processing
        resized = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = resized.shape
        blob = cv2.dnn.blobFromImage(resized, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # setup deep neural network
        net.setInput(blob)
        layers_names = net.getLayerNames()
        output_names = [(layers_names[i - 1]) for i in net.getUnconnectedOutLayers()]

        # send data to network
        outputs = net.forward(output_names)

        # find objects from network output
        post_process(outputs, resized, img)

        # sets middle line at 1/3 of the resized video height
        global middle_line_position
        middle_line_position = int(resized.shape[0] / 3)

        # draw middle (crossing) line (specifies where the objects are classified)
        cv2.line(resized, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)

        # draw data counters
        cv2.putText(resized, "Car(s):           " + str(count_list[0]), (20, 260), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    car_color, font_thickness)
        cv2.putText(resized, "Motorbike:        " + str(count_list[1]), (20, 280), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    bike_color, font_thickness)
        cv2.putText(resized, "Bus(es):          " + str(count_list[2]), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    bus_color, font_thickness)
        cv2.putText(resized, "Truck(s):         " + str(count_list[3]), (20, 320), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, truck_color, font_thickness)

        # show the frames
        cv2.imshow('Automobile detection output stream', resized)

        # For stopping the video
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        video_name = './assets/toto-4k.mp4'
    else:
        video_name = sys.argv[1]
    video_detection(video_name)

