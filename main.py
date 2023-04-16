import collections
import math
import sys
import time

import cv2
import numpy as np
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

detected_class_names = []

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
def post_process(outputs, img):
    global detected_class_names
    height, width = img.shape[:2]
    boxes = []
    class_ids = []
    confidence_scores = []
    detection = []

    # Variables for speed
    distance_in_meters = 27
    time_in_seconds = 1
    prev_frame_time = 1
    fps = 30

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in required_class_index:
                if confidence > conf_threshold:
                    # print(class_id)
                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            color = [int(c) for c in colors[class_ids[i]]]
            name = class_names[class_ids[i]]
            detected_class_names.append(name)

            # Detect speed
            d_total = math.sqrt(y ** 2)
            v = d_total / prev_frame_time * 3.6
            v = (600 - (y * 0.6))/6

            cv2.putText(img, str(int(v)) + " km/h", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw information
            cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            detection.append([x, y, w, h, required_class_index.index(class_ids[i])])

            # Draw color box



        # Update the tracker for each object
        boxes_ids = tracker.update(detection)
        for box_id in boxes_ids:
            count_vehicle(box_id, img)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time


def video_detection(video_name):
    # Initialize the videocapture object
    cap = cv2.VideoCapture(video_name)
    while True:
        success, img = cap.read()
        resized = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = resized.shape
        blob = cv2.dnn.blobFromImage(resized, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layers_names = net.getLayerNames()
        output_names = [(layers_names[i - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(output_names)

        # Find the objects from the network output
        post_process(outputs, resized)

        global middle_line_position
        middle_line_position = int(resized.shape[0] / 3)

        # Draw crossing line
        cv2.line(resized, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)

        # Draw detection counters
        # cv2.putText(img, "Counter", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(resized, "Car:        " + str(count_list[0]), (20, 260), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    car_color, font_thickness)
        cv2.putText(resized, "Motorbike:  " + str(count_list[1]), (20, 280), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    bike_color, font_thickness)
        cv2.putText(resized, "Bus:        " + str(count_list[2]), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    bus_color, font_thickness)
        cv2.putText(resized, "Truck:      " + str(count_list[3]), (20, 320), cv2.FONT_HERSHEY_SIMPLEX,
                    font_size, truck_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', resized)

        # To stop motion the video
        # if cv2.waitKey(0) == ord('c'):
        #     continue

        # For stopping the video
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_detection(image_name):
    img = cv2.imread(image_name)

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

    # Set the input of the network
    net.setInput(blob)
    layers_names = net.getLayerNames()
    output_names = [(layers_names[i - 1]) for i in net.getUnconnectedOutLayers()]
    # Feed data to the network
    outputs = net.forward(output_names)

    # Find the objects from the network output
    post_process(outputs, img)

    # count the frequency of detected classes
    frequency = collections.Counter(detected_class_names)
    print(frequency)

    # Draw counting texts in the frame
    cv2.putText(img, "Car:        " + str(count_list[0]), (20, 260), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                car_color, font_thickness)
    cv2.putText(img, "Motorbike:  " + str(count_list[1]), (20, 280), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                bike_color, font_thickness)
    cv2.putText(img, "Bus:        " + str(count_list[2]), (20, 300), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                bus_color, font_thickness)
    cv2.putText(img, "Truck:      " + str(count_list[3]), (20, 320), cv2.FONT_HERSHEY_SIMPLEX,
                font_size, truck_color, font_thickness)

    cv2.imshow("image", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        video_name = './assets/speedDetectionTest.mp4'
    else:
        video_name = sys.argv[1]
    video_detection(video_name)

    # 3/6 OK
    # plate_detection.detect_plate(cv2.imread("./assets/N37PBK.png")) # NOT OK
    # plate_detection.detect_plate(cv2.imread("./assets/X11TQD.png")) # OK
    # plate_detection.detect_plate(cv2.imread("./assets/Z08TLR.png")) # OK
    # plate_detection.detect_plate(cv2.imread("./assets/X72NDV.png")) # NOT OK
    # plate_detection.detect_plate(cv2.imread("./assets/W76GSW.png")) # OK
    # plate_detection.detect_plate(cv2.imread("./assets/RK2711X.png")) # NOT OK

    # read plates up to half height

