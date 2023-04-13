import cv2
import time
import numpy as np


def detect():
    # Initialize variables
    car_cascade = cv2.CascadeClassifier('cars.xml')
    cap = cv2.VideoCapture('./assets/speedDetectionTest.mp4')  # Change video
    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_frame_time = 0
    fps = 30
    distance_in_meters = 27
    time_in_seconds = 1
    pts = np.array([[450, 110], [800, 110], [1100, 620], [150, 620]])

    while True:
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale then detect cars
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_cascade.detectMultiScale(gray, 1.1, 3)

        # Calculate car speeds and write it in frame
        for (x, y, w, h) in cars:
            cv2.drawContours(frame, [pts], 0, (0, 255, 0), 3)  # Shows area of effect
            # Check if car is in area
            if 300 < x < 850 and 110 < y < 620:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                speed_in_meters_per_second = distance_in_meters / time_in_seconds / fps
                car_speed = speed_in_meters_per_second * w
                cv2.putText(frame, str(int(car_speed)) + " km/h", (x, y - 10), font, 0.5, (0, 0, 255), 2)

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, str(int(fps)) + " fps", (10, 30), font, 0.5, (0, 255, 0), 2)

        # Display frame with info added on it
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
