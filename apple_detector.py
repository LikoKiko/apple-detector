import cv2
import numpy as np
from config import (
    APPLE_WIDTH_CM, INIT_APPLE_WIDTH_PX,
    HSV_RED_LOWER, HSV_RED_UPPER, MIN_APPLE_CONTOUR_AREA
)
from distance_calculator import DistanceEstimator

def initialize_capture_device(device_index=0):
    capture = cv2.VideoCapture(device_index)
    if not capture.isOpened():
        raise IOError(f"Cannot open video capture device {device_index}")
    return capture

def process_frame(frame, estimator):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_frame, HSV_RED_LOWER, HSV_RED_UPPER)
    red_mask = cv2.erode(red_mask, None, iterations=2)
    red_mask = cv2.dilate(red_mask, None, iterations=2)
    contours, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > MIN_APPLE_CONTOUR_AREA:
            x, y, width, height = cv2.boundingRect(contour)
            estimator.width_in_frame_px = width
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, "Apple", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            estimated_distance = estimator.estimate_distance()
            cv2.putText(frame, f"Distance: {estimated_distance:.2f} cm", (x, y - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    try:
        capture = initialize_capture_device()
    except IOError as e:
        print(e)
        return

    distance_estimator = DistanceEstimator(APPLE_WIDTH_CM, INIT_APPLE_WIDTH_PX)
    calibration_distance_cm = 45  
    distance_estimator.calibrate_focal_length(calibration_distance_cm)

    print(f"Calibrated Focal Length: {distance_estimator.focal_length:.2f}")

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        processed_frame = process_frame(frame, distance_estimator)
        cv2.imshow("Apple Detector", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
