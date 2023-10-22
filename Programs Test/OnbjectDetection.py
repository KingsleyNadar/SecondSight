import cv2
import numpy as np
import math
from picamera2 import Picamera2
import pyttsx3
import os

classNames = []

classFile = "/home/SecondSight/project/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/SecondSight/project/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/SecondSight/project/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Load the reference image and define its known dimension in inches
ref_img = cv2.imread('/home/SecondSight/project/opencv-distance/rf.png')
ref_width = 6.0  # Change this value to the actual width of the object in your reference image

# Initialize text-to-speech engine with espeak
engine = pyttsx3.init('espeak')

def calculate_distance(focal_length, width, pixel_width):
    return (width * focal_length) / pixel_width

def alert_user(className, distance, min_distance=15):
    if distance < min_distance:
        alert_text = f"Warning! {className} is too close."
        print(alert_text)
        engine.say(alert_text)
        engine.runAndWait()

def get_nearest_objects(frame, thres, nms,target_class, max_objects=8, min_distance=30):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres, nmsThreshold=nms)
    
    object_info = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className == target_class:
                detect_traffic_light(frame)
                break
            
            # Calculate the distance to the object
            pixel_width = box[2] - box[0]
            distance = calculate_distance(ref_width, ref_img.shape[1], pixel_width)
            
            alert_user(className, distance, min_distance)
            
            object_info.append([box, className, distance])
            
            # Sort by distance
            object_info = sorted(object_info, key=lambda x: x[2])
            
            # Limit the number of objects to max_objects
            object_info = object_info[:max_objects]
            
    return object_info

def detect_traffic_light(frame):

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color ranges for traffic lights
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Threshold the image to find red, green, and yellow regions
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Check which color has the most pixels
    red_pixel_count = cv2.countNonZero(mask_red)
    green_pixel_count = cv2.countNonZero(mask_green)
    yellow_pixel_count = cv2.countNonZero(mask_yellow)

    if red_pixel_count > green_pixel_count and red_pixel_count > yellow_pixel_count:
        color = "red"
    elif green_pixel_count > red_pixel_count and green_pixel_count > yellow_pixel_count:
        color = "green"
    elif yellow_pixel_count > red_pixel_count and yellow_pixel_count > green_pixel_count:
        color = "yellow"
    else:
        color = "unknown"

    play_audio=f"The traffic light is {color}"
    print(play_audio)
    engine.say(play_audio)
    engine.runAndWait()


if _name_ == "_main_":
    piCam = Picamera2()
    piCam.preview_configuration.main.size = (640, 480)
    piCam.preview_configuration.main.format = "RGB888"
    piCam.preview_configuration.align()
    piCam.configure("preview")
    piCam.start()
    
    while True:
        frame = piCam.capture_array()
        object_info = get_nearest_objects(frame, 0.6, 0.2,target_class="traffic light")

        
        
        if object_info is not None:
            # Draw rectangles and labels on the frame
            for box, className, distance in object_info:
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                label = f"{className.upper()} ({distance:.2f} cm)"
                cv2.putText(frame, label, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
       
        cv2.imshow("Output", frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()