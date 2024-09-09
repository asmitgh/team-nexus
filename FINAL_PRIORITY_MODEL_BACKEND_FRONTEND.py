import cv2
import torch
import time
from PyQt5 import QtWidgets, QtGui, QtCore
import warnings
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QFont

# Suppress specific FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"torch.cuda.amp.autocast\(args...\) is deprecated. Please use torch.amp.autocast\('cuda', args...\) instead."
)

# Load the YOLOv5 model
print("Loading Manually Tunned YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
print("Model loaded successfully.")

total_vehicle_counts = [0] * 4
total_emergency_detection = [0] * 4
total_green_light_time = [0] * 4

# Load the fine-tuned YOLOv5 model for emergency vehicles
print("Loading custom YOLOv5 model for emergency vehicles...")
custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
print("Custom model loaded successfully.")

# Load the background image
background_image = cv2.imread('LatestDesign.png')
image_height, image_width = background_image.shape[:2]

# Video paths for four lanes (assuming separate videos for each lane)
video_paths = [
    'input/video/lane1.mp4',  # Lane 1 (e.g., North)
    'input/video/lane2.mp4',  # Lane 2 (e.g., East)
    'input/video/lane3.mp4',  # Lane 3 (e.g., South)
    'input/video/lane4.mp4',  # Lane 4 (e.g., West)
]

# Initialize video captures for each lane
caps = [cv2.VideoCapture(video_path) for video_path in video_paths]
print("Video captures initialized.")

# Check if all videos were opened successfully
for idx, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Could not open video for Lane {idx + 1}.")
        exit()  # Exit if any video could not be opened

# Ensure the output directories exist
output_video_dir = 'output/video/'
os.makedirs(output_video_dir, exist_ok=True)

# Use a consistent frame size and fps for all writers
fps = int(caps[0].get(cv2.CAP_PROP_FPS))
frame_size = (int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(f"Frame Size: {frame_size}, FPS: {fps}")

# Initialize video writers for each lane
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' codec

video_writers = []
for i in range(4):
    output_path = os.path.join(output_video_dir, f'lane{i+1}_processed.avi')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    video_writers.append(video_writer)
print("Video writers initialized.")

# Initialize baseline green light time (in seconds)
baseline_time = .1  # 1 second for each lane
yellow_light_time = .5  # Yellow light duration set to 2 seconds

# Initialize CSV logging
log_file_path = 'output/traffic_log.csv'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Initialize cycle count and set a limit for testing
cycle_count = 0
cycle_limit = 20  # Limit to 5 cycles for testing

# Initialize tracking variables for statistics
total_vehicle_count_per_cycle = [0] * 4  # Tracks total vehicles for each lane across cycles
cumulative_green_light_time = [0] * 4  # Cumulative green light time per lane
emergency_detections_per_cycle = 0  # Counts total emergency vehicle detections per cycle
cycle_frame_processing_times = []  # Track processing times for each frame

with open(log_file_path, 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Cycle', 'Lane', 'Vehicle Count', 'Percentage of Vehicle', 'Green Light Time','Emergency Vehicle Detection','Accident Detection'])

# Initialize timestamps for each lane to 0
lane_timestamps = [0 for _ in range(4)]  # Start time for each lane in seconds

# Class index for emergency vehicles in your model (change if necessary)
emergency_vehicle_class_index = 5  # Replace with your actual index for emergency vehicles

accident_vehicle_class_index = 0  # Replace with your actual index for accident vehicles

# Function to skip frames to the required timestamp
def set_video_to_timestamp(cap, timestamp, fps):
    # Calculate frame number to seek to based on the timestamp and fps
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Function to count vehicles in a frame using YOLOv5
def count_vehicles(frame):
    results = model(frame)
    vehicle_count = len(results.pandas().xyxy[0])
    return vehicle_count, results

def draw_arrow(image, position, direction, color, size=15):
    """Draws an arrow of specified direction, color, and size at the given position."""
    x, y = position
    if direction == 'up':
        # Draw upward arrow
        cv2.arrowedLine(image, (x, y+size), (x, y-size), color, 2, tipLength=.5)
    elif direction == 'left':
        # Draw leftward arrow
        cv2.arrowedLine(image, (x + size, y), (x - size, y), color, 2, tipLength=.5)
    elif direction == 'right':
        # Draw rightward arrow
        cv2.arrowedLine(image, (x - size, y), (x + size, y), color, 2, tipLength=.5)
    elif direction == 'down':
        # Draw Downward arrow
        cv2.arrowedLine(image, (x, y-size), (x, y+size), color, 2, tipLength=.5)

# Function to detect emergency vehicles in a frame using custom YOLOv5 model
def detect_emergency_vehicles(frame):
    print("Detecting emergency vehicles...")
    results = custom_model(frame)
    detected_classes = results.pandas().xyxy[0]['class'].tolist()
    emergency_detected = any(cls == emergency_vehicle_class_index for cls in detected_classes)
    print(f"Emergency vehicle detected: {emergency_detected}")
    return emergency_detected, results

def detect_accident_vehicles(frame):
    print("...Detecting accident vehicles...\n")
    results=custom_model(frame)
    detected_classes=results.pandas().xyxy[0]['class'].tolist()
    accident_detected=any(cls == accident_vehicle_class_index for cls in detected_classes)
    print(f"Emergency vehicle detected :{accident_detected}")
    return accident_detected, results

# Add a new list to keep track of whether an emergency vehicle is currently detected in each lane
emergency_vehicle_active = [False] * 4

# Add a new list to keep track of whether an accident vehicle is currently detected in each lane
accident_vehicle_active = [False] * 4

# Number of seconds to skip for faster emergency detection
emergency_time_skip = 0.6  # Skip 1 second in case of emergency detection
accident_time_skip = 2

def process_lane(idx, cap, video_writer, skip_time_emergency=False, skip_time_accident=False):
    global lane_timestamps
    
    # Skip to the correct timestamp for this lane
    if skip_time_emergency:
        # Get the current timestamp
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        # Calculate the new timestamp by skipping ahead
        new_time = current_time + emergency_time_skip
        set_video_to_timestamp(cap, new_time, fps)
        lane_timestamps[idx] = new_time  # Update the timestamp after skipping
        print(f"Skipping {emergency_time_skip} seconds for Lane {idx + 1} due to emergency.")
    elif skip_time_accident:
        # Get the current timestamp
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        # Calculate the new timestamp by skipping ahead
        new_time = current_time + accident_time_skip
        set_video_to_timestamp(cap, new_time, fps)
        lane_timestamps[idx] = new_time  # Update the timestamp after skipping
        print(f"Skipping {accident_time_skip} seconds for Lane {idx + 1} due to accident.")
    else:
        set_video_to_timestamp(cap, lane_timestamps[idx], fps)

    ret, frame = cap.read()
    if not ret:
        return None, None, None  # Indicate that the video has ended

    print(f"Frame captured from Lane {idx + 1}.")
    
    # Resize frame if not matching the expected size
    if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
        frame = cv2.resize(frame, frame_size)
        print(f"Resized frame for Lane {idx + 1}.")

    # Get vehicle count and detection results
    vehicle_count, results = count_vehicles(frame)

    print(f"Counting vehicles...")
    print(f"Vehicles detected for Lane {idx + 1}: {vehicle_count}")

    # Detect emergency vehicles and get detection results
    emergency_detected, emergency_results = detect_emergency_vehicles(frame)

    # Detect accident vehicles and get detection results
    accident_detected, accident_results = detect_accident_vehicles(frame)

    # Draw bounding boxes on the frame
    # Draw bounding boxes on the frame for emergency and accident vehicles
    if emergency_detected:
        for *xyxy, conf, cls in emergency_results.xyxy[0]:
            color = (0, 0, 255) if cls == emergency_vehicle_class_index else (0, 255, 0)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label = f"{emergency_results.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    elif accident_detected:
        for *xyxy, conf, cls in accident_results.xyxy[0]:
            color = (0, 255, 255) if cls == accident_vehicle_class_index else (0, 255, 255)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label = f"{accident_results.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        for *xyxy,conf,cls in results.xyxy[0]:
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0,255,0), 2)

    # Write the processed frame to output video
    video_writer.write(frame)

    if emergency_detected:
        print(f"Emergency vehicle detected in Lane {idx + 1}!")
        emergency_vehicle_active[idx] = True  # Set flag for this lane
    else:
        emergency_vehicle_active[idx] = False  # Reset flag if no emergency vehicle is detected

    if accident_detected:
        print(f"Accident detected in Lane {idx + 1}!")
        accident_vehicle_active[idx] = True  # Set flag for this lane
    else:
        accident_vehicle_active[idx] = False  # Reset flag if no accident is detected

    # Update the lane timestamp based on the time elapsed during processing
    lane_timestamps[idx] += (1 / fps)  # Move forward by one frame's duration
    return vehicle_count, frame, emergency_detected , accident_detected

def update_gui(lane_status, current_lane, time_left, vehicle_counts, emergency_vehicle_detected, accident_vehicle_detected):
    global left_turn_signal

    # Use the background image as the base for the simulation window
    window = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Full HD window size
    window[:image_height, :image_width] = background_image.copy()

    lane_names = ["North (Lane 1)", "East (Lane 2)", "South (Lane 3)", "West (Lane 4)"]

    colors = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255), "White": (255, 255, 255), "Magenta": (170,178,32), "Black": (0,0,0), "Orange":(0,69,255), "Blue": (112,25,25), "Brown" : (19,69,139), "olive" : (0,128,128), "dodger blue" : (255,144,30), "gold":(0,215,255), "Default" : (100,100,100)}

    # Define positions for each lane's text information (adjusted)
    positions = [
        (1140, 225),  # Lane 1 text position
        (1140, 430),  # Lane 2 text position
        (1140, 630),  # Lane 3 text position
        (1140, 830)  # Lane 4 text position
    ]

    # Define positions for each lane no.
    lane_position = [
        (540,160),  # Lane 1 text position
        (940,530), # Lane 2 text position
        (540,940),  # Lane 3 text position
        (220,530),  # Lane 4 text position
    ]

    # Define positions for traffic lights (small circles)
    light_positions = [
        (705, 660),  # Lane 1 light position
        (485, 660),  # Lane 2 light position
        (485, 295),  # Lane 3 light position
        (705, 295)  # Lane 4 light position
    ]

    # Adjust text position based on lane number
    text_position = [
        (752, 770), # Lane 1 
        (255, 718), # Lane 2
        (215, 314), # Lane 3
        (710, 374) # Lane 4
    ]

    # Draw the lane status, timing information, and last active times
    for i in range(4):
        
        #Colour for left turn signal(default)
        if accident_vehicle_detected:
            left_turn_signal = colors["Default"]
        else:
            left_turn_signal = colors["Green"] if (i == current_lane and time_left > yellow_light_time) else colors["Green"]

        # Lane text 
        if(emergency_vehicle_detected and i == current_lane):
            lane_status = "Green"
            lane_color = colors["Green"]
        elif(accident_vehicle_detected):
            lane_status = "Red"
            lane_color = colors["Red"]
        else:
            if i == current_lane:
                if time_left <= yellow_light_time:
                    lane_status = "Yellow"
                    lane_color = colors["gold"]
                else:
                    lane_status = "Green"
                    lane_color = colors["Green"]
            else:
                lane_status = "Red"
                lane_color = colors["Red"]

        # Change the color of the time left text to red if time_left <= 2 sec
        time_color = colors["Red"] if i == current_lane and time_left <= yellow_light_time else colors["Brown"]

        cv2.putText(window, lane_names[i], positions[i], cv2.FONT_HERSHEY_DUPLEX, 1.5, colors["Magenta"], 4, cv2.LINE_AA)

        cv2.putText(window, f"Status: {lane_status}", (positions[i][0], positions[i][1] + 40), cv2.FONT_HERSHEY_DUPLEX, 1, lane_color, 2, cv2.LINE_AA)

        if(emergency_vehicle_detected):
            if(i!=current_lane):
                cv2.putText(window,f"Wait for the Next Signal...",(positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, time_color, 2, cv2.LINE_AA)
            else:
                cv2.putText(window, f"EMERGENCY VEHICLE DETECTED!!!",(positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 1, colors["Red"], 2, cv2.LINE_AA)
        elif (accident_vehicle_detected):
            if(i!=current_lane):
                cv2.putText(window,f"WAIT FOR CLEARANCE...",(positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, time_color, 2, cv2.LINE_AA)
            else:
                cv2.putText(window, f"ACCIDENT DETECTED!!!",(positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_TRIPLEX, 1, colors["Red"], 2, cv2.LINE_AA)
        else:
            cv2.putText(window, f"Time Left: {time_left:.2f} sec" if i == current_lane else f"Wait for the Next Signal...",
                    (positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, time_color, 2, cv2.LINE_AA)
            
        cv2.putText(window, f"Vehicles detected for Lane {i + 1}: {vehicle_counts[i]}",
                    (positions[i][0], positions[i][1] + 120), cv2.FONT_HERSHEY_COMPLEX, 1.0, colors["dodger blue"], 2, cv2.LINE_AA)
        
        #Traffic Light Condition for Lane 1

        if i == 0:
            # Draw the traffic lights (small circles) for 5 signals per lane
            if(accident_vehicle_detected):
                # Red Light Signals
                red_light_color = colors["Red"]
                cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + 200), 20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Default"]
                cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + 150), 20, yellow_light_color, -1)

                for j in range(3):
                    y_offset = 0 + j * 50
                    green_light_color = colors["Default"]
                    cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + y_offset), 20, green_light_color, -1)

            else:
                # Red Light Signals
                red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + 200), 20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time and (not emergency_vehicle_detected)) else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + 150), 20, yellow_light_color, -1)

                # Green Light Signals
                for j in range(3):
                    y_offset = 0 + j * 50
                    if(emergency_vehicle_detected):
                        green_light_color = colors["Green"] if i == current_lane else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + y_offset), 20, green_light_color, -1)
                    else:
                        green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time  else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + y_offset), 20, green_light_color, -1)

                    # Draw the arrows at the green light positions
                    if green_light_color == colors["Green"]:
                        if j==0:
                            # For Right Turn
                            draw_arrow(window, (light_positions[i][0] + 14, light_positions[i][1] + y_offset), 'left', colors["Black"])
                        elif j==1:
                            # For Straight
                            draw_arrow(window, (light_positions[i][0] + 14, light_positions[i][1] + y_offset), 'down', colors["Black"])
                        elif j==2:
                            # For Left Turn
                            draw_arrow(window, (light_positions[i][0] + 14, light_positions[i][1] + y_offset), 'right', colors["Black"])

                    #Left Turn Signals separately 
                    cv2.circle(window, (light_positions[i][0] + 14, light_positions[i][1] + 100), 20, left_turn_signal, -1)
                    draw_arrow(window, (light_positions[i][0] + 14, light_positions[i][1] + 100), 'right', colors["Black"])

        #Traffic Light Condition for Lane 2
        elif i == 1:
            # Draw the traffic lights (small circles) for 5 signals per lane
            if(accident_vehicle_detected):
                # Red Light Signals
                red_light_color = colors["Red"]
                cv2.circle(window, (light_positions[i][0] - 217, light_positions[i][1]-1), 20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Default"]
                cv2.circle(window, (light_positions[i][0] + 43 - 210, light_positions[i][1]-1), 20, yellow_light_color, -1)

                for j in range(3):
                    x_offset = 70 + j * 50
                    green_light_color = colors["Default"]
                    cv2.circle(window, (light_positions[i][0] + x_offset - 188, light_positions[i][1]-1), 20, green_light_color, -1)
            else:
                # Red Light Signals
                red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] - 217, light_positions[i][1]-1), 20, red_light_color, -1)

                #Yellow Light Signals
                yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + 43 - 210, light_positions[i][1]-1), 20, yellow_light_color, -1)

                # Green Light Signals
                for j in range(3):
                    x_offset = 70 + j * 50
                    if(emergency_vehicle_detected):
                        green_light_color = colors["Green"] if i == current_lane else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0] + x_offset - 188, light_positions[i][1]-1), 20, green_light_color, -1)
                    else:
                        green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0] + x_offset - 188, light_positions[i][1]-1), 20, green_light_color, -1)

                    # Draw the arrows at the green light positions
                    if green_light_color == colors["Green"]:
                        if j==2:
                            # For Right Turn
                            draw_arrow(window, (light_positions[i][0]+ x_offset - 188, light_positions[i][1]-1), 'up', colors["Black"])
                        elif j==1:
                            # For Straight
                            draw_arrow(window, (light_positions[i][0]+ x_offset - 188, light_positions[i][1]-1), 'left', colors["Black"])
                        elif j==0:
                            # For Left Turn
                            draw_arrow(window, (light_positions[i][0]+ x_offset - 188, light_positions[i][1]-1), 'down', colors["Black"])

                    #Left Turn Signals separately 
                    cv2.circle(window, (light_positions[i][0] + 70 - 188, light_positions[i][1]-1), 20, left_turn_signal, -1)
                    draw_arrow(window, (light_positions[i][0] + 70 - 188, light_positions[i][1]-1), 'down', colors["Black"])

        #Traffic Light Condition for Lane 3
        elif i == 2:
            # Draw the traffic lights (small circles) for 5 signals per lane
            if(accident_vehicle_detected):
                # Red Light Signals
                red_light_color = colors["Red"]
                cv2.circle(window, (light_positions[i][0]-17, light_positions[i][0] - 277),20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Default"]
                cv2.circle(window, (light_positions[i][0]-17, light_positions[i][1] - 38), 20, yellow_light_color, -1)

                for j in range(3):
                    y_offset = 70 + j * 50
                    green_light_color = colors["Default"]
                    cv2.circle(window, (light_positions[i][0]-17, light_positions[i][1] + y_offset - 57), 20, green_light_color, -1)
            else:
                # Red Light Signals
                red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0]-17, light_positions[i][0] - 277),20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0]-17, light_positions[i][1] - 38), 20, yellow_light_color, -1)

                # Green Light Signals
                for j in range(3):
                    y_offset = 70 + j * 50
                    if(emergency_vehicle_detected):
                        green_light_color = colors["Green"] if i == current_lane else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0]-17, light_positions[i][1] + y_offset - 57), 20, green_light_color, -1)
                    else:
                        green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0]-17, light_positions[i][1] + y_offset - 57), 20, green_light_color, -1)
                    
                    # Draw the arrows at the green light positions
                    if green_light_color == colors["Green"]:
                        if j==0:
                            # For Left Turn
                            draw_arrow(window, (light_positions[i][0] - 17, light_positions[i][1] + y_offset - 57), 'left', colors["Black"])
                        elif j==1:
                            # For Straight
                            draw_arrow(window, (light_positions[i][0] - 17, light_positions[i][1] + y_offset - 57), 'up', colors["Black"])
                        elif j==2:
                            # For Right Turn
                            draw_arrow(window, (light_positions[i][0] - 17, light_positions[i][1] + y_offset - 57), 'right', colors["Black"])

                    #Left Turn Signals separately 
                    cv2.circle(window, (light_positions[i][0] - 17, light_positions[i][1] + 13), 20, left_turn_signal, -1)
                    draw_arrow(window, (light_positions[i][0] - 17, light_positions[i][1] + 13), 'left', colors["Black"])

        #Traffic Light Condition for Lane 4
        elif i == 3:
            # Draw the traffic lights (small circles) for 5 signals per lane
            if(accident_vehicle_detected):
                # Red Light Signals
                red_light_color = colors["Red"]
                cv2.circle(window, (light_positions[i][0] + 215, light_positions[i][1] + 114), 20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Default"]
                cv2.circle(window, (light_positions[i][0] + 165, light_positions[i][1] + 114), 20, yellow_light_color, -1)

                for j in range(3):
                    x_offset = 0 + j * 50
                    green_light_color = colors["Default"]
                    cv2.circle(window, (light_positions[i][0] + x_offset + 15, light_positions[i][1] + 114), 20, green_light_color, -1)
            else:
                # Red Light Signals
                red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + 215, light_positions[i][1] + 114), 20, red_light_color, -1)

                # Yellow Light Signals
                yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + 165, light_positions[i][1] + 114), 20, yellow_light_color, -1)

                # Green Light Signals
                for j in range(3):
                    x_offset = 0 + j * 50
                    if(emergency_vehicle_detected):
                        green_light_color = colors["Green"] if i == current_lane else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0] + x_offset + 15, light_positions[i][1] + 114), 20, green_light_color, -1)
                    else:
                        green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                        cv2.circle(window, (light_positions[i][0] + x_offset + 15, light_positions[i][1] + 114), 20, green_light_color, -1)
                    
                    # Draw the arrows at the green light positions
                    if green_light_color == colors["Green"]:
                        if j==2:
                            # For Left Turn
                            draw_arrow(window, (light_positions[i][0] + x_offset + 15, light_positions[i][1] + 114), 'up', colors["Black"])
                        elif j==1:
                            # For Straight
                            draw_arrow(window, (light_positions[i][0] + x_offset + 15, light_positions[i][1] + 114), 'right', colors["Black"])
                        elif j==0:
                            # For Right Turn
                            draw_arrow(window, (light_positions[i][0] + x_offset + 15, light_positions[i][1] + 114), 'down', colors["Black"])

                    #Left Turn Signals separately 
                    cv2.circle(window, (light_positions[i][0] + 115, light_positions[i][1] + 114), 20, left_turn_signal, -1)
                    draw_arrow(window, (light_positions[i][0] + 115, light_positions[i][1] + 114), 'up', colors["Black"])

        # Add "Signal for Lane no." text beside each signal
        cv2.putText(window, f"Signal for Lane {i + 1}", text_position[i], cv2.FONT_HERSHEY_COMPLEX, 0.7, colors["dodger blue"], 2, cv2.LINE_AA)

    # Convert window image to QImage format for PyQt
    window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
    height, width, channel = window_rgb.shape
    bytes_per_line = 3 * width
    q_img = QtGui.QImage(window_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

    # Update the QLabel with the new image
    label_image.setPixmap(QtGui.QPixmap.fromImage(q_img))

class TrafficSignalSimulator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.cycle_count = 0
        self.running = False
        self.paused = False

    def init_ui(self):
        global label_image
        self.setWindowTitle("Traffic Signal Simulation")

        # Create a vertical layout to stack widgets
        vbox = QtWidgets.QVBoxLayout()

        # Create a QLabel for displaying the image
        label_image = QtWidgets.QLabel(self)
        pixmap = QtGui.QPixmap('Teamlogo.png')  # Load the image
        label_image.setPixmap(pixmap)  # Set the pixmap to the QLabel
        vbox.addWidget(label_image)  # Add the image label to the vertical layout

        # Create a horizontal layout for buttons
        hbox = QtWidgets.QHBoxLayout()

        self.start_button = QtWidgets.QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_simulation)
        hbox.addWidget(self.start_button)

        self.pause_button = QtWidgets.QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_simulation)
        hbox.addWidget(self.pause_button)

        self.stop_button = QtWidgets.QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_simulation)
        hbox.addWidget(self.stop_button)

        # Add the button layout to the main vertical layout
        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.paused = False
            QtCore.QTimer.singleShot(100, self.update_gui_loop)

    def pause_simulation(self):
        if self.running:
            self.paused = not self.paused
            self.running = False

    def stop_simulation(self):
        self.running = False
        self.paused = False
        QtWidgets.qApp.quit()

    def update_gui_loop(self):
        global cycle_count
        if not self.running:
            return

        if self.paused:
            QtCore.QTimer.singleShot(100, self.update_gui_loop)
            return

        all_videos_ended = True
        vehicle_counts = []
        emergency_vehicle_detected_in_any_lane = False
        accident_vehicle_detected_in_any_lane = False
        print("\nProcessing frames from all lanes...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_lane, idx, caps[idx], video_writers[idx], emergency_vehicle_active[idx],accident_vehicle_active[idx]) for idx in range(4)]
            results = [f.result() for f in futures]

        for idx, result in enumerate(results):
            if result[0] is None:   # If video has ended for this lane
                print(f"Video for Lane {idx + 1} has ended. Stopping simulation.")
                QtWidgets.qApp.quit()
                return
            
            all_videos_ended = False
            vehicle_count, frame, emergency_detected,accident_detected = result
            vehicle_counts.append(vehicle_count)
            if emergency_detected:
                emergency_vehicle_detected_in_any_lane = True
                print(f"Emergency vehicle detected in Lane {idx + 1}!")
            elif accident_detected:
                accident_vehicle_detected_in_any_lane = True
                print(f"Accident detected in Lane {idx + 1}!")
                
        total_vehicles = sum(vehicle_counts) if sum(vehicle_counts) != 0 else 1
        print(f"\nTotal vehicles detected across all lanes: {total_vehicles}\n")

        if all_videos_ended or self.cycle_count >= cycle_limit:
            print("All videos have ended or cycle limit reached. Stopping simulation.")
            self.running = False
            QtWidgets.qApp.quit()
            return
        
        # Calculate average vehicle count per lane for the cycle
        average_vehicle_count = total_vehicles / len(vehicle_counts)
        print(f"Average vehicles per lane in this cycle: {average_vehicle_count:.2f}")

        # Calculate percentage of vehicles in each lane
        vehicle_percentages = [(count / total_vehicles) * 100 for count in vehicle_counts]
        for idx, percent in enumerate(vehicle_percentages):
            print(f"Lane {idx + 1} has {vehicle_counts[idx]} vehicles ({percent:.2f}%)")


        lane_timings = [
            baseline_time + (count / total_vehicles) * 4.1
            for count in vehicle_counts
        ]

        # Handle emergency vehicle scenario
        if emergency_vehicle_detected_in_any_lane:
            for i, active in enumerate(emergency_vehicle_active):
                if active:
                    print(f"Emergency vehicle active in Lane {i + 1}, setting green light until vehicle passes.")
                    lane_statuses = "Green"
                    time_left = lane_timings[i]

                    # Continue checking if emergency vehicle is still detected
                    while emergency_vehicle_active[i]:
                        print(f"Keeping Lane {i + 1} green for emergency vehicle.")
                        update_gui(lane_statuses, i, time_left, vehicle_counts, True, False)
                        QtWidgets.qApp.processEvents()
                        time.sleep(1)  # Pause for a second before next check
                        # Process next frame to check if emergency vehicle is still detected by skipping time
                        _, _, emergency_detected , _ = process_lane(i, caps[i], video_writers[i], skip_time_emergency = True, skip_time_accident = False)
                        emergency_vehicle_active[i] = emergency_detected

                    emergency_vehicle_detected_in_any_lane = True
                    emergency_detected_lane = idx
                    print(f"Emergency vehicle passed in Lane {i + 1}. Resuming normal traffic control.")

        elif accident_vehicle_detected_in_any_lane :
            for i, active in enumerate(accident_vehicle_active):
                if active:
                    print(f"Accident happend in Lane {i + 1}. We turns all signal into red")
                    lane_statuses = "Red"
                    time_left = lane_timings[i]
                    # Continue checking if emergency vehicle is still detected
                    while accident_vehicle_active[i]:
                        update_gui(lane_statuses, i, time_left, vehicle_counts, False, True)
                        QtWidgets.qApp.processEvents()
                        time.sleep(1)  # Pause for a second before next check
                        # Process next frame to check if emergency vehicle is still detected by skipping time
                        _, _, _, accident_detected = process_lane(i, caps[i], video_writers[i], False, True)
                        accident_vehicle_active[i] = accident_detected

                    accident_vehicle_detected_in_any_lane = True
                    accident_detected_lane = i
                    print(f"\n Accident safely cleared from lane {idx + 1}\n. Resuming normal traffic control.\n")

        else:
            # Sort lanes by vehicle count to determine the order of green signals
            global sorted_lane_indices
            sorted_lane_indices = sorted(range(4), key=lambda idx: vehicle_counts[idx], reverse=True)

            for i in sorted_lane_indices:
                if i > 0:
                    previous_lane = sorted_lane_indices[sorted_lane_indices.index(i) - 1]
                else:
                    previous_lane = sorted_lane_indices[-1]

                lane_statuses = ["Green" if i == j else "Red" for j in range(4)]

                time_left = lane_timings[i]
                print(f"Green light timing for Lane {i + 1}: {lane_timings[i]:.2f} sec")  # Display green light timings in terminal
                print(f"Yellow Light: Lane {previous_lane + 1} for {yellow_light_time} seconds before Lane {i+1} turns green\n")
                while time_left > 0:
                    update_gui(lane_statuses, i, time_left, vehicle_counts, False, False)
                    QtWidgets.qApp.processEvents()
                    time.sleep(0.005)
                    if not self.running or self.paused:
                        return
                    time_left -= 0.03

        # Log to CSV after processing all lanes in the sorted order of the current cycle
        with open(log_file_path, 'a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            if(emergency_vehicle_detected_in_any_lane):
                for idx in range(4):  
                    if(idx==emergency_detected_lane):
                        log_writer.writerow([
                            cycle_count + 1,  # Corrected cycle count
                            idx + 1,  # Lane number in sorted order
                            vehicle_counts[idx],  # Vehicle count for this lane
                            f"{vehicle_percentages[idx]:.2f}%",  # Percentage of total vehicles
                            "Until Emergency Vehicle Passed",  # Green light time for this lane
                            "Detected", # Emergency Vehicle Detection
                            "Not Detected"
                        ])
                    else:
                        log_writer.writerow([
                            cycle_count + 1,  # Corrected cycle count
                            idx + 1,  # Lane number in sorted order
                            vehicle_counts[idx],  # Vehicle count for this lane
                            f"{vehicle_percentages[idx]:.2f}%",  # Percentage of total vehicles
                            "Suspended due to Emergency Vehicles",  # Green light time for this lane
                            "Not Detected", # Emergency Vehicle Detection
                            "Not Detected"
                        ])
            elif(accident_vehicle_detected_in_any_lane):
                for idx in range(4):  
                    if(idx==accident_detected_lane):
                        log_writer.writerow([
                            cycle_count + 1,  # Corrected cycle count
                            idx + 1,  # Lane number in sorted order
                            vehicle_counts[idx],  # Vehicle count for this lane
                            f"{vehicle_percentages[idx]:.2f}%",  # Percentage of total vehicles
                            "Until Accident is cleared",  # Green light time for this lane
                            "Not Detected", # Emergency Vehicle Detection
                            "Detected"
                        ])
                    else:
                        log_writer.writerow([
                            cycle_count + 1,  # Corrected cycle count
                            idx + 1,  # Lane number in sorted order
                            vehicle_counts[idx],  # Vehicle count for this lane
                            f"{vehicle_percentages[idx]:.2f}%",  # Percentage of total vehicles
                            "Suspended due to Accident",  # Green light time for this lane
                            "Not Detected", # Emergency Vehicle Detection
                            "Not Detected"
                        ])
            else:
                for idx in sorted_lane_indices:
                    log_writer.writerow([
                        cycle_count + 1,  # Corrected cycle count
                        idx + 1,  # Lane number in sorted order
                        vehicle_counts[idx],  # Vehicle count for this lane
                        f"{vehicle_percentages[idx]:.2f}%",  # Percentage of total vehicles
                        lane_timings[idx],  # Green light time for this lane
                        "Not Detected", # Emergency Vehicle Detection
                        "Not Detected"
                    ])

        # Update lane timestamps for the next cycle
        for idx in range(4):
            lane_timestamps[idx] += (lane_timings[idx])

        # Increment cycle count after a full cycle
        self.cycle_count += 1
        cycle_count += 1

        QtCore.QTimer.singleShot(100, self.update_gui_loop)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    simulator = TrafficSignalSimulator()
    simulator.show()
    app.exec_()
