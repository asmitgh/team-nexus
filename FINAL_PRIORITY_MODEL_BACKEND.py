import cv2
import torch
import time
import tkinter as tk
from tkinter import Label, StringVar
import warnings
import csv
import os
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=FutureWarning, message="torch.cuda.amp.autocast(args...) is deprecated. Please use torch.amp.autocast('cuda', args...) instead.")

# Load the YOLOv5 model
print("Loading Manually Tuned YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
print("Model loaded successfully.")

# Load the fine-tuned YOLOv5 model for emergency vehicles
print("Loading custom YOLOv5 model for emergency vehicles...")
custom_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
print("Custom model loaded successfully.")


total_vehicle_counts = [0] * 4  # Total vehicles counted in each lane
total_emergency_detections = [0] * 4  # Total emergency detections per lane
total_green_light_time = [0] * 4  # Total green light time for each lane

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
baseline_time = 0.1  # 5 seconds for each lane
yellow_light_time = 0.5  # Yellow light duration set to 2 seconds

# Initialize the tkinter window
root = tk.Tk()
root.title("Traffic Signal Simulation")

# Variables to store lane states, vehicle counts, and left turn signals
lane_states = [StringVar() for _ in range(4)]
left_turn_signals = [StringVar(value="Green") for _ in range(4)]  # Left turn signals are always green
vehicle_counts_vars = [StringVar() for _ in range(4)]
right_turn_signals = [StringVar(value="Off") for _ in range(4)]

# Initialize labels for lane states, vehicle counts, and left turn signals
for i in range(4):
    Label(root, text=f"Lane {i+1} Vehicle Count:").grid(row=i, column=0, padx=10, pady=10)
    Label(root, textvariable=vehicle_counts_vars[i]).grid(row=i, column=1, padx=10, pady=10)
    Label(root, text=f"Lane {i+1} Signal:").grid(row=i, column=2, padx=10, pady=10)
    Label(root, textvariable=lane_states[i]).grid(row=i, column=3, padx=10, pady=10)
    Label(root, text=f"Lane {i+1} Left Turn Signal:").grid(row=i, column=4, padx=10, pady=10)
    Label(root, textvariable=left_turn_signals[i]).grid(row=i, column=5, padx=10, pady=10)
    Label(root, text=f"Lane {i+1} Right Turn Signal:").grid(row=i, column=6, padx=10, pady=10)
    Label(root, textvariable=right_turn_signals[i]).grid(row=i, column=7, padx=10, pady=10)
    
    vehicle_counts_vars[i].set("0")
    lane_states[i].set("Red")

# Initialize CSV logging
log_file_path = 'output/traffic_log.csv'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Initialize cycle count and set a limit for testing
cycle_count = 0
cycle_limit = 20  # Limit to 20 cycles for testing

# Initialize tracking variables for statistics
total_vehicle_count_per_cycle = [0] * 4  # Tracks total vehicles for each lane across cycles
cumulative_green_light_time = [0] * 4  # Cumulative green light time per lane
emergency_detections_per_cycle = 0  # Counts total emergency vehicle detections per cycle
cycle_frame_processing_times = []  # Track processing times for each frame

with open(log_file_path, 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Cycle', 'Lane', 'Vehicle Count','Percentage of Vehicle', 'Signal', 'Green Light Time', 'Right Signal'])

# Initialize timestamps for each lane to 0
lane_timestamps = [0 for _ in range(4)]  # Start time for each lane in seconds

# Class index for emergency vehicles in your model (change if necessary)
emergency_vehicle_class_index = 5  # Replace with your actual index for emergency vehicles

# Class index for accident vehicles in your model (change if necessary)
accident_vehicle_class_index = 0  # Replace with your actual index for emergency vehicles

# Function to skip frames to the required timestamp
def set_video_to_timestamp(cap, timestamp, fps):
    # Calculate frame number to seek to based on the timestamp and fps
    frame_number = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Function to count vehicles in a frame using YOLOv5
def count_vehicles(frame):
    print("Counting vehicles...")
    results = model(frame)
    vehicle_count = len(results.pandas().xyxy[0])
    print(f"Vehicles detected: {vehicle_count}")
    return vehicle_count, results  # Return both vehicle count and results

# Function to detect emergency vehicles in a frame using custom YOLOv5 model
def detect_emergency_vehicles(frame):
    print("...Detecting emergency vehicles...")
    results = custom_model(frame)
    detected_classes = results.pandas().xyxy[0]['class'].tolist()
    emergency_detected = any(cls == emergency_vehicle_class_index for cls in detected_classes)
    print(f"Emergency vehicle detected: {emergency_detected}")
    return emergency_detected, results

# Function to detect emergency vehicles in a frame using custom YOLOv5 model
def detect_accident_vehicles(frame):
    print("...Detecting accident vehicles...")
    results = custom_model(frame)
    detected_classes = results.pandas().xyxy[0]['class'].tolist()
    accident_detected = any(cls == accident_vehicle_class_index for cls in detected_classes)
    print(f"Emergency vehicle detected: {accident_detected}")
    return accident_detected, results

# Add a new list to keep track of whether an emergency vehicle is currently detected in each lane
emergency_vehicle_active = [False] * 4

# Add a new list to keep track of whether an emergency vehicle is currently detected in each lane
accident_vehicle_active = [False] * 4

# Number of seconds to skip for faster emergency detection
emergency_time_skip = 0.6  # Skip 1 second in case of emergency detection
accident_time_skip=0.6

# Function to process a frame for a single lane
def process_lane(idx, cap, video_writer, skip_time=False , skip_time1=False):
    
     # Skip to the correct timestamp for this lane   
    if skip_time:
        # Get the current timestamp
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        # Calculate the new timestamp by skipping ahead
        new_time = current_time + emergency_time_skip
        set_video_to_timestamp(cap, new_time, fps)
        lane_timestamps[idx] = new_time  # Update the timestamp after skipping
        print(f"Skipping {emergency_time_skip} seconds for Lane {idx + 1} due to emergency.")
    elif skip_time1:
        # Get the current timestamp
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time in seconds
        # Calculate the new timestamp by skipping ahead
        new_time = current_time + accident_time_skip
        set_video_to_timestamp(cap, new_time, fps)
        lane_timestamps[idx] = new_time  # Update the timestamp after skipping
        print(f"Skipping {emergency_time_skip} seconds for Lane {idx + 1} due to emergency.")
    else:
        set_video_to_timestamp(cap, lane_timestamps[idx], fps)

    ret, frame = cap.read()
    if not ret:
        print(f"Video for Lane {idx + 1} has ended.")
        return None, None ,None # Indicate that the video has ended

    print(f"Frame captured from Lane {idx + 1}.")

    # Resize frame if not matching the expected size
    if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
        frame = cv2.resize(frame, frame_size)
        print(f"Resized frame for Lane {idx + 1}.")

    # Get vehicle count and detection results
    vehicle_count, results = count_vehicles(frame)
    
    # Detect emergency vehicles and get detection results
    emergency_detected, emergency_results = detect_emergency_vehicles(frame)

    # Detect accident vehicles and get detection results
    accident_detected, accident_results = detect_accident_vehicles(frame)
    if emergency_detected:
        # Draw bounding boxes on the frame
        for *xyxy, conf, cls in emergency_results.xyxy[0]:
            color = (0, 0, 255) if cls == emergency_vehicle_class_index else (0, 255, 0)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label = f"{emergency_results.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    elif accident_detected:
        # Draw bounding boxes on the frame
        for *xyxy, conf, cls in accident_results.xyxy[0]:
            color = (0, 255, 255) if cls == accident_vehicle_class_index else (0, 255, 0)
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            label = f"{accident_results.names[int(cls)]} {conf:.2f}"
            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else :
            # Draw bounding boxes on the frame
            for *xyxy, conf, cls in results.xyxy[0]:
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    
        
    # Write the processed frame to output video
    video_writer.write(frame)

    if emergency_detected:
        print(f"Emergency vehicle detected in Lane {idx + 1}!\n")
        emergency_vehicle_active[idx] = True  # Set flag for this lane
    else:
        emergency_vehicle_active[idx] = False  # Reset flag if no emergency vehicle is detected
    
    if accident_detected:
        print(f"Accident detected in Lane {idx + 1}!\n")
        accident_vehicle_active[idx] = True  # Set flag for this lane
    else:
        accident_vehicle_active[idx] = False  # Reset flag if accident is detected
    

    # Update the lane timestamp based on the time elapsed during processing
    lane_timestamps[idx] += (1 / fps)  # Move forward by one frame's duration
    return vehicle_count, frame, emergency_detected , accident_detected

# Function to update the GUI with current lane statuses
def update_gui():
    global cycle_count
    print("\nProcessing frames from all lanes...")
    vehicle_counts = []
    emergency_vehicle_detected_in_any_lane = False
    accident_vehicle_detected_in_any_lane = False
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_lane, idx, caps[idx], video_writers[idx], emergency_vehicle_active[idx]) 
            for idx in range(4)
        ]
        results = [f.result() for f in futures]

    # Check results and update GUI
    for idx, result in enumerate(results):
        if result[0] is None:  # If video has ended for this lane
            print(f"Video for Lane {idx + 1} has ended. Stopping simulation.")
            root.quit()  # End the GUI loop immediately
            return

        vehicle_count, frame, emergency_detected,accident_detected = result
        vehicle_counts.append(vehicle_count)
        if emergency_detected:
            emergency_vehicle_detected_in_any_lane = True
            print(f"Emergency vehicle detected in Lane {idx + 1}!")
        
        if accident_detected:
            accident_vehicle_detected_in_any_lane = True
            print(f"Accident detected in Lane {idx + 1}!")

        vehicle_counts_vars[idx].set(str(vehicle_count))
    
    if cycle_count >= cycle_limit:
        print("Cycle limit reached. Stopping simulation.")
        root.quit()  # End the GUI loop if cycle limit is reached
        return


    # Initialize sorted_lane_indices to avoid UnboundLocalError
    sorted_lane_indices = []

    # Calculate total vehicles across all lanes
    total_vehicles = sum(vehicle_counts) if sum(vehicle_counts) != 0 else 1
    print(f"Total vehicles detected across all lanes: {total_vehicles}")

    # Calculate average vehicle count per lane for the cycle
    average_vehicle_count = total_vehicles / len(vehicle_counts)
    print(f"Average vehicles per lane in this cycle: {average_vehicle_count:.2f}")

    # Calculate percentage of vehicles in each lane
    vehicle_percentages = [(count / total_vehicles) * 100 for count in vehicle_counts]
    for idx, percent in enumerate(vehicle_percentages):
        print(f"Lane {idx + 1} has {vehicle_counts[idx]} vehicles ({percent:.2f}%)")

    # Determine green light time for each lane based on vehicle count
    lane_timings = [
        baseline_time + (count / total_vehicles) * 4.1
        for count in vehicle_counts
    ]

    # Handle emergency vehicle scenario
    if emergency_vehicle_detected_in_any_lane:
        for idx, active in enumerate(emergency_vehicle_active):
            if active:
                print(f"Emergency vehicle active in Lane {idx + 1}, setting green light until vehicle passes.")
                lane_states[idx].set("GREEN")
                right_turn_signals[idx].set("GREEN")
                left_turn_signals[idx].set("GREEN")
                root.update()

                # Continue checking if emergency vehicle is still detected
                while emergency_vehicle_active[idx]:
                    print(f"Keeping Lane {idx + 1} green for emergency vehicle.")
                    root.update()
                    time.sleep(1)  # Pause for a second before next check

                    # Process next frame to check if emergency vehicle is still detected by skipping time
                    _, _, emergency_detected ,_ = process_lane(idx, caps[idx], video_writers[idx], skip_time=True,skip_time1=False)
                    emergency_vehicle_active[idx] = emergency_detected

                print(f"\nEmergency vehicle passed in Lane {idx + 1}. Resuming normal traffic control.\n")
                lane_states[idx].set("RED")
                root.update()
    elif accident_vehicle_detected_in_any_lane:
        for idx, active in enumerate(accident_vehicle_active):
            if active:
                print(f"Accident detected in Lane {idx + 1}, setting red light until accident is cleared.")
                # Set all signals to red when an accident is detected
                for j in range(4):
                    lane_states[j].set("RED")
                    right_turn_signals[j].set("RED")
                    left_turn_signals[j].set("RED")
                root.update()

                # Continue checking if emergency vehicle is still detected
                while accident_vehicle_active[idx]:
                    print(f"Keeping Lane {idx + 1} red for emergency .")
                    root.update()
                    time.sleep(1)  # Pause for a second before next check

                    # Process next frame to check if emergency vehicle is still detected by skipping time
                    _, _, _,accident_detected = process_lane(idx, caps[idx], video_writers[idx], skip_time=False,skip_time1=True)
                    accident_vehicle_active[idx] = accident_detected

                print(f"\n Accident safely cleared from lane {idx + 1}\n. Resuming normal traffic control.\n")
                lane_states[idx].set("RED")
                root.update()
    else:               
        # Sort lanes by vehicle count to determine the order of green signals
        sorted_lane_indices = sorted(range(4), key=lambda idx: vehicle_counts[idx], reverse=True)
        # Simulate the signal operation in sequence
        for i in sorted_lane_indices:
            # Yellow light phase for the previous lane (if not the first lane in cycle)
            if i > 0:
                previous_lane = sorted_lane_indices[sorted_lane_indices.index(i) - 1]
            else:
                previous_lane = sorted_lane_indices[-1]

            print(f"\nYellow Light: Lane {previous_lane + 1} for {yellow_light_time} seconds before Lane {i+1} turns green")

            for j in range(4):
                if j == previous_lane:
                    lane_states[j].set("YELLOW")
                    right_turn_signals[j].set("OFF")
                else:
                    lane_states[j].set("RED")
                    right_turn_signals[j].set("OFF")
                left_turn_signals[j].set("GREEN")

            root.update()
            time.sleep(yellow_light_time)  # Sleep for yellow light duration
            print(f"Green Light: Lane {i+1} for {lane_timings[i]:.2f} seconds")
            # Green light phase for the current lane
            for j in range(4):
                if j == i:
                    lane_states[j].set("GREEN")
                    right_turn_signals[j].set("GREEN")
                else:
                    lane_states[j].set("RED")
                    right_turn_signals[j].set("OFF")
                left_turn_signals[j].set("GREEN")

            root.update()
            time.sleep(lane_timings[i])  # Green light duration
    
    with open(log_file_path, 'a', newline='') as log_file:
        log_writer = csv.writer(log_file)
        for idx in sorted_lane_indices:
            log_writer.writerow([
                cycle_count + 1,  # Corrected cycle count
                idx + 1,  # Lane number in sorted order
                vehicle_counts[idx],  # Vehicle count for this lane
                f"{vehicle_percentages[idx]:.2f}%",  # Percentage of total vehicles
                lane_states[idx].get(),  # Signal state for this lane
                lane_timings[idx],  # Green light time for this lane
                right_turn_signals[idx].get(),
            ])
    # Update lane timestamps for the next cycle
    for idx in range(4):
        lane_timestamps[idx] += (lane_timings[idx])

    # Increment cycle count after a full cycle
    cycle_count += 1

    # Schedule the next GUI update
    root.after(1000, update_gui)  # Call again after 1 second

# Schedule the initial GUI update
root.after(200, update_gui)

# Start the tkinter main loop
root.mainloop()

# Release video captures and writers after the loop ends
for cap in caps:
    cap.release()

for writer in video_writers:
    writer.release()

cv2.destroyAllWindows()
print("Program exited successfully.")
