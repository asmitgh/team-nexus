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
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

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

# Initialize the tkinter window
root = tk.Tk()
root.title("Traffic Signal Simulation")

# Variables to store lane states and vehicle counts
lane_states = [StringVar() for _ in range(4)]
vehicle_counts_vars = [StringVar() for _ in range(4)]

# Initialize labels for lane states and vehicle counts
for i in range(4):
    Label(root, text=f"Lane {i+1} Vehicle Count:").grid(row=i, column=0, padx=10, pady=10)
    Label(root, textvariable=vehicle_counts_vars[i]).grid(row=i, column=1, padx=10, pady=10)
    Label(root, text=f"Lane {i+1} Signal:").grid(row=i, column=2, padx=10, pady=10)
    Label(root, textvariable=lane_states[i]).grid(row=i, column=3, padx=10, pady=10)
    vehicle_counts_vars[i].set("0")
    lane_states[i].set("Red")

# Initialize CSV logging
log_file_path = 'output/traffic_log.csv'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Initialize cycle count and set a limit for testing
cycle_count = 0
cycle_limit = 10  # Limit to 4 cycles for testing

with open(log_file_path, 'w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Cycle', 'Lane', 'Vehicle Count', 'Signal', 'Green Light Time'])

# Initialize timestamps for each lane to 0
lane_timestamps = [0 for _ in range(4)]  # Start time for each lane in seconds

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

# Function to process a frame for a single lane
def process_lane(idx, cap, video_writer):
    # Skip to the correct timestamp for this lane
    set_video_to_timestamp(cap, lane_timestamps[idx], fps)

    ret, frame = cap.read()
    if not ret:
        print(f"Video for Lane {idx + 1} has ended.")
        return None, None  # Indicate that the video has ended

    print(f"Frame captured from Lane {idx + 1}.")

    # Resize frame if not matching the expected size
    if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
        frame = cv2.resize(frame, frame_size)
        print(f"Resized frame for Lane {idx + 1}.")

    # Get vehicle count and detection results
    vehicle_count, results = count_vehicles(frame)
    
    # Draw bounding boxes on the frame
    for *xyxy, conf, cls in results.xyxy[0]:
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    
    # Write the processed frame to output video
    video_writer.write(frame)

    return vehicle_count, frame

# Function to update the GUI with current lane statuses
def update_gui():
    global cycle_count
    all_videos_ended = True
    print("\nProcessing frames from all lanes...")
    vehicle_counts = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_lane, idx, caps[idx], video_writers[idx]) for idx in range(4)]
        results = [f.result() for f in futures]

    # Check results and update GUI
    for idx, result in enumerate(results):
        if result[0] is None:  # If video has ended for this lane
            continue
        all_videos_ended = False  # At least one video is still running

        vehicle_count, frame = result
        vehicle_counts.append(vehicle_count)
        vehicle_counts_vars[idx].set(str(vehicle_count))

    if all_videos_ended or cycle_count >= cycle_limit:
        print("All videos have ended or cycle limit reached. Stopping simulation.")
        root.quit()  # End the GUI loop
        return

    # Calculate total vehicles across all lanes
    total_vehicles = sum(vehicle_counts) if sum(vehicle_counts) != 0 else 1
    print(f"Total vehicles detected across all lanes: {total_vehicles}")

    # Determine green light time for each lane based on vehicle count
    lane_timings = [
        baseline_time + (count / total_vehicles) * 3.9
        for count in vehicle_counts
    ]

    # Simulate the signal operation in sequence
    for i in range(4):
        if len(vehicle_counts) <= i:
            continue  # Skip this lane if no vehicle count is available
        
        print(f"\nGreen Light: Lane {i+1}")
        print(f"Lane {i+1}: Green light for {lane_timings[i]:.2f} seconds")

        # Update the GUI for the current signal status
        for j in range(4):
            if i == j:
                lane_states[j].set("Green")
            else:
                lane_states[j].set("Red")
        root.update()

        # Log the current signal state and timings to file
        with open(log_file_path, 'a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            log_writer.writerow([cycle_count + 1, f'Lane {i+1}', vehicle_counts[i], 'Green' if i == j else 'Red', lane_timings[i]])

        # Update the timestamp for the next cycle based on the elapsed green time
        lane_timestamps[i] += lane_timings[i]  # Increment by the green light time

        # Simulate waiting time for the green light of the current lane
        time.sleep(lane_timings[i])

    cycle_count += 1  # Increment the cycle count after completing a round
    root.after(1000, update_gui)  # Schedule the next update after a short delay

# Start the simulation
root.after(1000, update_gui)
root.mainloop()

# Release all captures and close windows
for cap in caps:
    cap.release()

for writer in video_writers:
    writer.release()

cv2.destroyAllWindows()
print("All video captures released, program ended.")