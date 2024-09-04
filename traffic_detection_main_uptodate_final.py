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

# Suppress specific FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"torch.cuda.amp.autocast\(args...\) is deprecated. Please use torch.amp.autocast\('cuda', args...\) instead."
)

# Load the YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
print("Model loaded successfully.")

# Load the background image
background_image = cv2.imread('Prem.png')
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
baseline_time = 0.1  # 1 second for each lane
yellow_light_time = 2  # Yellow light duration set to 2 seconds

# Initialize CSV logging
log_file_path = 'output/traffic_log.csv'
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Initialize cycle count and set a limit for testing
cycle_count = 0
cycle_limit = 20  # Limit to 5 cycles for testing

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

def update_gui(lane_status, current_lane, time_left, vehicle_counts):
    global left_turn_signal
    # Use the background image as the base for the simulation window
    window = np.zeros((1124, 1920, 3), dtype=np.uint8)  # Full HD window size
    window[:image_height, :image_width] = background_image.copy()

    lane_names = ["North (Lane 1)", "East (Lane 2)", "South (Lane 3)", "West (Lane 4)"]
    colors = {"Green": (0, 255, 0), "Red": (0, 0, 255), "Yellow": (0, 255, 255), "Cyan": (255, 255, 0), "White": (255, 255, 255), "Magenta": (181, 61, 253), "Black": (0,0,0), "Orange":(0,69,255), "Blue": (112,25,25), "Brown" : (19,69,139), "olive" : (0,128,128), "dodger blue" : (255,144,30), "gold":(0,215,255)}

    # Define positions for each lane's text information (adjusted)
    positions = [
        (1180, 230),  # Lane 1 text position
        (1180, 430),  # Lane 2 text position
        (1180, 630),  # Lane 3 text position
        (1180, 830)  # Lane 4 text position
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
        (750, 750), # Lane 1
        (291, 715), # Lane 2
        (250, 350), # Lane 3
        (702, 385) # Lane 4
    ]

    # Draw the lane status, timing information, and last active times
    for i in range(4):
        
        #Colour for left turn signal(default)
        left_turn_signal = colors["Green"] if i == current_lane and time_left > yellow_light_time else colors["Green"]
        # Lane text 
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
        time_color = colors["Red"] if i == current_lane and time_left <= yellow_light_time else colors["olive"]

        cv2.putText(window, lane_names[i], positions[i], cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors["Blue"], 4, cv2.LINE_AA)
        cv2.putText(window, f"Status: {lane_status}", (positions[i][0], positions[i][1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, lane_color, 2, cv2.LINE_AA)
        cv2.putText(window, f"Time Left: {time_left:.2f} sec" if i == current_lane else f"Wait for the Next Signal...",
                    (positions[i][0], positions[i][1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, time_color, 2, cv2.LINE_AA)
        cv2.putText(window, f"Vehicles detected for Lane {i + 1}: {vehicle_counts[i]}",
                    (positions[i][0], positions[i][1] + 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors["dodger blue"], 2, cv2.LINE_AA)
        
        #Traffic Light Condition for Lane 1
        if i == 0:
            # Draw the traffic lights (small circles) for 5 signals per lane

            # Green Light Signals
            for j in range(3):
                y_offset = 0 + j * 43
                green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + 10, light_positions[i][1] + y_offset), 20, green_light_color, -1)

                # Draw the arrows at the green light positions
                if green_light_color == colors["Green"]:
                    if j==0:
                        # For Right Turn
                        draw_arrow(window, (light_positions[i][0] + 10, light_positions[i][1] + y_offset), 'left', colors["Black"])
                    elif j==1:
                        # For Straight
                        draw_arrow(window, (light_positions[i][0] + 10, light_positions[i][1] + y_offset), 'down', colors["Black"])
                    elif j==2:
                        # For Left Turn
                        draw_arrow(window, (light_positions[i][0] + 10, light_positions[i][1] + y_offset), 'right', colors["Black"])

            # Yellow Light Signals
            yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0] + 10, light_positions[i][1] + 129), 20, yellow_light_color, -1)

            # Red Light Signals
            red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0] + 10, light_positions[i][1] + 172), 20, red_light_color, -1)

            #Left Turn Signals separately 
            cv2.circle(window, (light_positions[i][0] + 10, light_positions[i][1] + 86), 20, left_turn_signal, -1)
            draw_arrow(window, (light_positions[i][0] + 10, light_positions[i][1] + 86), 'right', colors["Black"])

        #Traffic Light Condition for Lane 2
        elif i == 1:
           # Draw the traffic lights (small circles) for 5 signals per lane
            # Green Light Signals
            for j in range(3):
                x_offset = 70 + j * 43
                green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + x_offset - 165, light_positions[i][1]), 20, green_light_color, -1)

                # Draw the arrows at the green light positions
                if green_light_color == colors["Green"]:
                    if j==2:
                        # For Right Turn
                        draw_arrow(window, (light_positions[i][0]+ x_offset - 165, light_positions[i][1]), 'up', colors["Black"])
                    elif j==1:
                        # For Straight
                        draw_arrow(window, (light_positions[i][0]+ x_offset - 165, light_positions[i][1]), 'left', colors["Black"])
                    elif j==0:
                        # For Left Turn
                        draw_arrow(window, (light_positions[i][0]+ x_offset - 165 , light_positions[i][1]), 'down', colors["Black"])

            # Yellow Light Signals
            yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0] + 43 - 180, light_positions[i][1]), 20, yellow_light_color, -1)

            # Red Light Signals
            red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0] - 180, light_positions[i][1]), 20, red_light_color, -1)

            #Left Turn Signals separately 
            cv2.circle(window, (light_positions[i][0] + 70 - 165, light_positions[i][1]), 20, left_turn_signal, -1)
            draw_arrow(window, (light_positions[i][0] + 70 - 165, light_positions[i][1]), 'down', colors["Black"])

        #Traffic Light Condition for Lane 3
        elif i == 2:
            # Draw the traffic lights (small circles) for 5 signals per lane

            # Red Light Signals
            red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0]-8, light_positions[i][0] - 225),20, red_light_color, -1)

            # Yellow Light Signals
            yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0]-8, light_positions[i][1] + 7), 20, yellow_light_color, -1)

            # Green Light Signals
            for j in range(3):
                y_offset = 70 + j * 43
                green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0]-8, light_positions[i][1] + y_offset - 21), 20, green_light_color, -1)
                
                # Draw the arrows at the green light positions
                if green_light_color == colors["Green"]:
                    if j==0:
                        # For Left Turn
                        draw_arrow(window, (light_positions[i][0] - 8, light_positions[i][1] + y_offset - 21), 'left', colors["Black"])
                    elif j==1:
                        # For Straight
                        draw_arrow(window, (light_positions[i][0] - 8, light_positions[i][1] + y_offset - 21), 'up', colors["Black"])
                    elif j==2:
                        # For Right Turn
                        draw_arrow(window, (light_positions[i][0] - 8, light_positions[i][1] + y_offset - 21), 'right', colors["Black"])

                #Left Turn Signals separately 
                cv2.circle(window, (light_positions[i][0] - 8, light_positions[i][1] + 74 - 25), 20, left_turn_signal, -1)
                draw_arrow(window, (light_positions[i][0] - 8, light_positions[i][1] + 74 - 25), 'left', colors["Black"])

        #Traffic Light Condition for Lane 4
        elif i == 3:
            # Draw the traffic lights (small circles) for 5 signals per lane

            # Red Light Signals
            red_light_color = colors["Red"] if i != current_lane else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0] + 183, light_positions[i][1] + 130), 20, red_light_color, -1)

            # Yellow Light Signals
            yellow_light_color = colors["Yellow"] if (i == current_lane and time_left <= yellow_light_time) else (100, 100, 100)
            cv2.circle(window, (light_positions[i][0] + 140, light_positions[i][1] + 130), 20, yellow_light_color, -1)

            # Green Light Signals
            for j in range(3):
                x_offset = 0 + j * 43
                green_light_color = colors["Green"] if i == current_lane and time_left > yellow_light_time else (100, 100, 100)
                cv2.circle(window, (light_positions[i][0] + x_offset + 10, light_positions[i][1] + 130), 20, green_light_color, -1)
                
                # Draw the arrows at the green light positions
                if green_light_color == colors["Green"]:
                    if j==2:
                        # For Left Turn
                        draw_arrow(window, (light_positions[i][0] + x_offset + 10, light_positions[i][1] + 130), 'up', colors["Black"])
                    elif j==1:
                        # For Straight
                        draw_arrow(window, (light_positions[i][0] + x_offset + 10, light_positions[i][1] + 130), 'right', colors["Black"])
                    elif j==0:
                        # For Right Turn
                        draw_arrow(window, (light_positions[i][0] + x_offset + 10, light_positions[i][1] + 130), 'down', colors["Black"])

                #Left Turn Signals separately 
                cv2.circle(window, (light_positions[i][0] + 96, light_positions[i][1] + 130), 20, left_turn_signal, -1)
                draw_arrow(window, (light_positions[i][0] + 96, light_positions[i][1] + 130), 'up', colors["Black"])

        # Add "Signal for Lane no." text beside each signal
        cv2.putText(window, f"Signal for Lane {i + 1}", text_position[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Black"], 2, cv2.LINE_AA)

        # Add "Lane No." text on the each lane to clarify lane no.
        cv2.putText(window, f"Lane : {i + 1}", lane_position[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["Cyan"], 2, cv2.LINE_AA)

    # Convert window image to QImage format for PyQt
    window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
    height, width, channel = window_rgb.shape
    bytes_per_line = 3 * width
    q_img = QtGui.QImage(window_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

    # Update the QLabel with the new image
    label_image.setPixmap(QtGui.QPixmap.fromImage(q_img))

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

    print(f"Counting vehicles...")
    print(f"Vehicles detected for Lane {idx + 1}: {vehicle_count}")

    # Draw bounding boxes on the frame
    for *xyxy, conf, cls in results.xyxy[0]:
        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
    
    # Write the processed frame to output video
    video_writer.write(frame)

    return vehicle_count, frame

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

        vbox = QtWidgets.QVBoxLayout()

        label_image = QtWidgets.QLabel(self)
        vbox.addWidget(label_image)

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

        vbox.addLayout(hbox)
        self.setLayout(vbox)

    def start_simulation(self):
        if not self.running:
            self.running = True
            self.paused = False
            self.update_gui_loop()

    def pause_simulation(self):
        if self.running:
            self.paused = not self.paused

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

        if self.cycle_count >= cycle_limit:
            print("All videos have ended or cycle limit reached. Stopping simulation.")
            self.running = False
            QtWidgets.qApp.quit()
            return
        
        print("\nProcessing frames from all lanes...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_lane, idx, caps[idx], video_writers[idx]) for idx in range(4)]
            results = [f.result() for f in futures]

        for idx, result in enumerate(results):
            if result[0] is None:
                print(f"Video for Lane {idx + 1} has ended. Stopping simulation.")
                QtWidgets.qApp.quit()
                return
            
            all_videos_ended = False
            vehicle_count, frame = result
            vehicle_counts.append(vehicle_count)

        total_vehicles = sum(vehicle_counts) if sum(vehicle_counts) != 0 else 1
        print(f"\nTotal vehicles detected across all lanes: {total_vehicles}\n")

        if all_videos_ended or self.cycle_count >= cycle_limit:
            print("All videos have ended or cycle limit reached. Stopping simulation.")
            self.running = False
            QtWidgets.qApp.quit()
            return

        lane_timings = [
            baseline_time + (count / total_vehicles) * 3.9
            for count in vehicle_counts
        ]

        # Sort lanes by vehicle count to determine the order of green signals
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
                update_gui(lane_statuses, i, time_left, vehicle_counts)
                QtWidgets.qApp.processEvents()
                time.sleep(0.001)
                if not self.running or self.paused:
                    return
                time_left -= 0.03

            lane_statuses[i] = "Red"

        # Log to CSV after processing all lanes in the sorted order of the current cycle
        with open(log_file_path, 'a', newline='') as log_file:
            log_writer = csv.writer(log_file)
            for idx in sorted_lane_indices:
                log_writer.writerow([
                    cycle_count + 1,  # Corrected cycle count
                    idx + 1,  # Lane number in sorted order
                    vehicle_counts[idx],  # Vehicle count for this lane
                    lane_statuses[idx],  # Signal state for this lane
                    lane_timings[idx],  # Green light time for this lane
                ])

        # Update lane timestamps for the next cycle
        for idx in range(4):
            lane_timestamps[idx] += (lane_timings[idx])

        # Increment cycle count after a full cycle
        cycle_count += 1
        QtCore.QTimer.singleShot(100, self.update_gui_loop)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    simulator = TrafficSignalSimulator()
    simulator.show()
    app.exec_()