
![Logo](https://i.postimg.cc/mkc0wgyp/1.png)


# AI Based Traffic Management System
This project is an AI-based traffic management system that utilizes a fine-tuned and trained YOLOv5 model for real-time vehicle detection, with a focus on identifying and prioritizing emergency vehicles. The system optimizes traffic flow by detecting congestion and dynamically adjusting lane activity, ensuring that emergency vehicles are given priority and can pass through quickly. By tracking the time since each lane was last active and analyzing traffic conditions, it enhances overall efficiency and reduces delays.
<p align="left"> <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>

Designed for city planners, traffic management authorities, and organizations, this solution provides a smart, cost-effective approach to traffic management while ensuring rapid emergency response, all without requiring extensive infrastructure.

# AI-Based Traffic Management System - Roadmap

## Project Overview

This project is an AI-based Traffic Management System that dynamically allocates green light times based on real-time vehicle detection across multiple lanes. It uses YOLOv5 for vehicle detection and a custom YOLOv5 model to prioritize emergency vehicles. The system adjusts traffic signals according to vehicle density and the presence of emergency vehicles, aiming to reduce congestion and improve traffic flow.

### Core Features:
- Real-time vehicle detection across four traffic lanes.
- Prioritization of emergency vehicles to clear the way for faster response.
- Dynamic signal control based on vehicle count per lane.
- Integrated GUI to simulate traffic signal operations.

## Project Roadmap

### Phase 1: Problem Definition and Research
- **Goal:** Identify core issues with traditional traffic management systems and explore AI-based solutions.
- **Tasks:**
  - Conducted research on traffic congestion problems.
  - Evaluated existing AI models for vehicle detection.
  - Investigated YOLOv5's potential for real-time traffic detection.

### Phase 2: YOLOv5 Model Selection and Training
- **Goal:** Set up a base model using YOLOv5 for vehicle detection and fine-tune for emergency vehicles.
- **Tasks:**
  - Implemented YOLOv5 for general vehicle detection.
  - Trained a custom YOLOv5 model (`best.pt`) to specifically detect emergency vehicles like ambulances and fire trucks.
  - Verified model performance with different datasets to ensure accuracy.

### Phase 3: Video Processing and Vehicle Detection Logic
- **Goal:** Set up video feeds for multiple lanes and integrate YOLOv5 models for real-time vehicle detection.
- **Tasks:**
  - Established video input for four lanes and processed video frames.
  - Integrated YOLOv5 model to count vehicles in each frame.
  - Developed logic to detect and prioritize emergency vehicles when detected in a lane.
  
### Phase 4: Traffic Signal Control Logic
- **Goal:** Implement dynamic traffic signal management based on vehicle count and emergency vehicle detection.
- **Tasks:**
  - Designed and coded algorithms to allocate green light times based on vehicle count across all lanes.
  - Integrated special rules for emergency vehicles, giving them continuous green light until they pass.
  - Handled yellow light transitions between lane signals.

### Phase 5: GUI Development and Visualization
- **Goal:** Provide a visual representation of the traffic signals and lane statuses.
- **Tasks:**
  - Developed a GUI using Tkinter to display lane status (vehicle count, signal state, left and right turn signals).
  - Integrated the GUI with the real-time video feed, allowing users to monitor the traffic system.
  - Enabled control updates based on the detection results.

### Phase 6: Emergency Vehicle Detection and Prioritization
- **Goal:** Ensure that emergency vehicles are prioritized correctly within the system.
- **Tasks:**
  - Implemented emergency vehicle detection in parallel with regular vehicle counting.
  - Skipped regular signal cycles when an emergency vehicle was detected, ensuring priority clearance.
  - Integrated a time-skipping feature to speed up emergency response by skipping frames during emergency detection.

### Phase 7: Testing and Optimization
- **Goal:** Ensure the system operates efficiently and reliably.
- **Tasks:**
  - Performed tests using different traffic video inputs to validate the detection algorithm.
  - Optimized vehicle detection to work at high accuracy and low latency.
  - Fine-tuned the emergency vehicle detection to prevent false positives and unnecessary green light allocation.

### Phase 8: Documentation and Deployment
- **Goal:** Prepare the project for demonstration and future development.
- **Tasks:**
  - Documented the core logic, including vehicle detection, signal control, and emergency prioritization.
  - Created a feedback form for end-users to gather insights and suggestions.
  - Packaged the project for deployment and demo, including user instructions for installation and usage.

## Future Improvements:
- **Web-based Interface:** Shift the GUI to a web-based platform for easier access and remote control.
- **Scalability:** Adapt the system to handle larger, more complex intersections.
- **Real-time CCTV integration:** Allow for direct input from live traffic camera feeds for more comprehensive testing.
- **Machine Learning Enhancements:** Explore other AI models to improve detection accuracy and reduce processing time.

---

This roadmap serves as a guide to the developmental milestones and can be updated as the project progresses or if new features are added in the future.
## Features

- **Real-time Vehicle Detection:** Utilizes YOLOv5 to detect and classify vehicles in real time.
- **Emergency Vehicle Detection:** Fine-tuned YOLOv5 model trained specifically to identify emergency vehicles like ambulances, fire trucks, and police cars.
- **Priority Lane Activation:** Automatically prioritizes lanes for emergency vehicles, allowing them to pass through quickly.  
- **Dynamic Traffic Management:** Adjusts lane activity based on current traffic congestion, optimizing flow and reducing delays.
- **Time-based Lane Tracking:** Tracks the time since each lane was last active to ensure balanced traffic flow.
- **Customizable Regions of Interest:** Allows users to define traffic monitoring areas with freeform input, providing flexibility in monitoring.
- **Inbuilt Simulation Window:** Allows user to see the overview of that particular intersection through a lite simulation mirroring the traffic lights conditions.
- **Increased Safety:** By optimizing traffic light timings, the system helps in reducing the likelihood of accidents at intersections. Better traffic management leads to fewer sudden stops and smoother transitions, enhancing overall road safety for drivers and pedestrians that  can outweigh the initial costs.
- **Low Infrastructure Requirements:** Can be implemented with minimal infrastructure, using existing camera systems for vehicle detection.
- **Web App Integration:** Offers an optional web app for easy access, where users can upload CCTV data directly to the system without installing additional dependencies.
- **Real-Time Alerts and Notifications:** Sends alerts for high congestion or emergency vehicle detections to relevant authorities for immediate action.

# Detection Algorithm
This project demonstrates an AI-based traffic management system, utilizing YOLOv5 for vehicle detection and a custom YOLOv5 model for emergency vehicle identification. The system dynamically adjusts traffic signal timings based on vehicle counts and prioritizes emergency vehicles.

## Core Logic Overview

The detection algorithm comprises two primary functions:

1. **Vehicle Detection and Counting:**
   - The YOLOv5 model is used to detect vehicles in the frame of each video feed from the four lanes.
   - The count of vehicles in each frame determines how long the green light stays active for that lane.

2. **Emergency Vehicle Detection:**
   - A custom YOLOv5 model detects emergency vehicles, giving them immediate priority by skipping to the necessary frames and ensuring uninterrupted green lights until the emergency vehicle has passed.

### Vehicle Detection Process

1. **Video Frame Processing:**
   - The system captures frames from four video inputs (one for each lane).
   - Each frame is passed to the pre-trained YOLOv5 model, which returns the number of vehicles detected.
   - The vehicle count for each lane is updated in real-time.

2. **Bounding Box and Labeling:**
   - The detection results from YOLOv5 are used to draw bounding boxes around detected vehicles. A label showing the vehicle class and confidence score is added to each bounding box.

3. **Green Light Time Allocation:**
   - The total number of vehicles across all lanes is computed.
   - The green light time for each lane is determined by the proportion of vehicles in that lane relative to the total number of vehicles. This is calculated as:
     ```
     green_light_time = baseline_time + (vehicle_count / total_vehicles) * additional_time
     ```
   - The baseline green light time is predefined (e.g., 0.1 seconds), and additional time is distributed according to vehicle density.

### Emergency Vehicle Detection Process

1. **Frame Analysis:**
   - Each frame is passed through the custom YOLOv5 model to detect emergency vehicles.
   - If an emergency vehicle is detected, the system prioritizes that lane, keeping the green light active until the vehicle passes.

2. **Emergency Response:**
   - Once an emergency vehicle is detected, the frame processing skips ahead in time to ensure faster detection and reduced delays.
   - The green light for the affected lane remains active, overriding the normal traffic cycle until the emergency vehicle has cleared the lane.

### Traffic Signal Control

- **Signal Cycle:** 
  - After vehicle counting, the system sorts the lanes by vehicle density. The lane with the most vehicles gets the green light first.
  - Other lanes remain red, except during the transition

periods. Once a lane's green light cycle completes, the system proceeds to the next lane based on the vehicle counts.

- **Dynamic Timing:**
  - The green light duration for each lane is dynamically adjusted based on real-time data, ensuring smoother traffic flow by reducing wait times for lanes with fewer vehicles.

### Example Algorithm Flow

1. Capture a frame from each lane's video feed.
2. Use YOLOv5 to detect vehicles in each lane's frame.
3. Count the number of vehicles detected in each lane.
4. If an emergency vehicle is detected, override the normal traffic signal cycle and prioritize the lane with the emergency vehicle.
5. Calculate green light duration for each lane based on vehicle counts.
6. Switch signals accordingly, ensuring that each lane gets green light time proportionate to its traffic load.
7. Repeat the process for the next cycle.

## Conclusion

The detection algorithm is the core of the traffic management system, using AI to dynamically adjust traffic signal timings and ensure that emergency vehicles are given priority. This real-time adjustment helps to reduce traffic congestion and improve response times for emergency situations.
# Traffic Management System Optimization

## Overview

The Traffic Management System is designed to optimize traffic flow by reducing redundancy and minimizing processing time. Our approach involves a dynamic algorithm that prioritizes lanes based on real-time vehicle counts, ensuring efficient use of green signal time while responding to emergency situations promptly.

## Optimization Strategy

### Frame Handling and Vehicle Detection

To streamline frame processing, our system initially checks all lanes in the first frame and calculates the number of vehicles. Based on the calculated vehicle counts and a priority-setting algorithm, the system allocates reduced green signal durations to each lane. The lane with the highest priority receives the green signal first. 

As the green light duration nears its end, the system re-evaluates the remaining three lanes. It rechecks the current frames of these lanes to detect any increase in vehicle counts. If there is a significant rise in vehicle numbers, the algorithm dynamically reallocates green signal time, potentially altering lane priorities. This process continues until all four lanes have had their turn with a green signal. A cycle is completed when each lane has been given a green signal once, after which the system resets and repeats the process.

### Emergency Vehicle Handling

The system incorporates a specialized mechanism to handle emergency vehicles. Every 5 seconds, all four lanes are scanned simultaneously to detect any emergency vehicles, such as ambulances or fire trucks. Upon detection, the system immediately switches all lanes to red and grants priority to the lane where the emergency vehicle is detected, setting it to green until the emergency vehicle passes.

## Key Features

- **Reduced Processing Time**: By skipping frames and focusing on a minimal number of frames in live feed, the system significantly reduces processing time and computational load.
- **Dynamic Lane Prioritization**: Adjusts green signal durations and lane priorities in real-time based on vehicle counts.
- **Emergency Response**: Automatically prioritizes lanes for emergency vehicles and adjusts signal timings to ensure swift passage.
## Tech Stack
**Workflow:** ![App Screenshot](https://i.postimg.cc/gjvB9jkt/Picture1.png)

**Client:** Python, OpenCV, PyTorch, YOLOv5, Manually tuned YOLOv5-s, PyQt, SciKit-Learn, Pandas, NumPy.

**Server:** React, Node, Express, JavaScript, CSS Tailwind and other frameworks.

## Screenshots

![App Screenshot](https://miro.medium.com/v2/resize:fit:828/format:webp/1*qmnZgXVuIlx9rreFjeO0sg.jpeg)
![App Screenshot](https://www.mdpi.com/mathematics/mathematics-12-01514/article_deploy/html/images/mathematics-12-01514-g007-550.jpg)
![App Screenshot](https://i.postimg.cc/RhNKfb27/Picture2.png)

## Installation
To set up the Traffic Management System, clone the repository and follow the installation instructions in the [INSTALLATION.md](INSTALLATION.md) file.
## Getting Started
To use and install the project on your device you can either download the requirements.txt (in main branch) and run it to install the necessary dependencies.

Alternatively you can visit our Website on which the code is hosted to run and execute the model without the processing and dependencies being installed on your device
## Usage

For detailed usage instructions, refer to the [USAGE.md](USAGE.md) file, which provides information on how to configure and run the system.
## Feedback

If you have any feedback, please reach out to us at 
https://forms.gle/yVQEcytCQXXDWWcb9


## FAQ

[![FAQ](https://i.postimg.cc/zBj43pgS/file.jpg)](https://jmp.sh/G6jmexEg)
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://katherineoelsner.com/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/)


## Support

For support, you can email us at :
1. asmit.ghosh.ece27@heritagit.edu.in
2. akash.shaw.ece27@heritagit.edu.in
3. mohar.chatterjee.ece27@heritagit.edu.in
4. rohan.chatterjee.ece27@heritagit.edu.in
5. vedant.thakur.ece27@heritagit.edu.in
6. bidisha.sinha.ece28@heritagit.edu.in
