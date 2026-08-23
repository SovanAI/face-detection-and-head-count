# HAWKEYE GateWatch - Advanced Gate People Counter

A high-performance, real-time people counter and occupancy monitoring dashboard designed for basic indoor gate-style tracking. Built with **Python**, **OpenCV**, and **FastAPI**, this system provides advanced human detection, centroid tracking, and automated crossing/occupancy logic, accessible via an interactive web dashboard.

---

## 🌟 Key Features

- **Real-Time Video Stream**: Live web stream displaying camera feed with a custom HUD, bounding boxes, target IDs, and historical motion trails.
- **Three Detection Modes**:
  - **Motion Detection (`motion`)**: Uses a Gaussian blur and MOG2 background subtractor combined with morphological opening/closing/dilation for high-speed motion tracking.
  - **HOG Detector (`hog`)**: Utilizes standard Histogram of Oriented Gradients (HOG) descriptor with a linear Support Vector Machine (SVM) classifier for human detection.
  - **Hybrid Mode (`hybrid`)**: High-efficiency mode that runs motion detection first, extracts moving Regions of Interest (ROIs), and runs the HOG classifier only on those small regions, delivering both speed and accuracy.
- **Centroid Tracking & Motion History**:
  - Euclidean distance-based centroid tracking.
  - Blinking state indicators, random track colors, and custom corner bounding boxes.
  - Particle trails tracking up to 20 past coordinates for each active target.
  - A customizable disappeared frame budget (up to 12 frames) to handle brief occlusions.
- **Dynamic Virtual Gate Line**:
  - Supports both **Vertical** (left/right) and **Horizontal** (top/bottom) orientations.
  - Dynamic line position slider adjustable from 10% to 90% of frame dimensions.
- **Auto-calibration & Counting**:
  - Automatically establishes entry and exit directions based on the first recorded crossing.
  - Tracks total entries, exits, and current occupancy inside the monitored area.
- **High-Tech Slate-Blue UI Dashboard**: Fully-responsive web interface featuring dynamic glassmorphism cards, real-time statistics polling, system configurations, and reset actions.

---

## 📂 Project Structure

- **[main.py](file:///e:/face_detection_hackethon/main.py)**: Configures the FastAPI server, serves the interactive HTML/JavaScript dashboard, streams MJPEG frames, and handles configuration updates.
- **[people_counter.py](file:///e:/face_detection_hackethon/people_counter.py)**: Houses the core [`PeopleCounter`](file:///e:/face_detection_hackethon/people_counter.py#L7) class where the OpenCV frame capturing, human detection, centroid tracking, HUD rendering, and gate crossing logic are executed.
- **[requirements.txt](file:///e:/face_detection_hackethon/requirements.txt)**: Specifies the Python environment dependencies.

---

## 🛠️ Technical Details

### 1. Detection Pipelines
The system uses the [`PeopleCounter.detect_people`](file:///e:/face_detection_hackethon/people_counter.py#L252) method to run the selected detection engine:
- **Motion Core**: Applies `cv2.GaussianBlur` followed by `cv2.createBackgroundSubtractorMOG2`. Thresholds to binary mask, performs morphology closing and opening to reduce noise, dilates to connect segmented body parts, and merges bounding boxes within 45px distance.
- **HOG SVM Core**: Resizes frame internally to 640px width (maintaining aspect ratio) for speed, runs SVM multi-scale pedestrian detection, and applies Non-Maximum Suppression (`cv2.dnn.NMSBoxes`) to eliminate duplicate overlap.
- **Hybrid Core**: Restricts HOG SVM detection exclusively to ROIs identified by the motion processor, speeding up classification substantially.

### 2. Centroid Tracking
- Centroid matching is handled in [`PeopleCounter.update_tracking`](file:///e:/face_detection_hackethon/people_counter.py#L265) via Euclidean distance calculations between existing track centroids and incoming frame detections.
- Unmatched targets are assigned a new ID, color, and tracking history.
- If a target is undetected for 12 consecutive frames, its track is deleted.

### 3. Crossing Logic
- [`PeopleCounter.check_crossings`](file:///e:/face_detection_hackethon/people_counter.py#L349) tracks lines crossings. When a centroid transitions across the dynamic gate line:
  - If it is the first crossing event, the system locks this direction as the standard entry vector (and the opposite as the exit vector).
  - Subsequent crossings increment the respective entry or exit counters.

---

## 🚀 Installation & Setup

### Requirements
- Python 3.10+
- A connected USB webcam or integrated camera device
- OpenCV and FastAPI dependencies (defined in [requirements.txt](file:///e:/face_detection_hackethon/requirements.txt))

### Step-by-Step Installation

1. **Clone/Navigate to the directory**:
   ```bash
   cd face_detection_hackethon
   ```

2. **Create and Activate a Virtual Environment**:
   - **Windows**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Run the Application

Start the FastAPI application by running:
```bash
python main.py
```
Or run directly via Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, navigate to:
```text
http://localhost:8000/
```

### ⚙️ Camera Index Configuration
By default, the application captures from camera index `0`. You can override this using the `CAMERA_INDEX` environment variable.

- **Windows Command Prompt**:
  ```cmd
  set CAMERA_INDEX=1
  python main.py
  ```
- **Windows PowerShell**:
  ```powershell
  $env:CAMERA_INDEX="1"
  python main.py
  ```
- **macOS/Linux Shell**:
  ```bash
  export CAMERA_INDEX=1
  python main.py
  ```

---

## 📡 API Endpoints

| Endpoint | Method | Payload (JSON) | Description |
| --- | --- | --- | --- |
| `/` | GET | *None* | Serves the web dashboard UI template. |
| `/video` | GET | *None* | MJPEG multipart live video stream feed. |
| `/stats` | GET | *None* | Returns JSON with current counts, directions, occupancy, and active settings. |
| `/config` | POST | `ConfigUpdate` model | Updates the detection mode, gate orientation, and line position on-the-fly. |
| `/reset` | POST | *None* | Resets all counts, directions, history, and tracks. |

### Config Update JSON Payload Example
```json
{
  "detection_mode": "hybrid",
  "line_orientation": "vertical",
  "line_position": 0.5
}
```

### Stats JSON Response Example
```json
{
  "entry_count": 5,
  "exit_count": 2,
  "entry_direction": "left_to_right",
  "exit_direction": "right_to_left",
  "current_inside": 3,
  "detection_mode": "hybrid",
  "line_orientation": "vertical",
  "line_position": 0.5
}
```

---

## 🛡️ License & Team Info

This application is provided as a base implementation for Smart India Hackathon (SIH) purposes.  
**Team Name**: HAWKEYE
