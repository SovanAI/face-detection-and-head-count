# Gate People Counter

A simple real-time people counter built with Python, OpenCV, and FastAPI. It detects moving people from a webcam, tracks them across a virtual gate line, and counts how many people entered and exited the scene.

## Overview

This project creates a web dashboard that shows:

- Live camera stream
- Entry counter
- Exit counter
- Current number of people inside
- Estimated crossing direction

The counting logic is handled in `people_counter.py`, while the web UI and API routes are defined in `main.py`.

## Features

- Real-time video stream from webcam
- Background subtraction-based motion detection
- Object tracking across a gate line
- Entry/exit counting logic
- Web dashboard with live stats
- Reset endpoint for clearing counters
- Camera index configuration via environment variable

## Project Structure

```text
face_detection_hackethon/
├── main.py              # FastAPI app and web interface
├── people_counter.py     # Computer vision and counting logic
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── .venv/               # Optional local virtual environment
```

## Requirements

- Python 3.10+
- Webcam or camera device
- OpenCV and FastAPI dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd face_detection_hackethon
```

2. Create and activate a virtual environment:

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the server:

```bash
python main.py
```

Or use Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

```text
http://localhost:8000/
```

## Configuration

The app can use a different camera source by setting the `CAMERA_INDEX` environment variable.

Example:

```bash
set CAMERA_INDEX=1
python main.py
```

or on Linux/macOS:

```bash
export CAMERA_INDEX=1
python main.py
```

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Web dashboard UI |
| `/video` | GET | Live video stream |
| `/stats` | GET | JSON with entry/exit counts and inside total |
| `/reset` | POST | Reset the counting state |

Example stats response:

```json
{
  "entry_count": 12,
  "exit_count": 7,
  "entry_direction": "left_to_right",
  "exit_direction": "right_to_left",
  "current_inside": 5
}
```

## How It Works

1. The webcam captures frames continuously.
2. Background subtraction detects moving objects.
3. The system tracks each detected object by centroid.
4. A virtual gate line is drawn across the frame.
5. When an object crosses the line, the app determines the direction.
6. The counter increments entry or exit based on the established movement pattern.

## Notes

- The tracking logic is designed for basic indoor gate-style monitoring.
- Lighting conditions, camera angle, and object size can affect detection accuracy.
- For better results, use a stable camera position with a clear entrance/exit path.

## License

This project is provided as a base implementation for hackathon or learning purposes.
