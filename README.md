# Face Recognition Attendance System

This project implements a smart attendance system using DeepFace for facial recognition and MediaPipe for face detection.

## Features
- **Real-time Face Detection**: Uses MediaPipe for fast and accurate face detection.
- **Face Recognition**: Utilizes DeepFace (default: Facenet) to identify registered students.
- **Attendance Logging**: Automatically marks attendance in an Excel sheet, ensuring unique entries per session.
- **Live Video Feed**: Displays camera feed with bounding boxes and names.

## Prerequisites
- Python 3.8+
- Webcam

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KhalandarS/deepface.git
   cd deepface
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment:
   - Copy `.env.example` to `.env` (optional, defaults are provided in `config.py`).
   - Create a `students_db` folder and add student images named as `RollNumber_Name.jpg` (e.g., `101_JohnDoe.jpg`).

## Usage

Run the main script:
```bash
python main.py
```

Arguments:
- `--db-path`: Path to the database directory (default: `students_db`).
- `--min-confidence`: Minimum detection confidence (default: 0.5).
- `--model-selection`: MediaPipe model selection (0 for close range, 1 for far).

## Utilities
- `check_camera.py`: Verify webcam functionality.

