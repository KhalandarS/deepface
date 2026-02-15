# Project Architecture

## Overview
The Face Recognition Attendance System is structured into modular components for maintainability and extensibility.

## Directory Structure
```
deepface/
├── main.py                    # Entry point
├── config.py                  # Configuration management
├── constants.py               # Application constants
├── exceptions.py              # Custom exceptions
├── attendance_manager.py      # Attendance tracking
├── detector.py                # Face detection (MediaPipe)
├── recognizer.py              # Face recognition (DeepFace)
├── video_stream.py            # Video capture handling
├── utils.py                   # Utility functions
├── check_camera.py            # Camera testing utility
├── test_attendance_manager.py # Unit tests
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
├── README.md                  # Project documentation
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # License file
└── setup.sh                   # Setup script
```

## Core Components

### 1. Video Stream (`video_stream.py`)
- Handles camera initialization and frame capture
- Provides error handling for camera failures

### 2. Face Detector (`detector.py`)
- Uses MediaPipe for real-time face detection
- Extracts face crops and bounding boxes

### 3. Face Recognizer (`recognizer.py`)
- Uses DeepFace for face recognition
- Matches detected faces against database

### 4. Attendance Manager (`attendance_manager.py`)
- Tracks attendance per session
- Prevents duplicate entries
- Saves records to Excel

### 5. Configuration (`config.py`)
- Centralized configuration using environment variables
- Provides sensible defaults

## Data Flow
```
Camera → VideoStream → FaceDetector → FaceRecognizer → AttendanceManager → Excel
```

## Design Patterns
- **Singleton-like**: Configuration is loaded once at startup
- **Dependency Injection**: Components receive dependencies via constructors
- **Error Handling**: Custom exceptions for different error types
