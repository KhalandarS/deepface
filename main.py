"""
Main script for Face Recognition Attendance System using DeepFace.

Expected Database Structure:
- Store images in the `students_db` directory (or specify via --db-path).
- Filenames should follow the format: `RollNumber_Name.jpg` (e.g., `101_JohnDoe.jpg`).
"""
import argparse
import logging
import os
import sys
import atexit
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from deepface import DeepFace
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(description="DeepFace Attendance System")
parser.add_argument("--db-path", type=str, default="students_db", help="Path to the database directory")
parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum detection confidence")
parser.add_argument("--model-selection", type=int, default=0, help="Model selection for MediaPipe (0 or 1)")
args = parser.parse_args()

# Directories and paths
DATABASE_PATH = args.db_path
LOGS_DIRECTORY = "attendance_logs"
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# Configuration
FACE_RECOG_MODEL: str = "Facenet"
MIN_DETECTION_CONFIDENCE: float = args.min_confidence
MODEL_SELECTION: int = args.model_selection
FACE_TARGET_SIZE: tuple[int, int] = (160, 160)

from recognizer import FaceRecognizer
face_recognizer = FaceRecognizer(db_path=DATABASE_PATH, model_name=FACE_RECOG_MODEL)










# Initialize attendance manager
from attendance_manager import AttendanceManager
attendance_manager = AttendanceManager(log_dir=LOGS_DIRECTORY)

def cleanup():
    """Release resources on exit."""
    if 'video_stream' in globals() and video_stream is not None:
        video_stream.release()
    cv2.destroyAllWindows()
    attendance_manager.save_records()
    logging.info("Resources released.")

atexit.register(cleanup)


# Face detection
from detector import FaceDetector
face_detector = FaceDetector(model_selection=MODEL_SELECTION, min_detection_confidence=MIN_DETECTION_CONFIDENCE)

from video_stream import VideoStream
try:
    video_stream = VideoStream(0)
except ValueError:
    sys.exit(1)


# FPS calculation
import time
prev_frame_time = 0
new_frame_time = 0

logging.info("Starting Attendance System. Press 'q' to quit...")

while True:

    ret, frame = video_stream.read()
    if not ret:
        break

    results = face_detector.process_frame(frame)

    if results.detections:
        for face_crop, (x1, y1, x2, y2) in FaceDetector.extract_faces(frame, results.detections):

            # Resize face for consistent recognition
            face_crop_resized = cv2.resize(face_crop, FACE_TARGET_SIZE)

            roll, name = face_recognizer.recognize(face_crop_resized)
            
            if roll != "UNK":
                label = f"{roll} - {name}"
                color = (0, 255, 0)  # Green for known

                # Mark attendance using manager
                attendance_manager.mark_attendance(roll, name)
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            # Draw label and rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    fps_str = str(int(fps))

    # Display FPS
    cv2.putText(frame, f"FPS: {fps_str}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # Display the live feed
    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


