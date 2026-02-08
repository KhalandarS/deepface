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
import time
from datetime import datetime
from typing import List, Dict, Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from deepface import DeepFace
from PIL import Image

# Local imports
from config import Config
from recognizer import FaceRecognizer
from attendance_manager import AttendanceManager
from detector import FaceDetector
from video_stream import VideoStream

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command line arguments (Override config if provided)
parser = argparse.ArgumentParser(description="DeepFace Attendance System")
parser.add_argument("--db-path", type=str, default=Config.DATABASE_PATH, help="Path to the database directory")
parser.add_argument("--min-confidence", type=float, default=Config.MIN_DETECTION_CONFIDENCE, help="Minimum detection confidence")
parser.add_argument("--model-selection", type=int, default=Config.MODEL_SELECTION, help="Model selection for MediaPipe (0 or 1)")
args = parser.parse_args()

# Update Config with args
Config.DATABASE_PATH = args.db_path
Config.MIN_DETECTION_CONFIDENCE = args.min_confidence
Config.MODEL_SELECTION = args.model_selection

# Initialize components
face_recognizer = FaceRecognizer(db_path=Config.DATABASE_PATH, model_name=Config.FACE_RECOG_MODEL)
attendance_manager = AttendanceManager(log_dir=Config.LOGS_DIRECTORY)
face_detector = FaceDetector(model_selection=Config.MODEL_SELECTION, min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE)

try:
    video_stream = VideoStream(0)
except ValueError:
    sys.exit(1)

def cleanup():
    """Release resources on exit."""
    if 'video_stream' in globals() and video_stream is not None:
        video_stream.release()
    cv2.destroyAllWindows()
    attendance_manager.save_records()
    logging.info("Resources released.")

atexit.register(cleanup)

# FPS calculation
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
            face_crop_resized = cv2.resize(face_crop, Config.FACE_TARGET_SIZE)

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


