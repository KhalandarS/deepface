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



def parse_filename(filename: str) -> tuple[str, str]:
    """Parse filename to extract roll number and name."""
    try:
        base = filename.rsplit(".", 1)[0]
        if "_" in base:
            return base.split("_", 1)
        return "UNK", base
    except Exception:
        return "UNK", "Unknown"

def load_student_db(db_path: str) -> List[Dict[str, Any]]:
    """Load student images and data from the database directory."""
    data: List[Dict[str, Any]] = []
    if not os.path.exists(db_path):
        logging.warning(f"Database path '{db_path}' does not exist.")
        return data

    for filename in os.listdir(db_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                path = os.path.join(db_path, filename)
                roll, name = parse_filename(filename)
                
                img = Image.open(path)
                img.verify()  # Ensure image isn't corrupted
                data.append({"roll": roll, "name": name, "path": path})
            except Exception as e:
                logging.warning(f"Skipping file {filename}: {e}")
    return data



def recognize_face(face_img: np.ndarray, db_path: str, model_name: str) -> tuple[str, str]:
    """
    Recognize a face against the database using DeepFace.

    Args:
        face_img: The face image crop.
        db_path: Path to the student database.
        model_name: The DeepFace model to use.

    Returns:
        tuple[str, str]: A tuple containing (RollNumber, Name). Returns ("UNK", "Unknown") if not recognized.
    """
    try:
         matches = DeepFace.find(img_path=face_img, db_path=db_path, model_name=model_name, enforce_detection=False, silent=True)
         if len(matches[0]) > 0:
             identity_path = matches[0].iloc[0]['identity']
             identity_filename = os.path.basename(identity_path)
             return parse_filename(identity_filename)
    except Exception as e:
         logging.error(f"Recognition error: {e}")
    return "UNK", "Unknown"

# Load known student images and data
student_data = load_student_db(DATABASE_PATH)
logging.info(f"Loaded {len(student_data)} students from {DATABASE_PATH}")


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

            roll, name = recognize_face(face_crop_resized, DATABASE_PATH, FACE_RECOG_MODEL)
            
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

    # Display the live feed
    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


