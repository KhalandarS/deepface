"""
Main script for Face Recognition Attendance System using DeepFace.

Expected Database Structure:
- Store images in the `students_db` directory (or specify via --db-path).
- Filenames should follow the format: `RollNumber_Name.jpg` (e.g., `101_JohnDoe.jpg`).
"""
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from deepface import DeepFace
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(description="DeepFace Attendance System")
parser.add_argument("--db-path", type=str, default="students_db", help="Path to the database directory")
args = parser.parse_args()

# Directories and paths
DATABASE_PATH = args.db_path
LOGS_DIRECTORY = "attendance_logs"
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# Configuration
FACE_RECOG_MODEL = "Facenet"
MIN_DETECTION_CONFIDENCE = 0.5
MODEL_SELECTION = 0
FACE_TARGET_SIZE = (160, 160)



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
        print(f"Warning: Database path '{db_path}' does not exist.")
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
                print(f"Skipping file {filename}: {e}")
    return data

# Load known student images and data
student_data = load_student_db(DATABASE_PATH)
print(f"Loaded {len(student_data)} students from {DATABASE_PATH}")


# Initialize attendance tracking
today_str = datetime.now().strftime("%Y-%m-%d")
excel_path = os.path.join(LOGS_DIRECTORY, f"attendance_{today_str}.xlsx")
attendance_records = []

# Set to prevent multiple entries per student
marked_students = set()

# Mediapipe face detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=MODEL_SELECTION, min_detection_confidence=MIN_DETECTION_CONFIDENCE)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

print("Starting Attendance System. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int(x1 + bboxC.width * w)
            y2 = int(y1 + bboxC.height * h)

            face_crop = frame[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]

            # Resize face for consistent recognition
            face_crop_resized = cv2.resize(face_crop, FACE_TARGET_SIZE)

            try:
                # Perform facial recognition using DeepFace
                matches = DeepFace.find(img_path=face_crop_resized, db_path=DATABASE_PATH, model_name=FACE_RECOG_MODEL, enforce_detection=False, silent=True)

                if len(matches[0]) > 0:
                    identity_path = matches[0].iloc[0]['identity']
                    identity_filename = os.path.basename(identity_path)
                    roll, name = parse_filename(identity_filename)

                    label = f"{roll} - {name}"

                    # Mark attendance only once per student
                    if roll not in marked_students:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        attendance_records.append({"Roll": roll, "Name": name, "Time": timestamp})
                        marked_students.add(roll)
                        print(f"[✓] Marked: {label} at {timestamp}")
                else:
                    label = "Unknown"
            except Exception as e:
                label = "Unknown"
                print("Error:", e)

            # Draw label and rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the live feed
    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save Excel file at the end
if attendance_records:
    df = pd.DataFrame(attendance_records)
    try:
        df.to_excel(excel_path, index=False)
        print(f"\n✅ Attendance saved to {excel_path}")
    except PermissionError:
        print(f"\n❌ Error: Could not save to {excel_path}. Is the file open?")
    except Exception as e:
        print(f"\n❌ Failed to save attendance: {e}")
else:
    print("\n⚠️ No attendance recorded today.")

cap.release()
cv2.destroyAllWindows()


