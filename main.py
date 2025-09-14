import cv2
import os
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import mediapipe as mp
from PIL import Image
import numpy as np

# Directories and paths
DB_PATH = "students_db"
LOG_DIR = "attendance_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Load known student images and data
student_data = []
for filename in os.listdir(DB_PATH):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        try:
            path = os.path.join(DB_PATH, filename)
            roll, name = filename.rsplit(".", 1)[0].split("_", 1)
            img = Image.open(path)
            img.verify()  # Ensure image isn't corrupted
            student_data.append({"roll": roll, "name": name, "path": path})
        except Exception as e:
            print(f"Skipping file {filename}: {e}")

# Initialize attendance tracking
today_str = datetime.now().strftime("%Y-%m-%d")
excel_path = os.path.join(LOG_DIR, f"attendance_{today_str}.xlsx")
attendance_records = []

# Set to prevent multiple entries per student
marked_students = set()

# Mediapipe face detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

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
            face_crop_resized = cv2.resize(face_crop, (160, 160))

            try:
                # Perform facial recognition using DeepFace
                matches = DeepFace.find(img_path=face_crop_resized, db_path=DB_PATH, model_name="Facenet", enforce_detection=False, silent=True)

                if len(matches[0]) > 0:
                    identity_path = matches[0].iloc[0]['identity']
                    identity_filename = os.path.basename(identity_path)
                    try:
                        roll, name = identity_filename.rsplit(".", 1)[0].split("_", 1)
                    except ValueError:
                        roll, name = "UNK", "Unknown"

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
    df.to_excel(excel_path, index=False)
    print(f"\n✅ Attendance saved to {excel_path}")
else:
    print("\n⚠️ No attendance recorded today.")

cap.release()
cv2.destroyAllWindows()


