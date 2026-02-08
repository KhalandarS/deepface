import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Configuration settings for the application.
    Loads environment variables or uses default values.
    """
    DATABASE_PATH = os.getenv("DATABASE_PATH", "students_db")
    LOGS_DIRECTORY = os.getenv("LOGS_DIRECTORY", "attendance_logs")
    MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.5"))
    MODEL_SELECTION = int(os.getenv("MODEL_SELECTION", "0"))
    FACE_RECOG_MODEL = os.getenv("FACE_RECOG_MODEL", "Facenet")
    FACE_TARGET_SIZE = (160, 160)
