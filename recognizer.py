from deepface import DeepFace
import pandas as pd
import logging
import os
import numpy as np

class FaceRecognizer:
    """
    Wrapper for DeepFace Recognition.
    """
    def __init__(self, db_path: str, model_name: str = "Facenet"):
        self.db_path = db_path
        self.model_name = model_name
        
    def recognize(self, face_img: np.ndarray) -> tuple[str, str]:
        """
        Recognize a face against the database.
        Returns: (RollNumber, Name) or ("UNK", "Unknown")
        """
        try:
             matches = DeepFace.find(
                 img_path=face_img, 
                 db_path=self.db_path, 
                 model_name=self.model_name, 
                 enforce_detection=False, 
                 silent=True
             )
             
             if len(matches[0]) > 0:
                 identity_path = matches[0].iloc[0]['identity']
                 identity_filename = os.path.basename(identity_path)
                 return self._parse_filename(identity_filename)
        except Exception as e:
             logging.error(f"Recognition error: {e}")
             
        return "UNK", "Unknown"

    @staticmethod
    def _parse_filename(filename: str) -> tuple[str, str]:
        """Parse filename to extract roll number and name."""
        try:
            base = filename.rsplit(".", 1)[0]
            if "_" in base:
                return base.split("_", 1)
            return "UNK", base
        except Exception:
            return "UNK", "Unknown"
