import mediapipe as mp
import cv2
import numpy as np
from typing import List, Tuple, Any

class FaceDetector:
    """
    Wrapper for MediaPipe Face Detection.
    """
    def __init__(self, model_selection: int = 0, min_detection_confidence: float = 0.5) -> None:
        """
        Initialize the FaceDetector.
        
        Args:
            model_selection (int): 0 for short-range (within 2 meters), 1 for long-range (within 5 meters).
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for face detection.
        """
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=model_selection, 
            min_detection_confidence=min_detection_confidence
        )
    
    def process_frame(self, frame: np.ndarray) -> Any:
        """
        Process the frame to detect faces.
        
        Args:
            frame (np.ndarray): The input image frame.
            
        Returns:
            Any: The detection results from MediaPipe.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.face_detector.process(rgb_frame)

    @staticmethod
    def extract_faces(frame: np.ndarray, detections: Any) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract face crops and coordinates from MediaPipe detections.
        
        Args:
            frame (np.ndarray): The input image frame.
            detections (Any): THe detection results.
            
        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int]]]: A list of tuples containing the face crop and its bounding box (x1, y1, x2, y2).
        """
        extracted = []
        if not detections:
            return extracted
            
        h, w, _ = frame.shape
        for detection in detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = int(x1 + bboxC.width * w)
            y2 = int(y1 + bboxC.height * h)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                 continue
                 
            extracted.append((face_crop, (x1, y1, x2, y2)))
        return extracted
