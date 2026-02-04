import cv2
import logging

class VideoStream:
    """
    Handles video capture from a webcam/device.
    """
    def __init__(self, source: int = 0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logging.error(f"Could not open video source {self.source}.")
            raise ValueError(f"Could not open video source {self.source}")

    def read(self):
        """Reads a frame from the stream."""
        return self.cap.read()

    def release(self):
        """Releases the video source."""
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Video stream released.")

    def is_opened(self) -> bool:
        return self.cap.isOpened()
