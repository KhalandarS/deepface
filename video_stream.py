import cv2
import logging

class VideoStream:
    """
    Handles video capture from a webcam/device.
    """
    def __init__(self, source: int = 0) -> None:
        """
        Initialize the VideoStream.
        
        Args:
            source (int): The video source index (e.g., 0 for webcam).
        """
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logging.error(f"Could not open video source {self.source}.")
            raise ValueError(f"Could not open video source {self.source}")

    def read(self) -> tuple[bool, cv2.Mat]:
        """
        Reads a frame from the stream.
        
        Returns:
            tuple[bool, cv2.Mat]: A tuple containing the return status (True if successful) and the frame.
        """
        return self.cap.read()

    def release(self):
        """Releases the video source."""
        if self.cap.isOpened():
            self.cap.release()
            logging.info("Video stream released.")

    def is_opened(self) -> bool:
        return self.cap.isOpened()
