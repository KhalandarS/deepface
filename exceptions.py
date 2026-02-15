"""
Custom exception classes for the attendance system.
"""


class AttendanceSystemError(Exception):
    """Base exception for attendance system errors."""
    pass


class DatabaseNotFoundError(AttendanceSystemError):
    """Raised when the student database directory is not found."""
    pass


class VideoStreamError(AttendanceSystemError):
    """Raised when there are issues with the video stream."""
    pass


class FaceRecognitionError(AttendanceSystemError):
    """Raised when face recognition fails."""
    pass


class AttendanceLogError(AttendanceSystemError):
    """Raised when there are issues saving attendance logs."""
    pass
