"""
Unit tests for attendance_manager module.
"""
import unittest
import os
import shutil
from datetime import datetime
from attendance_manager import AttendanceManager


class TestAttendanceManager(unittest.TestCase):
    """Test cases for AttendanceManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = "test_attendance_logs"
        self.manager = AttendanceManager(log_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_mark_attendance_success(self):
        """Test successful attendance marking."""
        result = self.manager.mark_attendance("101", "John Doe")
        self.assertTrue(result)
        self.assertIn("101", self.manager.marked_students)
    
    def test_mark_attendance_duplicate(self):
        """Test duplicate attendance marking returns False."""
        self.manager.mark_attendance("101", "John Doe")
        result = self.manager.mark_attendance("101", "John Doe")
        self.assertFalse(result)
    
    def test_save_records_empty(self):
        """Test saving with no records."""
        # Should not raise an exception
        self.manager.save_records()


if __name__ == "__main__":
    unittest.main()
