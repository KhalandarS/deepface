import os
import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Set

class AttendanceManager:
    """
    Manages attendance records, ensuring uniqueness and saving to file.
    """
    def __init__(self, log_dir: str = "attendance_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.marked_students: Set[str] = set()
        self.attendance_records: List[Dict[str, str]] = []
        
        today_str = datetime.now().strftime("%Y-%m-%d")
        self.excel_path = os.path.join(self.log_dir, f"attendance_{today_str}.xlsx")

    def mark_attendance(self, roll: str, name: str) -> bool:
        """
        Marks attendance for a student if not already marked.
        Returns True if marked nicely, False if already marked.
        """
        if roll not in self.marked_students:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.attendance_records.append({
                "Roll": roll,
                "Name": name,
                "Time": timestamp
            })
            self.marked_students.add(roll)
            logging.info(f"Marked: {roll} - {name} at {timestamp}")
            return True
        return False

    def save_records(self):
        """Saves the attendance records to an Excel file."""
        if not self.attendance_records:
            logging.warning("No attendance recorded to save.")
            return

        df = pd.DataFrame(self.attendance_records)
        try:
            df.to_excel(self.excel_path, index=False)
            logging.info(f"Attendance saved to {self.excel_path}")
        except PermissionError:
            logging.error(f"Could not save to {self.excel_path}. Is the file open?")
        except Exception as e:
            logging.error(f"Failed to save attendance: {e}")
