"""
Utility module for attendance analysis and reporting.
"""
import pandas as pd
import os
from datetime import datetime
from typing import List, Dict


def generate_daily_report(log_dir: str = "attendance_logs", date: str = None) -> pd.DataFrame:
    """
    Generate a daily attendance report.
    
    Args:
        log_dir (str): Directory containing attendance logs.
        date (str): Date in YYYY-MM-DD format. If None, uses today's date.
        
    Returns:
        pd.DataFrame: DataFrame containing the attendance records.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    file_path = os.path.join(log_dir, f"attendance_{date}.xlsx")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No attendance file found for {date}")
    
    return pd.read_excel(file_path)


def get_attendance_statistics(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate basic statistics from attendance data.
    
    Args:
        df (pd.DataFrame): Attendance DataFrame.
        
    Returns:
        Dict[str, int]: Dictionary containing statistics.
    """
    stats = {
        "total_students": len(df),
        "unique_rolls": df["Roll"].nunique() if "Roll" in df.columns else 0
    }
    
    return stats
