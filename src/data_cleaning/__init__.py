"""
Data cleaning module for ASEWIS.

Provides robust, deterministic data cleaning for Aadhaar location data
without modifying existing pipeline logic.
"""

import sys
from pathlib import Path

# Import from the parent data_cleaning.py file
parent_module_path = Path(__file__).parent.parent / 'data_cleaning.py'
if parent_module_path.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_cleaning_legacy", parent_module_path)
    data_cleaning_legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_cleaning_legacy)
    clean_dataframe = data_cleaning_legacy.clean_dataframe
else:
    clean_dataframe = None

from .location_cleaner import clean_location_columns

__all__ = ['clean_location_columns', 'clean_dataframe']
