"""
Data Utilities

This script contains utility functions to load reference and current data from CSV files
and extract specific columns.

Functions:
    - load_reference_data: Load reference data from a CSV file and extract specific columns.
    - load_current_data: Load current data from a CSV file and extract specific columns.
"""

from typing import List, Text

import pandas as pd


def load_reference_data(columns: List[Text]) -> pd.DataFrame:
    """
    Load reference data from a CSV file and extract specific columns.

    Args:
        columns (List[Text]): List of column names to extract from the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the reference data with selected columns.
    """
    cur_path = "/home/konradballegro/data/scored/offers_scored_reference.csv"
    cur_data = pd.read_csv(cur_path)
    current_data = cur_data.loc[:, columns]
    return current_data


def load_current_data(columns: List[Text]) -> pd.DataFrame:
    """
    Load current data from a CSV file and extract specific columns.

    Args:
        columns (List[Text]): List of column names to extract from the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the current data with selected columns.
    """
    ref_path = "/home/konradballegro/data/scored/offers_scored_current.csv"
    ref_data = pd.read_csv(ref_path)
    reference_data = ref_data.loc[:, columns]
    return reference_data
