"""
Data Utility Functions for Assignment 2 - MAI201 MLOps
Somayeh Balashi - 147025241
"""

import pandas as pd
import re
import os


def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)

    if df.empty:
        raise ValueError(f"The file is empty: {filepath}")

    return df


def clean_phone(phone: str) -> str:
    """
    Clean and normalize a phone number to digits only.

    Args:
        phone: Raw phone number string in any format.

    Returns:
        str: Digits-only phone number, or empty string if invalid.
    """
    if not isinstance(phone, str) or not phone.strip():
        return ""

    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)

    # Remove leading country codes (1 for US/Canada, 61 for AU, 44 for UK)
    if digits.startswith('1') and len(digits) == 11:
        digits = digits[1:]
    elif digits.startswith('61') and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith('44') and len(digits) == 12:
        digits = digits[2:]

    # Return only if exactly 10 digits remain
    if len(digits) == 10:
        return digits

    return ""


def validate_email(email: str) -> bool:
    """
    Validate an email address using regex.

    Args:
        email: Email address string.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(email, str) or not email.strip():
        return False

    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))
