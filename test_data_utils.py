"""
pytest Unit Tests for Data Utility Functions
Assignment 2 - MAI201 MLOps
Somayeh Balashi - 147025241
"""

import os
import pytest
import pandas as pd
import tempfile

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_csv, clean_phone, validate_email


# ─────────────────────────────────────────────
# Tests for load_csv()
# ─────────────────────────────────────────────

class TestLoadCsv:
    """Tests for the load_csv function."""

    def test_file_not_found(self):
        """Should raise FileNotFoundError when file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_csv("/non/existent/path/file.csv")

    def test_empty_file(self, tmp_path):
        """Should raise ValueError when file is empty."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("customer_id,age,email\n")  # header only = empty DataFrame
        with pytest.raises(ValueError, match="empty"):
            load_csv(str(empty_file))

    def test_successful_loading(self, tmp_path):
        """Should return a DataFrame with correct shape on valid CSV."""
        csv_content = "customer_id,age,email\nC001,25,test@example.com\nC002,30,other@test.com\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        df = load_csv(str(csv_file))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["customer_id", "age", "email"]

    def test_successful_loading_correct_values(self, tmp_path):
        """Should load values correctly."""
        csv_content = "customer_id,age,email\nC001,25,test@example.com\n"
        csv_file = tmp_path / "test2.csv"
        csv_file.write_text(csv_content)

        df = load_csv(str(csv_file))
        assert df.iloc[0]["customer_id"] == "C001"
        assert df.iloc[0]["age"] == 25


# ─────────────────────────────────────────────
# Tests for clean_phone()
# ─────────────────────────────────────────────

class TestCleanPhone:
    """Tests for the clean_phone function."""

    def test_dashes_format(self):
        """Should clean dashed US format."""
        assert clean_phone("555-123-4567") == "5551234567"

    def test_dots_format(self):
        """Should clean dotted format."""
        assert clean_phone("555.123.4567") == "5551234567"

    def test_parentheses_format(self):
        """Should clean (area) format."""
        assert clean_phone("(555) 123-4567") == "5551234567"

    def test_plus_one_prefix(self):
        """Should strip US country code +1."""
        assert clean_phone("+1-555-123-4567") == "5551234567"

    def test_plain_digits(self):
        """Should handle plain 10-digit input."""
        assert clean_phone("5551234567") == "5551234567"

    def test_spaces_format(self):
        """Should clean space-separated format."""
        assert clean_phone("+1 555 567 8901") == "5555678901"

    def test_australian_prefix(self):
        """Should strip Australian country code 61."""
        assert clean_phone("+61-555-890-1234") == "5558901234"

    def test_uk_prefix(self):
        """Should strip UK country code 44."""
        assert clean_phone("+44-555-789-3456") == "5557893456"

    def test_invalid_too_short(self):
        """Should return empty string for too-short input."""
        assert clean_phone("123") == ""

    def test_invalid_empty_string(self):
        """Should return empty string for empty input."""
        assert clean_phone("") == ""

    def test_invalid_none_type(self):
        """Should handle non-string input gracefully."""
        assert clean_phone(None) == ""

    def test_invalid_letters_only(self):
        """Should return empty string for letter-only input."""
        assert clean_phone("abcdefghij") == ""


# ─────────────────────────────────────────────
# Tests for validate_email()
# ─────────────────────────────────────────────

class TestValidateEmail:
    """Tests for the validate_email function."""

    def test_valid_simple_email(self):
        """Should validate a basic valid email."""
        assert validate_email("user@example.com") is True

    def test_valid_with_dots(self):
        """Should validate email with dots in local part."""
        assert validate_email("john.doe@example.com") is True

    def test_valid_with_plus(self):
        """Should validate email with plus sign."""
        assert validate_email("user+tag@domain.org") is True

    def test_valid_subdomain(self):
        """Should validate email with subdomain."""
        assert validate_email("user@mail.domain.co.uk") is True

    def test_valid_io_tld(self):
        """Should validate .io TLD."""
        assert validate_email("user@domain.io") is True

    def test_invalid_missing_at(self):
        """Should reject email missing @ symbol."""
        assert validate_email("invalidemail.com") is False

    def test_invalid_missing_domain(self):
        """Should reject email with no domain after @."""
        assert validate_email("user@") is False

    def test_invalid_missing_tld(self):
        """Should reject email with no TLD."""
        assert validate_email("user@domain") is False

    def test_invalid_double_at(self):
        """Should reject email with two @ symbols."""
        assert validate_email("user@@domain.com") is False

    def test_invalid_empty_string(self):
        """Should return False for empty string."""
        assert validate_email("") is False

    def test_invalid_none(self):
        """Should return False for None."""
        assert validate_email(None) is False

    def test_invalid_spaces(self):
        """Should reject email with spaces (not stripped)."""
        assert validate_email("user @domain.com") is False

    def test_edge_case_numeric_local(self):
        """Should accept numeric local part."""
        assert validate_email("123@domain.com") is True
