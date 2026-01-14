"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path

import pytest

from brainfs.database import DocumentDatabase


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def test_db(temp_dir):
    """Create a test database in a temporary directory."""
    db_path = temp_dir / "test.db"
    return DocumentDatabase(db_path)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Python is a high-level programming language.
    It supports object-oriented and functional programming.
    Python has extensive libraries for data science.
    """


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Test Document

This is a **test** document with:
- Lists
- *Emphasis*
- Code: `print("hello")`

## Section Two
More content here."""


@pytest.fixture
def sample_files(temp_dir):
    """Create sample files for testing."""
    # Create sample text file
    txt_file = temp_dir / "sample.txt"
    txt_file.write_text("This is a sample text file for testing purposes.")

    # Create sample markdown file
    md_file = temp_dir / "sample.md"
    md_file.write_text("""# Sample Document

This is a sample markdown document.

## Features
- Feature one
- Feature two""")

    return {"txt": txt_file, "md": md_file}
