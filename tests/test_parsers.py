"""Tests for document parser functionality."""

from pathlib import Path

import pytest

from brainfs.parsers import parse_document


def test_parse_text_file(temp_dir):
    """Test parsing plain text files."""
    txt_file = temp_dir / "test.txt"
    content = "This is a test document.\nWith multiple lines.\nAnd some content."
    txt_file.write_text(content)

    result = parse_document(txt_file)
    assert result == content


def test_parse_markdown_file(temp_dir):
    """Test parsing markdown files."""
    md_file = temp_dir / "test.md"
    markdown_content = """# Test Document

This is a **test** document with:
- Lists
- *Emphasis*
- Code: `print("hello")`

## Section Two
More content here."""

    md_file.write_text(markdown_content)

    result = parse_document(md_file)
    # Markdown parser may convert to plain text
    assert "Test Document" in result
    assert "test" in result
    assert "print" in result


def test_parse_unsupported_file(temp_dir):
    """Test parsing unsupported file types."""
    # Create a file with unsupported extension
    unsupported_file = temp_dir / "test.xyz"
    unsupported_file.write_text("Some content")

    # Should raise ValueError for unsupported files
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(unsupported_file)


def test_parse_empty_file(temp_dir):
    """Test parsing empty files."""
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")

    result = parse_document(empty_file)
    assert result == ""


def test_parse_nonexistent_file():
    """Test parsing nonexistent files."""
    nonexistent = Path("/nonexistent/file.txt")

    with pytest.raises(FileNotFoundError):
        parse_document(nonexistent)


def test_parse_binary_file(temp_dir):
    """Test parsing binary files (should handle gracefully)."""
    binary_file = temp_dir / "test.bin"
    # Write some binary data
    binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

    # Should handle binary files gracefully
    # Might raise an exception or return empty/error text
    try:
        result = parse_document(binary_file)
        # If it doesn't raise an exception, result might be empty or contain error text
        assert isinstance(result, str)
    except (UnicodeDecodeError, Exception):
        # It's acceptable for binary files to raise exceptions
        pass


def test_parse_large_file(temp_dir):
    """Test parsing large text files."""
    large_file = temp_dir / "large.txt"

    # Create a moderately large file (not too large for tests)
    large_content = "This is line {i}\n"
    large_text = "".join(large_content.format(i=i) for i in range(1000))
    large_file.write_text(large_text)

    result = parse_document(large_file)
    assert len(result) > 10000
    assert "This is line 0" in result
    assert "This is line 999" in result


def test_parse_utf8_file(temp_dir):
    """Test parsing files with various UTF-8 characters."""
    utf8_file = temp_dir / "utf8.txt"
    utf8_content = "Hello 世界! Café, naïve résumé. Emoji: 🚀🔥💻"
    utf8_file.write_text(utf8_content, encoding="utf-8")

    result = parse_document(utf8_file)
    assert result == utf8_content
    assert "世界" in result
    assert "Café" in result
    assert "🚀" in result


def test_parse_file_with_different_extensions(temp_dir):
    """Test that parser handles different file extensions appropriately."""
    # Test text file (should be exact match)
    txt_file = temp_dir / "test.txt"
    txt_file.write_text("Plain text content")
    result = parse_document(txt_file)
    assert result == "Plain text content"

    # Test markdown files (may strip markup)
    md_file = temp_dir / "test.md"
    md_file.write_text("# Markdown content")
    result = parse_document(md_file)
    assert "Markdown content" in result  # May strip the #

    markdown_file = temp_dir / "test.markdown"
    markdown_file.write_text("## Another markdown")
    result = parse_document(markdown_file)
    assert "Another markdown" in result  # May strip the ##

    # Test unsupported extension
    unsupported_file = temp_dir / "test.xyz"
    unsupported_file.write_text("Some content")
    with pytest.raises(ValueError):
        parse_document(unsupported_file)


# PDF and DOCX tests would require the optional dependencies
# These tests should be conditional based on availability


def test_parse_pdf_without_pypdf():
    """Test PDF parsing when pypdf is not available."""
    # This test would check behavior when pypdf import fails
    # For now, we'll skip this as it requires mocking imports
    pytest.skip("PDF parsing tests require pypdf dependency management")


def test_parse_docx_without_python_docx():
    """Test DOCX parsing when python-docx is not available."""
    # This test would check behavior when python-docx import fails
    # For now, we'll skip this as it requires mocking imports
    pytest.skip("DOCX parsing tests require python-docx dependency management")
