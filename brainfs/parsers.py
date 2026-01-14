"""Document parsers for various file formats."""

from pathlib import Path
from typing import Any, Optional

# Optional imports with type safety
pypdf_module: Any = None
try:
    import pypdf

    pypdf_module = pypdf
except ImportError:
    pass

docx_module: Any = None
try:
    import docx

    docx_module = docx
except ImportError:
    pass

markdown_module: Any = None
try:
    import markdown

    markdown_module = markdown
except ImportError:
    pass


class DocumentParser:
    """Base class for document parsers."""

    @staticmethod
    def parse(file_path: Path) -> str:
        """Parse document and return text content."""
        raise NotImplementedError


class TextParser(DocumentParser):
    """Parser for plain text files."""

    @staticmethod
    def parse(file_path: Path) -> str:
        """Parse text file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try latin-1 encoding as fallback
            with open(file_path, encoding="latin-1") as f:
                return f.read()


class PDFParser(DocumentParser):
    """Parser for PDF files."""

    @staticmethod
    def parse(file_path: Path) -> str:
        """Parse PDF file."""
        if pypdf_module is None:
            raise ImportError("pypdf is required for PDF parsing. Install with: pip install pypdf")

        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf_module.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text


class MarkdownParser(DocumentParser):
    """Parser for Markdown files."""

    @staticmethod
    def parse(file_path: Path) -> str:
        """Parse Markdown file and extract plain text."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if markdown_module is None:
            # Simple markdown to text conversion
            import re

            # Remove headers
            content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
            # Remove emphasis
            content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
            content = re.sub(r"\*(.*?)\*", r"\1", content)
            # Remove links
            content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)
            # Remove code blocks
            content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
            content = re.sub(r"`([^`]+)`", r"\1", content)
            return content
        else:
            # Use markdown library to convert to HTML then extract text
            import re

            html = markdown_module.markdown(content)
            # Simple HTML to text conversion
            text = re.sub(r"<[^>]+>", "", html)
            return text


class DOCXParser(DocumentParser):
    """Parser for DOCX files."""

    @staticmethod
    def parse(file_path: Path) -> str:
        """Parse DOCX file."""
        if docx_module is None:
            raise ImportError(
                "python-docx is required for DOCX parsing. Install with: pip install python-docx"
            )

        doc = docx_module.Document(str(file_path))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text


class DocumentParserFactory:
    """Factory for creating appropriate document parsers."""

    _parsers = {
        ".txt": TextParser,
        ".text": TextParser,
        ".md": MarkdownParser,
        ".markdown": MarkdownParser,
        ".pdf": PDFParser,
        ".docx": DOCXParser,
    }

    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[DocumentParser]:
        """Get appropriate parser for file extension."""
        suffix = file_path.suffix.lower()
        parser_class = cls._parsers.get(suffix)
        return parser_class() if parser_class else None

    @classmethod
    def supported_extensions(cls) -> list[str]:
        """Get list of supported file extensions."""
        return list(cls._parsers.keys())

    @classmethod
    def is_supported(cls, file_path: Path) -> bool:
        """Check if file extension is supported."""
        return file_path.suffix.lower() in cls._parsers


def parse_document(file_path: Path) -> str:
    """Parse a document and return its text content."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    parser = DocumentParserFactory.get_parser(file_path)
    if parser is None:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    return parser.parse(file_path)
