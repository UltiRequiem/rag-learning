# AI Vector Search & RAG System

A Python implementation of a vector search engine with Retrieval-Augmented Generation (RAG) capabilities using modern Python tooling.

## Features

- **Vector Search**: Efficient similarity search using cosine similarity and dot products
- **Text Tokenization**: Bag-of-words with stemming for better word matching
- **RAG Pipeline**: Complete chunking → indexing → searching → prompt generation workflow
- **Optimized Storage**: Heap-based top-k search with normalized vectors

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw Text  │───▶│  Chunking   │───▶│ Tokenization│
└─────────────┘    └─────────────┘    └─────────────┘
                                               │
┌─────────────┐    ┌─────────────┐    ┌───────▼─────┐
│ LLM Prompt  │◀───│   Search    │◀───│Vector Store │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo>
cd ai-vector-search

# Install dependencies
uv sync --group dev
```

### Usage

```bash
# Run the demo
uv run python main.py

# Or use custom input
echo "Is Python good for machine learning?" | uv run python main.py
```

## Development

### Code Quality Tools

This project uses modern Python tooling:

- **[uv](https://github.com/astral-sh/uv)**: Fast Python package manager
- **[ruff](https://github.com/astral-sh/ruff)**: Lightning-fast linter and formatter
- **[ty](https://docs.astral.sh/ty/)**: Astral's fast type checker
- **[pre-commit](https://pre-commit.com/)**: Git hooks for code quality

### Commands

```bash
# Linting and formatting
uv run ruff check .          # Check for issues
uv run ruff check --fix .    # Fix auto-fixable issues
uv run ruff format .         # Format code

# Type checking
uv run ty check .

# Run all checks
uv run pre-commit run --all-files
```

### Project Structure

```
├── main.py           # RAG pipeline orchestration
├── helpers.py        # Core logic and utility functions
├── tokenizer.py      # Text tokenization with stemming
├── vector_store.py   # Vector storage and search
├── vector.py         # Vector math operations
├── text.py           # Text chunking utilities
├── pyproject.toml    # Project configuration
└── .github/
    └── workflows/
        └── ci.yml    # GitHub Actions CI
```

## Core Components

### Tokenizer (`tokenizer.py`)
- Bag-of-words vectorization
- Built-in stemming for better matching
- Configurable punctuation removal

### Vector Store (`vector_store.py`)
- Normalized vector storage
- Efficient top-k search using heaps
- Cosine similarity scoring

### Vector Operations (`vector.py`)
- Dot product, magnitude, normalization
- Cosine similarity calculation
- Matrix-vector multiplication

## CI/CD

GitHub Actions automatically runs:
- Code quality checks (ruff)
- Type checking (ty)
- Multi-Python version testing (3.9-3.12)
- Basic functionality tests

## Contributing

1. Install pre-commit hooks: `uv run pre-commit install`
2. Make changes
3. Run tests: `uv run python main.py`
4. Commit (hooks will run automatically)

## License

MIT License - see LICENSE file for details.