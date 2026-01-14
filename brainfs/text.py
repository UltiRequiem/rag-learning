import re


def chunk_text(text: str, chunk_size: int = 5, overlap: int = 2) -> list[str]:
    """Simple word-based chunking (fallback method)."""
    words = text.split()
    chunks: list[str] = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]

        if len(chunk_words) < 1:
            continue

        chunks.append(" ".join(chunk_words))

    return chunks


def chunk_by_sentences(
    text: str, sentences_per_chunk: int = 3, overlap: int = 1, min_chunk_length: int = 50
) -> list[str]:
    """
    Chunk text by sentences for better semantic coherence.

    Args:
        text: Input text to chunk
        sentences_per_chunk: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
        min_chunk_length: Minimum character length for a chunk

    Returns:
        List of text chunks
    """
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text.strip())

    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    step = sentences_per_chunk - overlap

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(chunk_sentences)

        if len(chunk_text) >= min_chunk_length:
            chunks.append(chunk_text)

    return chunks


def chunk_by_paragraphs(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Chunk text by paragraphs, respecting natural document structure.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum character size per chunk
        overlap: Character overlap between chunks

    Returns:
        List of text chunks
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if not paragraphs:
        return chunk_by_sentences(text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed max_chunk_size
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
            # Add current chunk and start new one
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Start new chunk with overlap from previous chunk
            if overlap > 0 and current_chunk:
                overlap_text = (
                    current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                )
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def smart_chunk(text: str, method: str = "sentences", **kwargs) -> list[str]:
    """
    Smart chunking that chooses the best method based on text characteristics.

    Args:
        text: Input text to chunk
        method: Chunking method ('sentences', 'paragraphs', 'words')
        **kwargs: Additional arguments for specific chunking methods

    Returns:
        List of text chunks
    """
    if not text.strip():
        return []

    if method == "sentences":
        sentence_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["sentences_per_chunk", "overlap", "min_chunk_length"]
        }

        return chunk_by_sentences(text, **sentence_kwargs)
    elif method == "paragraphs":
        paragraph_kwargs = {k: v for k, v in kwargs.items() if k in ["max_chunk_size", "overlap"]}
        return chunk_by_paragraphs(text, **paragraph_kwargs)
    elif method == "words":
        word_kwargs = {k: v for k, v in kwargs.items() if k in ["chunk_size", "overlap"]}
        return chunk_text(text, **word_kwargs)
    else:
        if "\n\n" in text and len(text.split("\n\n")) > 2:
            return chunk_by_paragraphs(text)
        elif len(text.split(".")) > 3:
            return chunk_by_sentences(text)
        else:
            return chunk_text(text)
