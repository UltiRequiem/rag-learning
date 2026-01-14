def chunk_text(text: str, chunk_size: int = 5, overlap: int = 2) -> list[str]:
    words = text.split()
    chunks: list[str] = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]

        if len(chunk_words) < 1:
            continue

        chunks.append(" ".join(chunk_words))

    return chunks
