"""Tests for tokenizer functionality."""

from brainfs.tokenizer import Tokenizer


def test_tokenizer_basic():
    """Test basic tokenizer functionality."""
    tokenizer = Tokenizer()
    documents = ["hello world", "world peace", "hello peace"]

    # Test fitting
    tokenizer.fit(documents)
    assert len(tokenizer.vocab) > 0

    # Test that common words are in vocab
    expected_words = {"hello", "world", "peac"}  # peac due to stemming
    assert all(word in tokenizer.vocab for word in expected_words)


def test_tokenizer_embed():
    """Test text embedding."""
    tokenizer = Tokenizer()
    documents = ["hello world", "world peace"]
    tokenizer.fit(documents)

    # Test embedding
    vector = tokenizer.embed("hello world")
    assert len(vector) == len(tokenizer.vocab)
    assert sum(vector) > 0  # Should have some non-zero values

    # Test empty string
    empty_vector = tokenizer.embed("")
    assert all(v == 0 for v in empty_vector)

    # Test unknown words
    unknown_vector = tokenizer.embed("xyz123")
    assert all(v == 0 for v in unknown_vector)


def test_tokenizer_stemming():
    """Test that stemming is working."""
    tokenizer = Tokenizer()

    # Test stemming with different word forms
    words = tokenizer._clean("running runs runner")
    # All should be stemmed to similar forms
    assert len(set(words)) <= 2  # Should be fewer unique words due to stemming


def test_tokenizer_cleaning():
    """Test text cleaning functionality."""
    tokenizer = Tokenizer()

    # Test punctuation removal
    cleaned = tokenizer._clean("Hello, world! How are you?")
    assert "," not in " ".join(cleaned)
    assert "!" not in " ".join(cleaned)
    assert "?" not in " ".join(cleaned)

    # Test lowercase conversion
    cleaned = tokenizer._clean("HELLO World")
    assert all(word.islower() or not word.isalpha() for word in cleaned)


def test_tokenizer_empty_input():
    """Test tokenizer with empty or None input."""
    tokenizer = Tokenizer()

    # Test with empty documents
    tokenizer.fit([])
    assert len(tokenizer.vocab) == 0

    # Test with empty string documents
    tokenizer.fit(["", "  ", "\n"])
    assert len(tokenizer.vocab) == 0

    # Test embedding with no vocabulary
    vector = tokenizer.embed("hello")
    assert len(vector) == 0


def test_tokenizer_consistency():
    """Test that tokenizer produces consistent results."""
    tokenizer1 = Tokenizer()
    tokenizer2 = Tokenizer()

    documents = ["hello world", "world peace", "hello peace"]

    tokenizer1.fit(documents)
    tokenizer2.fit(documents)

    # Both should produce the same vocabulary
    assert tokenizer1.vocab == tokenizer2.vocab

    # Both should produce the same embeddings
    text = "hello world"
    vector1 = tokenizer1.embed(text)
    vector2 = tokenizer2.embed(text)
    assert vector1 == vector2
