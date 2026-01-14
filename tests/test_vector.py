"""Tests for vector operations."""

import pytest

from brainfs.vector import cosine_similarity, dot_product, magnitude, matmul, normalize


def test_dot_product():
    """Test dot product calculation."""
    vec_a = [1, 2, 3]
    vec_b = [4, 5, 6]
    result = dot_product(vec_a, vec_b)
    assert result == 32  # 1*4 + 2*5 + 3*6

    # Test empty vectors
    assert dot_product([], []) == 0

    # Test different lengths should raise error
    with pytest.raises(ValueError):
        dot_product([1, 2], [1, 2, 3])


def test_magnitude():
    """Test vector magnitude calculation."""
    vec = [3, 4]
    result = magnitude(vec)
    assert result == 5.0  # sqrt(3² + 4²)

    # Test zero vector
    assert magnitude([0, 0, 0]) == 0.0

    # Test single element
    assert magnitude([5]) == 5.0


def test_normalize():
    """Test vector normalization."""
    vec = [3, 4]
    result = normalize(vec)
    assert len(result) == 2
    assert abs(magnitude(result) - 1.0) < 1e-10

    # Test that normalized vector has unit length
    normalized = normalize([1, 1, 1])
    assert abs(magnitude(normalized) - 1.0) < 1e-10

    # Test zero vector should raise error
    with pytest.raises(ValueError):
        normalize([0, 0, 0])


def test_cosine_similarity():
    """Test cosine similarity calculation."""
    # Test identical vectors
    vec_a = [1, 1, 1]
    vec_b = [1, 1, 1]
    assert abs(cosine_similarity(vec_a, vec_b) - 1.0) < 1e-10

    # Test orthogonal vectors
    vec_a = [1, 0]
    vec_b = [0, 1]
    assert abs(cosine_similarity(vec_a, vec_b)) < 1e-10

    # Test opposite vectors
    vec_a = [1, 1]
    vec_b = [-1, -1]
    assert abs(cosine_similarity(vec_a, vec_b) - (-1.0)) < 1e-10

    # Test zero vectors should raise error
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])


def test_matmul():
    """Test matrix-vector multiplication."""
    matrix = [[1, 2], [3, 4], [5, 6]]
    vector = [2, 3]
    result = matmul(matrix, vector)

    expected = [8, 18, 28]  # [1*2+2*3, 3*2+4*3, 5*2+6*3]
    assert result == expected

    # Test empty matrix
    assert matmul([], []) == []

    # Test single row matrix
    matrix = [[1, 2, 3]]
    vector = [1, 1, 1]
    result = matmul(matrix, vector)
    assert result == [6]
