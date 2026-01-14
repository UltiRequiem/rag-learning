import math
from collections.abc import Sequence
from typing import Union

try:
    import numpy as np

    NUMPY_AVAILABLE = True
    ArrayLike = Union[Sequence[float], np.ndarray]
except ImportError:
    NUMPY_AVAILABLE = False
    ArrayLike = Sequence[float]


def dot_product(vec_a: ArrayLike, vec_b: ArrayLike) -> float:
    """Compute dot product of two vectors with NumPy acceleration when available."""
    if NUMPY_AVAILABLE:
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)

        if a.shape != b.shape:
            raise ValueError("Vectors must be of the same length")

        return float(np.dot(a, b))
    else:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of the same length")

        return sum(a * b for a, b in zip(vec_a, vec_b))


def magnitude(vec: ArrayLike) -> float:
    """Compute magnitude of a vector with NumPy acceleration when available."""
    if NUMPY_AVAILABLE:
        arr = np.asarray(vec, dtype=np.float64)
        return float(np.linalg.norm(arr))
    else:
        return math.sqrt(sum(x**2 for x in vec))


def normalize(vec: ArrayLike) -> list[float]:
    """Normalize a vector with NumPy acceleration when available."""
    if NUMPY_AVAILABLE:
        arr = np.asarray(vec, dtype=np.float64)
        mag = np.linalg.norm(arr)

        if mag == 0:
            raise ValueError("Cannot normalize a zero-length vector")

        normalized = arr / mag
        return normalized.tolist()
    else:
        mag = magnitude(vec)

        if mag == 0:
            raise ValueError("Cannot normalize a zero-length vector")

        return [x / mag for x in vec]


def cosine_similarity(vec_a: ArrayLike, vec_b: ArrayLike) -> float:
    """Compute cosine similarity with NumPy acceleration when available."""
    if NUMPY_AVAILABLE:
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)

        if a.shape != b.shape:
            raise ValueError("Vectors must be of the same length")

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Cannot compute cosine similarity for zero-length vectors")

        return float(np.dot(a, b) / (norm_a * norm_b))
    else:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must be of the same length")

        mag_a, mag_b = magnitude(vec_a), magnitude(vec_b)

        if mag_a == 0 or mag_b == 0:
            raise ValueError("Cannot compute cosine similarity for zero-length vectors")

        return dot_product(vec_a, vec_b) / (mag_a * mag_b)


def matmul(matrix: Sequence[ArrayLike], vector: ArrayLike) -> list[float]:
    """Matrix-vector multiplication with NumPy acceleration when available."""
    if not matrix or not vector:
        return []

    if NUMPY_AVAILABLE:
        mat = np.asarray(matrix, dtype=np.float64)
        vec = np.asarray(vector, dtype=np.float64)

        result = np.dot(mat, vec)
        return result.tolist()
    else:
        results: list[float] = []
        for row in matrix:
            result = dot_product(row, vector)
            results.append(result)
        return results


def batch_cosine_similarity(
    queries: Sequence[ArrayLike], documents: Sequence[ArrayLike]
) -> ArrayLike:
    """Compute cosine similarities for multiple query-document pairs efficiently."""
    if NUMPY_AVAILABLE:
        q_matrix = np.asarray(queries, dtype=np.float64)
        d_matrix = np.asarray(documents, dtype=np.float64)

        q_norms = np.linalg.norm(q_matrix, axis=1, keepdims=True)
        d_norms = np.linalg.norm(d_matrix, axis=1, keepdims=True)

        q_norms = np.where(q_norms == 0, 1, q_norms)
        d_norms = np.where(d_norms == 0, 1, d_norms)

        q_normalized = q_matrix / q_norms
        d_normalized = d_matrix / d_norms

        return np.sum(q_normalized * d_normalized, axis=1)
    else:
        results = []

        for q, d in zip(queries, documents):
            try:
                sim = cosine_similarity(q, d)
                results.append(sim)
            except ValueError:
                results.append(0.0)

        return results
