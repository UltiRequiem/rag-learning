import math

from collections.abc import Sequence

def dot_product(vec_a: Sequence[float],vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length")

    return sum(a * b for a, b in zip(vec_a, vec_b))

def magnitude(vec: Sequence[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in vec))

def normalize(vec: Sequence[float]) -> Sequence[float]:
    mag = magnitude(vec)

    if mag == 0:
        raise ValueError("Cannot normalize a zero-length vector")

    return [x / mag for x in vec]

def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be of the same length")

    mag_a, mag_b = magnitude(vec_a), magnitude(vec_b)

    if mag_a == 0 or mag_b == 0:
        raise ValueError("Cannot compute cosine similarity for zero-length vectors")

    return dot_product(vec_a, vec_b) / (mag_a * mag_b)

def matmul(matrix:Sequence[Sequence[float]], vector:Sequence[float]) -> list[float]:
    results: list[float] = []

    for row in matrix:
        result = dot_product(row, vector)
        results.append(result)

    return results
