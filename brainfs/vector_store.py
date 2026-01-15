import heapq
from collections.abc import Sequence
from typing import TypedDict, Union

from .vector import dot_product, normalize

try:
    import numpy as np

    NUMPY_AVAILABLE = True
    ArrayLike = Union[Sequence[float], np.ndarray]
except ImportError:
    NUMPY_AVAILABLE = False
    ArrayLike = Sequence[float]


class VectorItem(TypedDict):
    text: str
    vector: Sequence[float]


class VectorStore:
    """High-performance vector store with NumPy optimization."""

    def __init__(self) -> None:
        self.database: list[VectorItem] = []
        self._vectors_matrix: Union[np.ndarray, None] = None
        self._texts: list[str] = []
        self._cache_dirty = True

    def add_item(self, text: str, vector: ArrayLike) -> None:
        """Add an item to the vector store."""
        normalized_vector = normalize(vector)
        item = VectorItem(text=text, vector=normalized_vector)
        self.database.append(item)
        self._cache_dirty = True

    def add_items_batch(self, texts: list[str], vectors: list[ArrayLike]) -> None:
        """Add multiple items efficiently."""
        if len(texts) != len(vectors):
            raise ValueError("Number of texts and vectors must match")

        if NUMPY_AVAILABLE and len(vectors) > 10:
            vector_matrix = np.asarray(vectors, dtype=np.float64)
            norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)

            norms = np.where(norms == 0, 1, norms)
            normalized_vectors = vector_matrix / norms

            for text, norm_vec in zip(texts, normalized_vectors):
                item = VectorItem(text=text, vector=norm_vec)
                self.database.append(item)
        else:
            for text, vector in zip(texts, vectors):
                self.add_item(text, vector)

        self._cache_dirty = True

    def _update_cache(self) -> None:
        """Update the NumPy cache for fast batch operations."""
        if not NUMPY_AVAILABLE or not self._cache_dirty:
            return

        if self.database:
            vectors = [item["vector"] for item in self.database]
            self._vectors_matrix = np.asarray(vectors, dtype=np.float64)
            self._texts = [item["text"] for item in self.database]
        else:
            self._vectors_matrix = None
            self._texts = []

        self._cache_dirty = False

    def search(self, query_vec: ArrayLike, top_k: int = 5) -> list[tuple[float, str]]:
        """Search for most similar vectors with NumPy acceleration."""
        if top_k <= 0 or not any(query_vec):
            return []

        if NUMPY_AVAILABLE and len(self.database) > 50:
            return self._search_vectorized(query_vec, top_k)

        return self._search_fallback(query_vec, top_k)

    def _search_vectorized(self, query_vec: ArrayLike, top_k: int) -> list[tuple[float, str]]:
        """Vectorized search using NumPy for maximum performance."""
        self._update_cache()

        if self._vectors_matrix is None:
            return []

        query_norm = np.asarray(normalize(query_vec), dtype=np.float64)

        similarities = np.dot(self._vectors_matrix, query_norm)

        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        results = []

        for idx in top_indices:
            score = float(similarities[idx])
            text = self._texts[idx]
            results.append((score, text))

        return results

    def _search_fallback(self, query_vec: ArrayLike, top_k: int) -> list[tuple[float, str]]:
        """Fallback search implementation."""
        query_norm = normalize(query_vec)
        heap: list[tuple[float, str]] = []

        for item in self.database:
            score = dot_product(item["vector"], query_norm)

            if len(heap) < top_k:
                heapq.heappush(heap, (score, item["text"]))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, item["text"]))

        return sorted(heap, reverse=True)

    def full_search(self, query_vec: ArrayLike) -> list[tuple[float, str]]:
        """Search all vectors and return sorted results."""
        if NUMPY_AVAILABLE and len(self.database) > 100:
            return self._search_vectorized(query_vec, len(self.database))
        else:
            results: list[tuple[float, str]] = []
            query_norm = normalize(query_vec)

            for item in self.database:
                score = dot_product(item["vector"], query_norm)
                results.append((score, item["text"]))

            results.sort(key=lambda x: x[0], reverse=True)
            return results

    def clear(self) -> None:
        """Clear all stored vectors."""
        self.database.clear()
        self._vectors_matrix = None
        self._texts = []
        self._cache_dirty = True

    def size(self) -> int:
        """Return the number of stored vectors."""
        return len(self.database)
