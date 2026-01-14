import heapq
from collections.abc import Sequence
from typing import TypedDict

from vector import dot_product, normalize


class VectorItem(TypedDict):
    text: str
    vector: Sequence[float]


class VectorStore:
    database: list[VectorItem]

    def __init__(self):
        self.database = []

    def add_item(self, text: str, vector: Sequence[float]) -> None:
        item = VectorItem(text=text, vector=normalize(vector))

        self.database.append(item)

    def full_search(self, query_vec: Sequence[float]):
        results: list[tuple[float, str]] = []

        for item in self.database:
            score = dot_product(item["vector"], query_vec)
            results.append((score, item["text"]))

        results.sort(key=lambda x: x[0], reverse=True)

        return results

    def search(self, query_vec: Sequence[float], top_k: int = 5) -> list[tuple[float, str]]:
        if top_k <= 0:
            return []

        query_norm = normalize(query_vec)

        heap: list[tuple[float, str]] = []

        for item in self.database:
            score = dot_product(item["vector"], query_norm)

            if len(heap) < top_k:
                heapq.heappush(heap, (score, item["text"]))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score, item["text"]))

        return sorted(heap, reverse=True)
