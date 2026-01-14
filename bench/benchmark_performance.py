#!/usr/bin/env python3
"""Performance benchmarks for brainfs optimizations.

Run this script to see the performance improvements from NumPy integration.
"""

import random
import time

from brainfs.text import smart_chunk
from brainfs.tokenizer import Tokenizer
from brainfs.vector import dot_product
from brainfs.vector_store import VectorStore


class OldVectorOps:
    """Mock the old implementations for comparison."""

    @staticmethod
    def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
        """Old dot product implementation."""
        return sum(a * b for a, b in zip(vec_a, vec_b))

    @staticmethod
    def magnitude(vec: list[float]) -> float:
        """Old magnitude implementation."""
        return sum(x**2 for x in vec) ** 0.5

    @staticmethod
    def normalize(vec: list[float]) -> list[float]:
        """Old normalize implementation."""
        mag = OldVectorOps.magnitude(vec)
        if mag == 0:
            raise ValueError("Cannot normalize zero vector")
        return [x / mag for x in vec]


def generate_test_data(num_vectors: int, dim: int) -> list[list[float]]:
    """Generate random test vectors."""
    random.seed(42)  # For reproducible results
    vectors = []
    for _ in range(num_vectors):
        vec = [random.random() for _ in range(dim)]
        vectors.append(vec)
    return vectors


def benchmark_vector_operations():
    """Benchmark vector operations."""
    print("🔥 VECTOR OPERATIONS BENCHMARK")
    print("=" * 50)

    # Test data
    dimensions = [100, 500, 1000]
    num_ops = 10000

    for dim in dimensions:
        print(f"\n📊 Vector dimension: {dim}")

        # Generate test vectors
        vec1 = [random.random() for _ in range(dim)]
        vec2 = [random.random() for _ in range(dim)]

        # Benchmark old implementation
        start_time = time.time()
        for _ in range(num_ops):
            OldVectorOps.dot_product(vec1, vec2)
        old_time = time.time() - start_time

        # Benchmark new implementation
        start_time = time.time()
        for _ in range(num_ops):
            dot_product(vec1, vec2)
        new_time = time.time() - start_time

        speedup = old_time / new_time if new_time > 0 else float("inf")
        print(f"  Old implementation: {old_time:.4f}s")
        print(f"  New implementation: {new_time:.4f}s")
        print(f"  🚀 Speedup: {speedup:.2f}x")


def benchmark_tokenizer_embedding():
    """Benchmark tokenizer embedding operations."""
    print("\n🔤 TOKENIZER EMBEDDING BENCHMARK")
    print("=" * 50)

    # Create test documents
    documents = [
        f"This is test document number {i} with various words and content." for i in range(100)
    ]

    # Texts to embed
    test_texts = [f"Query text number {i} with different words and phrases." for i in range(500)]

    tokenizer = Tokenizer()
    tokenizer.fit(documents)

    print(f"📚 Vocabulary size: {len(tokenizer.vocab)}")
    print(f"📄 Number of texts to embed: {len(test_texts)}")

    # Benchmark individual embeddings
    start_time = time.time()
    _ = [tokenizer.embed(text) for text in test_texts]
    individual_time = time.time() - start_time

    # Benchmark batch embeddings (if available)
    try:
        start_time = time.time()
        _ = tokenizer.embed_batch(test_texts)
        batch_time = time.time() - start_time

        speedup = individual_time / batch_time if batch_time > 0 else float("inf")
        print(f"  Individual embedding: {individual_time:.4f}s")
        print(f"  Batch embedding: {batch_time:.4f}s")
        print(f"  🚀 Speedup: {speedup:.2f}x")
    except AttributeError:
        print("  Batch embedding not available")


def benchmark_vector_search():
    """Benchmark vector search operations."""
    print("\n🔍 VECTOR SEARCH BENCHMARK")
    print("=" * 50)

    # Test configurations
    test_configs = [
        (100, 50),  # 100 docs, 50-dim vectors
        (1000, 100),  # 1000 docs, 100-dim vectors
        (5000, 200),  # 5000 docs, 200-dim vectors
    ]

    for num_docs, dim in test_configs:
        print(f"\n📊 {num_docs} documents, {dim}-dimensional vectors")

        # Generate test data
        vectors = generate_test_data(num_docs, dim)
        texts = [f"Document {i}" for i in range(num_docs)]
        query_vector = [random.random() for _ in range(dim)]

        # Create and populate vector store
        store = VectorStore()

        # Benchmark individual additions
        start_time = time.time()
        for text, vector in zip(texts, vectors):
            store.add_item(text, vector)
        individual_add_time = time.time() - start_time

        # Clear and benchmark batch additions (if available)
        store.clear()
        try:
            start_time = time.time()
            # Type ignore for compatibility between list[list[float]] and list[ArrayLike]
            store.add_items_batch(texts, vectors)  # type: ignore[arg-type]
            batch_add_time = time.time() - start_time

            add_speedup = (
                individual_add_time / batch_add_time if batch_add_time > 0 else float("inf")
            )
            print(f"  Individual add: {individual_add_time:.4f}s")
            print(f"  Batch add: {batch_add_time:.4f}s")
            print(f"  🚀 Add speedup: {add_speedup:.2f}x")
        except AttributeError:
            print("  Batch add not available")

        # Benchmark search operations
        num_searches = 100

        start_time = time.time()
        for _ in range(num_searches):
            _ = store.search(query_vector, top_k=10)
        search_time = time.time() - start_time

        avg_search_time = search_time / num_searches * 1000  # ms per search
        print(f"  Average search time: {avg_search_time:.2f}ms")

        # Memory usage estimation
        try:
            import numpy as np  # noqa: F401

            if hasattr(store, "_vectors_matrix") and store._vectors_matrix is not None:
                memory_mb = store._vectors_matrix.nbytes / (1024 * 1024)
                print(f"  📊 Vector cache memory: {memory_mb:.2f}MB")
        except ImportError:
            pass


def benchmark_end_to_end():
    """Benchmark end-to-end document processing."""
    print("\n🏁 END-TO-END BENCHMARK")
    print("=" * 50)

    # Simulate document processing pipeline
    documents = [
        f"This is a comprehensive test document number {i}. " * 10
        + "It contains various topics including technology, science, and literature. " * 5
        + "The document has multiple paragraphs and sentences to test chunking. " * 3
        for i in range(50)
    ]

    print(f"📚 Processing {len(documents)} documents")

    # Full pipeline benchmark
    start_time = time.time()

    # Step 1: Chunk documents
    all_chunks = []
    for doc in documents:
        chunks = smart_chunk(doc, method="sentences", sentences_per_chunk=3)
        all_chunks.extend(chunks)

    # Step 2: Train tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit(all_chunks)

    # Step 3: Create embeddings
    embeddings = [tokenizer.embed(chunk) for chunk in all_chunks]

    # Step 4: Build vector store
    store = VectorStore()
    for chunk, embedding in zip(all_chunks, embeddings):
        store.add_item(chunk, embedding)

    # Step 5: Test queries
    query_texts = [
        "technology and science",
        "literature and writing",
        "comprehensive test document",
    ]

    query_results = []
    for query in query_texts:
        query_embedding = tokenizer.embed(query)
        results = store.search(query_embedding, top_k=5)
        query_results.append(results)

    total_time = time.time() - start_time

    print(f"  📊 Total chunks: {len(all_chunks)}")
    print(f"  📊 Vocabulary size: {len(tokenizer.vocab)}")
    print(f"  📊 Vector store size: {store.size()}")
    print(f"  ⏱️  Total processing time: {total_time:.2f}s")
    print(f"  ⚡ Throughput: {len(documents) / total_time:.1f} docs/second")


if __name__ == "__main__":
    print("🚀 BRAINFS PERFORMANCE BENCHMARK")
    print("=" * 60)
    print("This benchmark compares optimized vs unoptimized implementations.")
    print()

    try:
        import numpy as np

        print(f"✅ NumPy {np.__version__} available - optimizations enabled!")
    except ImportError:
        print("⚠️  NumPy not available - using fallback implementations")

    print()

    # Run all benchmarks
    benchmark_vector_operations()
    benchmark_tokenizer_embedding()
    benchmark_vector_search()
    benchmark_end_to_end()

    print("\n" + "=" * 60)
    print("🎯 BENCHMARK COMPLETE")
    print("The results show the performance improvements from NumPy integration!")
