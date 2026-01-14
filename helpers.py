from text import chunk_text
from vector_store import VectorStore
from tokenizer import Tokenizer

raw_text = """
    Python is a high-level, general-purpose programming language.
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically typed and garbage-collected.
    It supports multiple programming paradigms, including structured, object-oriented and functional programming.
    """

default_query = "What paradigms does Python support?"


def print_chunks(chunks: list[str]) -> None:
    print("--- 1. CHUNKING ---")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk}")


def print_indexing_step() -> None:
    print("\n--- 2. INDEXING (Vectorization) ---")


def print_search_results(query: str, results: list[tuple[float, str]]) -> None:
    print(f"\n--- 3. SEARCHING FOR: '{query}' ---")

    top_match = results[0]
    print(f"\nTop Match Score: {top_match[0]:.4f}")
    print("All Matches:")

    for score, text in results:
        print(f"Score: {score:.4f} | Text: \"{text}\"")

    print(f"Retrieved Context: \"{top_match[1]}\"")


def print_llm_prompt(context: str, query: str) -> None:
    print("\n--- 4. PROMPT FOR LLM ---")
    prompt = f"""
    CONTEXT: {context}
    QUESTION: {query}
    ANSWER: (This is where you would send this string to ChatGPT/Llama)
    """
    print(prompt)


def create_chunks() -> list[str]:
    return chunk_text(raw_text, chunk_size=8, overlap=3)

def build_vector_store(chunks: list[str]) -> tuple[VectorStore, Tokenizer]:
    vectorizer = Tokenizer()
    vectorizer.fit(chunks)
    store = VectorStore()

    for chunk in chunks:
        vec = vectorizer.embed(chunk)
        store.add_item(chunk, vec)

    return store, vectorizer


def search_query(store: VectorStore, vectorizer: Tokenizer, query: str) -> list[tuple[float, str]]:
    query_vec = vectorizer.embed(query)
    return store.search(query_vec, top_k=3)
