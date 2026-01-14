from text import chunk_text
from vector import cosine_similarity
from vector_store import VectorStore, VectorItem
from tokenizer import Tokenizer

raw_text = """
    Python is a high-level, general-purpose programming language. 
    Its design philosophy emphasizes code readability with the use of significant indentation.
    Python is dynamically typed and garbage-collected.
    It supports multiple programming paradigms, including structured, object-oriented and functional programming.
    """


default_query = "What paradigms does Python support?"
    

def run_rag_demo(query:str = default_query) -> None:
    print("--- 1. CHUNKING ---")

    chunks = chunk_text(raw_text, chunk_size=8, overlap=3)

    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c}")

    print("\n--- 2. INDEXING (Vectorization) ---")
    vectorizer = Tokenizer()
    vectorizer.fit(chunks) 
    store = VectorStore()
    

    for chunk in chunks:
        vec = vectorizer.embed(chunk)
        store.add_item(chunk, vec)

    print(f"\n--- 3. SEARCHING FOR: '{query}' ---")
    
    query_vec = vectorizer.embed(query)
    results = store.search(query_vec, top_k=3)
    

    top_match = results[0]
    print(f"\nTop Match Score: {top_match[0]:.4f}")
    print(f"All Matches:")

    for score, text in results:
        print(f"Score: {score:.4f} | Text: \"{text}\"")

    print(f"Retrieved Context: \"{top_match[1]}\"")
    
    print("\n--- 4. PROMPT FOR LLM ---")
    prompt = f"""
    CONTEXT: {top_match[1]}
    QUESTION: {query}
    ANSWER: (This is where you would send this string to ChatGPT/Llama)
    """
    print(prompt)

if __name__ == "__main__":
    print("Based on the following text:")
    print(raw_text)
    question = input("Enter your question about Python (or press Enter to use default): ")
    run_rag_demo()
