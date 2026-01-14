from helpers import build_vector_store, create_chunks, default_query, print_chunks, print_indexing_step, print_llm_prompt, print_search_results, raw_text, search_query


def run_rag_pipeline(query: str = default_query) -> None:
    chunks = create_chunks()
    print_chunks(chunks)

    print_indexing_step()
    store, vectorizer = build_vector_store(chunks)

    results = search_query(store, vectorizer, query)
    print_search_results(query, results)

    top_context = results[0][1]
    print_llm_prompt(top_context, query)


if __name__ == "__main__":
    print("Based on the following text:")
    print(raw_text)
    question = input("Enter your question about Python (or press Enter to use default): ").strip() or default_query

    run_rag_pipeline(question)
