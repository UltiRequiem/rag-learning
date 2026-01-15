"""LLM integration for answer generation."""

import os

from dotenv import load_dotenv

load_dotenv()

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class LLMClient:
    """OpenAI client for generating answers from context."""

    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_answer(
        self, query: str, contexts: list[str], model: str = "gpt-3.5-turbo", max_tokens: int = 500
    ) -> str:
        """Generate an answer using the provided contexts."""

        combined_context = "\n\n---\n\n".join(contexts)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{combined_context}

Question: {query}

Instructions:
- Answer the question based only on the information provided in the context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided context."
- Be concise but thorough in your response
- If you quote from the context, use quotation marks

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.1,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error generating answer: {e}"

    def is_available(self) -> bool:
        """Check if LLM is available and configured."""
        return OPENAI_AVAILABLE and bool(self.api_key)
