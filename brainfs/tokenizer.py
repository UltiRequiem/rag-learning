from dotenv import load_dotenv

load_dotenv()

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    NLTK_AVAILABLE = True

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    STEMMER = PorterStemmer()
    STOP_WORDS = set(stopwords.words("english"))

except ImportError:
    NLTK_AVAILABLE = False
    STEMMER = None
    STOP_WORDS = set()


class Tokenizer:
    vocab: dict[str, int]
    TRANSLATION_TABLE: dict[int, int | None] = str.maketrans("", "", "!.?,;:\"'()[]{}<>")
    SUFFIXES: list[tuple[str, str]] = [
        ("ing", ""),
        ("ly", ""),
        ("ed", ""),
        ("ies", "y"),
        ("ied", "y"),
        ("ies", "y"),
        ("s", ""),
        ("es", ""),
        ("er", ""),
        ("est", ""),
        ("tion", ""),
        ("ness", ""),
        ("ment", ""),
        ("able", ""),
        ("ible", ""),
    ]

    def __init__(self):
        self.vocab = {}

    def fit(self, documents: list[str]):
        unique_words: set[str] = set()

        for doc in documents:
            words = self._clean(doc)
            unique_words.update(words)

        sorted_words = sorted(unique_words)

        self.vocab = {word: idx for idx, word in enumerate(sorted_words)}

    def embed(self, text: str) -> list[float]:
        words = self._clean(text)

        try:
            import numpy as np

            vector = np.zeros(len(self.vocab), dtype=np.float64)

            word_indices = [self.vocab[word] for word in words if word in self.vocab]

            if word_indices:
                unique_indices, counts = np.unique(word_indices, return_counts=True)
                vector[unique_indices] = counts

            return vector.tolist()
        except ImportError:
            vector = self._create_empty_vector()

            for word in words:
                if word in self.vocab:
                    index = self.vocab[word]
                    vector[index] += 1

            return vector

    def _create_empty_vector(self) -> list[float]:
        return [0] * len(self.vocab)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently using NumPy when available."""
        try:
            import numpy as np

            batch_size = len(texts)
            vocab_size = len(self.vocab)
            embeddings = np.zeros((batch_size, vocab_size), dtype=np.float64)

            for i, text in enumerate(texts):
                words = self._clean(text)
                word_indices = [self.vocab[word] for word in words if word in self.vocab]

                if word_indices:
                    unique_indices, counts = np.unique(word_indices, return_counts=True)
                    embeddings[i, unique_indices] = counts

            return embeddings.tolist()
        except ImportError:
            return [self.embed(text) for text in texts]

    @staticmethod
    def _stem_word(word: str, extra: int = 2) -> str:
        if NLTK_AVAILABLE and STEMMER:
            return STEMMER.stem(word)

        for suffix, replacement in Tokenizer.SUFFIXES:
            if word.endswith(suffix) and len(word) > len(suffix) + extra:
                return word[: -len(suffix)] + replacement

        return word

    @staticmethod
    def _clean(word: str) -> list[str]:
        clean = word.lower().translate(Tokenizer.TRANSLATION_TABLE)
        words = clean.split()

        if NLTK_AVAILABLE and STOP_WORDS:
            words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        stemmed = [Tokenizer._stem_word(w) for w in words]

        not_empty = [w for w in stemmed if w and w.strip()]

        return not_empty
