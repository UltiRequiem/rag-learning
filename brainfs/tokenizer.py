from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    NLTK_AVAILABLE = True

    # Download required NLTK data if not already present
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
        vector = self._create_empty_vector()

        for word in words:
            if word in self.vocab:
                index = self.vocab[word]
                vector[index] += 1

        return vector

    def _create_empty_vector(self) -> list[float]:
        return [0] * len(self.vocab)

    @staticmethod
    def _stem_word(word: str, extra: int = 2) -> str:
        if NLTK_AVAILABLE and STEMMER:
            return STEMMER.stem(word)
        else:
            # Fallback to simple stemming
            for suffix, replacement in Tokenizer.SUFFIXES:
                if word.endswith(suffix) and len(word) > len(suffix) + extra:
                    return word[: -len(suffix)] + replacement
            return word

    @staticmethod
    def _clean(word: str) -> list[str]:
        clean = word.lower().translate(Tokenizer.TRANSLATION_TABLE)
        words = clean.split()

        # Remove stopwords if NLTK is available
        if NLTK_AVAILABLE and STOP_WORDS:
            words = [w for w in words if w not in STOP_WORDS and len(w) > 2]

        # Stem words
        stemmed = [Tokenizer._stem_word(w) for w in words]

        # Remove empty strings
        return [w for w in stemmed if w]
