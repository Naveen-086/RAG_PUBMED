
# Sentence-based chunking and sentence-transformers embedding
from sentence_transformers import SentenceTransformer
import nltk

# Download punkt if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

def chunk_text(text, min_sentences=2, max_sentences=5):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i+max_sentences])
        chunks.append(chunk)
        i += max_sentences - 1  # overlap by one sentence
    return chunks