import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
import json
from backend.embeddings import TextEmbedder, chunk_text
from backend.vector_store import VectorStore

# Load articles from pubmed_raw.json
with open("data/raw/pubmed_raw.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

all_chunks = []
all_metadata = []

for article in articles:
    text = article["title"] + " " + article["abstract"]
    chunks = chunk_text(text)
    for chunk in chunks:
        all_chunks.append(chunk)
        meta = article.copy()
        meta["chunk"] = chunk
        all_metadata.append(meta)

print(f"Total chunks: {len(all_chunks)}")
embedder = TextEmbedder()
print("Embedding chunks...")
embeddings = embedder.embed(all_chunks)
print("Storing embeddings in FAISS...")
store = VectorStore()
store.add_embeddings(embeddings, all_metadata)
print("Indexing complete!")