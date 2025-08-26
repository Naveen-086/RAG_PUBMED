from backend.fetch_pubmed import fetch_pubmed_articles
from backend.embeddings import TextEmbedder, chunk_text
from backend.vector_store import VectorStore

TOPICS = ["cancer", "heart attack", "covid-19"]
MAX_RESULTS = 20

if __name__ == "__main__":
    all_chunks = []
    all_metadata = []
    for topic in TOPICS:
        print(f"Fetching articles for topic: {topic}")
        articles = fetch_pubmed_articles(topic, max_results=MAX_RESULTS)
        for article in articles:
            text = article["title"] + " " + article["abstract"]
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                meta = article.copy()
                meta["topic"] = topic
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
