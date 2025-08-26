import json
from backend.vector_store import VectorStore
from backend.embeddings import TextEmbedder

class RAGPipeline:
    def __init__(self):
        self.embedder = TextEmbedder()
        self.store = VectorStore()
        # Load all chunks and metadata for keyword search
        with open("data/raw/pubmed_raw.json", "r", encoding="utf-8") as f:
            self.articles = json.load(f)

    def query(self, text, topk=5):
        # Dynamic query expansion: generate all combinations of keywords (length >= 2)
        import itertools
        words = [w for w in text.lower().split() if w.isalnum() or '-' in w]
        expansions = [text]
        for r in range(2, len(words)+1):
            for combo in itertools.combinations(words, r):
                expansions.append(' '.join(combo))

        # Aggregate semantic results from all expansions
        semantic_results = []
        for q in expansions:
            q_emb = self.embedder.embed([q])
            results = self.store.search(q_emb, topk)
            semantic_results.extend(results)

        # Keyword search: match full query and any individual word (case-insensitive)
        keyword_results = []
        query_lower = text.lower()
        words = [w for w in text.lower().split() if w.isalnum() or '-' in w]
        for article in self.articles:
            found = False
            # Check full query
            if (
                query_lower in article.get("title", "").lower()
                or query_lower in article.get("abstract", "").lower()
                or query_lower in article.get("chunk", "").lower()
            ):
                found = True
            # Check individual words
            if not found:
                for w in words:
                    if (
                        w in article.get("title", "").lower()
                        or w in article.get("abstract", "").lower()
                        or w in article.get("chunk", "").lower()
                    ):
                        found = True
                        break
            if found:
                keyword_results.append({
                    "metadata": article,
                    "score": 0.0  # Highest relevance for exact match
                })

        # Combine results, prioritizing exact matches
        combined = keyword_results + [r for r in semantic_results if r not in keyword_results]
        # Remove duplicates (by pmid and chunk)
        seen = set()
        final_results = []
        for r in combined:
            key = (r["metadata"].get("pmid"), r["metadata"].get("chunk", ""))
            if key not in seen:
                final_results.append(r)
                seen.add(key)
            if len(final_results) >= topk:
                break
        return final_results