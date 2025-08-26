import json
import os
import pickle
from backend.vector_store import VectorStore
from backend.embeddings import TextEmbedder

class RAGPipeline:
    def __init__(self):
        self.embedder = TextEmbedder()
        self.store = VectorStore()
        # Load chunked metadata for keyword search
        # Use the same metadata as stored in the FAISS index
        if os.path.exists(self.store.mapping_path):
            with open(self.store.mapping_path, "rb") as f:
                self.chunked_metadata = list(pickle.load(f).values())
        else:
            self.chunked_metadata = []

    def query(self, text, topk=5):
        import itertools
        import os
        import pickle
        # Dynamic query expansion: generate all combinations of keywords (length >= 2)
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
        for meta in self.chunked_metadata:
            found = False
            # Check full query
            if (
                query_lower in meta.get("title", "").lower()
                or query_lower in meta.get("abstract", "").lower()
                or query_lower in meta.get("chunk", "").lower()
            ):
                found = True
            # Check individual words
            if not found:
                for w in words:
                    if (
                        w in meta.get("title", "").lower()
                        or w in meta.get("abstract", "").lower()
                        or w in meta.get("chunk", "").lower()
                    ):
                        found = True
                        break
            if found:
                keyword_results.append({
                    "metadata": meta,
                    "score": 0.0  # Highest relevance for exact match
                })

        # Combine results, prioritizing semantic matches
        combined = semantic_results + [r for r in keyword_results if (r["metadata"].get("pmid"), r["metadata"].get("chunk", "")) not in {(s["metadata"].get("pmid"), s["metadata"].get("chunk", "")) for s in semantic_results}]
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