import faiss
import numpy as np
import os
import pickle

class VectorStore:
    def __init__(self, index_path="db/faiss_index.bin", mapping_path="db/id_mapping.pkl", dim=384):
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.id_mapping = {}

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            self._load()

    def add_embeddings(self, embeddings, metadata_list):
        embeddings_np = np.array(embeddings).astype("float32")
        self.index.add(embeddings_np)
        start_idx = len(self.id_mapping)
        for i, metadata in enumerate(metadata_list):
            self.id_mapping[start_idx + i] = metadata
        self._save()

    def search(self, query_embedding, k=5):
        query_embedding_np = np.array(query_embedding).astype("float32")
        if query_embedding_np.ndim == 1:
            query_embedding_np = query_embedding_np.reshape(1, -1)
        distances, indices = self.index.search(query_embedding_np, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.id_mapping:
                results.append({
                    "metadata": self.id_mapping[idx],
                    "score": float(distances[0][i])
                })
        return results

    def _save(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.mapping_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.id_mapping, f)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.mapping_path, "rb") as f:
            self.id_mapping = pickle.load(f)