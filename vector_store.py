import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.texts = []

    def add(self, embedding, text):
        self.vectors.append(embedding)
        self.texts.append(text)
        self.index.add(np.array([embedding], dtype=np.float32))

    def search(self, query_emb, top_k=3):
        D, I = self.index.search(np.array([query_emb], dtype=np.float32), top_k)
        return [(self.texts[i], D[0][j]) for j, i in enumerate(I[0])]
