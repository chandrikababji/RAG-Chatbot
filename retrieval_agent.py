from embeddings import get_embedding
from vector_store import VectorStore

class RetrievalAgent:
    def __init__(self):
        self.store = VectorStore()

    def add_document(self, chunks):
        for ch in chunks:
            emb = get_embedding(ch)
            self.store.add(emb, ch)

    def retrieve(self, query):
        q_emb = get_embedding(query)
        return self.store.search(q_emb)
