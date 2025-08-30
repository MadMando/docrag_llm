from __future__ import annotations
import os
from typing import List, Sequence
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
import ollama

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str = "nomic-embed-text") -> None:
        self.model = model
    def __call__(self, inputs: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for doc in inputs:
            resp = ollama.embeddings(model=self.model, prompt=doc)
            out.append(resp["embedding"])
        return out

class ChromaIndexer:
    def __init__(self, persist_path: str, collection: str, embed_model: str) -> None:
        os.makedirs(persist_path, exist_ok=True)
        self._client = PersistentClient(path=persist_path)
        self._ef = OllamaEmbeddingFunction(embed_model)
        self._col = self._client.get_or_create_collection(name=collection, embedding_function=self._ef)
    def reset_ids(self, count: int) -> None:
        ids = [f"chunk_{i}" for i in range(count)]
        try: self._col.delete(ids=ids)
        except Exception: pass
    def add(self, chunks: Sequence[str]) -> None:
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self._col.add(documents=list(chunks), ids=ids)
    def query(self, query_text: str, top_k: int = 5) -> List[str]:
        res = self._col.query(query_texts=[query_text], n_results=max(1, top_k))
        return (res.get("documents") or [[]])[0]
