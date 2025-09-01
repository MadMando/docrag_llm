# src/docrag/index.py
from __future__ import annotations

from typing import List, Dict, Any
import uuid
import chromadb
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
import ollama


class _OllamaEmbedding(EmbeddingFunction):
    """Minimal embedding function using Ollama."""
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        embs: List[List[float]] = []
        for text in inputs:
            out = ollama.embeddings(model=self.model, prompt=text)
            embs.append(out["embedding"])
        return embs


class ChromaIndexer:
    def __init__(self, persist_path: str, collection: str, embed_model: str):
        self._client: PersistentClient = chromadb.PersistentClient(path=persist_path)
        self._col = self._client.get_or_create_collection(
            name=collection,
            embedding_function=_OllamaEmbedding(model=embed_model),
        )

    def add(self, chunks: List[str]) -> int:
        """Add chunks and return how many were added."""
        if not chunks:
            return 0
        ids = [f"chunk_{uuid.uuid4().hex}_{i}" for i in range(len(chunks))]
        # best-effort delete to avoid duplicate IDs if re-ingesting same doc
        try:
            self._col.delete(ids=ids)
        except Exception:
            pass
        self._col.add(documents=chunks, ids=ids)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Return a dict with a 'documents' key (shape expected by RAGPipeline.ask)."""
        res = self._col.query(query_texts=[query_text], n_results=max(1, top_k))
        # ensure shape: {'documents': [...]}
        return {"documents": (res.get("documents") or [[]])}
