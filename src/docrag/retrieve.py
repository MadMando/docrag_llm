from __future__ import annotations
from typing import List, Optional
import ollama

from .config import DocragSettings
from .convert import DoclingConverter
from .index import ChromaIndexer

class RAGPipeline:
    def __init__(self, cfg: DocragSettings):
        self.cfg = cfg
        self.converter = DoclingConverter()
        self.indexer = ChromaIndexer(
            persist_path=cfg.persist_path,
            collection=cfg.collection,
            embed_model=cfg.embed_model,
        )

    # --- INGEST ---

    def ingest(self, source: str) -> int:
        """Ingest a single source (URL or local path). Returns #chunks added."""
        text = self.converter.to_markdown(source)
        if self.cfg.fail_on_empty and (not text or not text.strip()):
            raise ValueError("Empty text extracted from source.")
        chunks = self.indexer.chunk_text(
            text,
            target_chars=self.cfg.chunk_chars,
            overlap=self.cfg.chunk_overlap,
        )
        return self.indexer.add(chunks)

    def ingest_many(self, sources: List[str]) -> int:
        """Ingest many sources, returns total chunks added."""
        total = 0
        for s in sources:
            total += self.ingest(s)
        return total

    # --- ASK ---

    def ask(self, question: str, *, stream: bool = False) -> str:
        """
        Retrieve top_k chunks and ask the LLM.
        If stream=True, yields tokens to stdout (CLI handles printing) while building the final answer.
        Always returns the full final answer.
        """
        res = self.indexer.query(question, top_k=max(1, self.cfg.top_k))
        docs = (res.get("documents") or [[]])[0]
        context = "\n\n".join(docs) if docs else ""

        prompt = (
            "Use ONLY the provided context to answer. If insufficient, say so briefly.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
        )

        if not stream:
            resp = ollama.chat(
                model=self.cfg.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["message"]["content"]

        # streaming mode
        full = []
        for chunk in ollama.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            part = chunk.get("message", {}).get("content", "")
            if part:
                full.append(part)
                # The CLI will just print this raw; in library use, callers can capture stdout or adapt.
                print(part, end="", flush=True)
        print()  # newline after stream
        return "".join(full)
