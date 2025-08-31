from __future__ import annotations

from typing import List, Any
import re
import ollama

from .config import DocragSettings
from .convert import DoclingConverter
from .index import ChromaIndexer


def _chunk_text(text: str, target_chars: int = 500, overlap: int = 100) -> List[str]:
    """
    Sentence-aware chunking with overlap. Falls back to fixed-size chunks if needed.
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[\.\?\!])\s+", text)
    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    for s in sentences:
        s_len = len(s)
        if size + s_len + 1 > target_chars and buf:
            chunk = " ".join(buf).strip()
            if chunk:
                chunks.append(chunk)
            if overlap > 0 and chunk:
                tail = chunk[-overlap:]
                buf = [tail, s]
                size = len(tail) + s_len
            else:
                buf = [s]
                size = s_len
        else:
            buf.append(s)
            size += s_len + 1

    if buf:
        chunk = " ".join(buf).strip()
        if chunk:
            chunks.append(chunk)

    if not chunks:
        return [text[i : i + target_chars] for i in range(0, len(text), target_chars)]
    return chunks


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

        chunks = _chunk_text(
            text,
            target_chars=self.cfg.chunk_chars,
            overlap=self.cfg.chunk_overlap,
        )
        return self.indexer.add(chunks)

    def ingest_many(self, sources: List[str]) -> int:
        """Ingest many sources; returns total chunks added."""
        total = 0
        for s in sources:
            total += self.ingest(s)
        return total

    # --- ASK ---

    def ask(self, question: str, *, stream: bool = False) -> str:
        """
        Retrieve top_k chunks and ask the LLM.

        If stream=True, tokens are printed to stdout as they arrive and the full
        answer is still returned at the end.
        """
        res: Any = self.indexer.query(question, top_k=max(1, self.cfg.top_k))

        # Support both dict-shaped and list-shaped return values from Chroma
        if isinstance(res, dict):
            docs = (res.get("documents") or [[]])[0]
        elif isinstance(res, list):
            docs = res[0] if res else []
        else:
            docs = []

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

        # Streaming mode
        full: List[str] = []
        for chunk in ollama.chat(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            part = chunk.get("message", {}).get("content", "")
            if part:
                full.append(part)
                print(part, end="", flush=True)
        print()  # newline after stream
        return "".join(full)
