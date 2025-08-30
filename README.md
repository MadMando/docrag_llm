# 📄 DocRAG LLM

**DocRAG LLM** is a simple, Retrieval-Augmented Generation (RAG) pipeline.  
It connects **Docling** (document parsing) → **ChromaDB** (vector store) → **Ollama** (local LLMs) into a single workflow, with both a CLI and a Python API.

---

## ✨ Features
- 🔍 Parse documents with [Docling](https://github.com/docling-project/docling) (PDF, DOCX, PPTX, HTML, etc.).
- 📑 Chunk text intelligently for retrieval.
- 🧠 Store embeddings in [ChromaDB](https://www.trychroma.com/).
- 🤖 Answer questions using [Ollama](https://ollama.ai/) (default: `llama3.2:1b`).
- 🛡️ Designed for **local execution** (no cloud lock-in).
- 🖥️ Works both as a **CLI tool** and a **Python library**.

---

## 📦 Installation

```bash
pip install docrag-llm
```

### Requirements
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Models pulled locally:
  ```bash
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```

---

## 🚀 Quickstart (CLI)

### Ingest a document into Chroma

```bash
python -m docrag.cli ingest https://arxiv.org/pdf/2408.09869   --persist ./.chroma   --collection demo
```

- `--persist` → directory for Chroma DB (default: `./.chroma`)  
- `--collection` → logical collection name (default: `demo`)  
- `--embed` → embedding model (default: `nomic-embed-text`)  

---

### Ask a question (default LLM = `llama3.2:1b`)

```bash
python -m docrag.cli ask "Give a concise bullet summary of the paper's main contributions."   --persist ./.chroma   --collection demo
```

- `--llm` → LLM model to use (default: `llama3.2:1b`)  
- `--top-k` → number of chunks retrieved (default: 5)  

---

### Export parsed text

```bash
python -m docrag.cli export https://arxiv.org/pdf/2408.09869   --out-dir ./exports
```

Saves parsed text (Markdown/JSON).

---

### CLI Help

```bash
python -m docrag.cli --help
python -m docrag.cli ingest --help
python -m docrag.cli ask --help
python -m docrag.cli export --help
```

---

## 🐍 Usage as a Python Library

```python
from docrag import DocragSettings, RAGPipeline

# Configure pipeline
cfg = DocragSettings(
    persist_path="./.chroma",
    collection="demo",
    embed_model="nomic-embed-text",
    llm_model="llama3.2:1b",
)

pipeline = RAGPipeline(cfg)

# Ingest a document
n_chunks = pipeline.ingest("https://arxiv.org/pdf/2408.09869")
print(f"Ingested {n_chunks} chunks")

# Ask a question
answer = pipeline.ask("Give a concise bullet summary of the paper's main contributions.")
print(answer)
```

---

## ⚙️ Configuration

Both the **Python API** and **CLI** allow controlling:

- `persist_path` → path to Chroma DB  
- `collection` → collection name  
- `embed_model` → embedding model (Ollama tag)  
- `llm_model` → LLM model (default: `llama3.2:1b`)  
- `chunk_chars` / `chunk_overlap` → chunking granularity  

---

## 📊 Roadmap
- [ ] Add `model-check` CLI command to list installed Ollama models.  
- [ ] Support multiple backends (Weaviate, Milvus).  
- [ ] Add streaming output for long answers.  
- [ ] Expand test suite with large document regression cases.  

---

## 🤝 Contributing

PRs and issues welcome! Please run lint and tests before submitting:

```bash
ruff check .
pytest
```

---

## 📜 License

MIT License © 2025
