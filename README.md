# 📄 DocRAG LLM  
**Docling → Chroma → Ollama: Simple Local RAG Pipeline**  

[![PyPI](https://img.shields.io/pypi/v/docrag-llm)](https://pypi.org/project/docrag-llm/)  
[![Python](https://img.shields.io/pypi/pyversions/docrag-llm)](https://pypi.org/project/docrag-llm/)  
[![License](https://img.shields.io/pypi/l/docrag-llm)](https://github.com/one1cat/docrag-llm/blob/main/LICENSE)  


---

### 🔎 What is DocRAG LLM?  
**DocRAG LLM** is a local-first **Retrieval-Augmented Generation (RAG)** pipeline.  
It connects **[Docling](https://github.com/DS4SD/docling)** for parsing → **[ChromaDB](https://www.trychroma.com/)** for vector storage → **[Ollama](https://ollama.com/)** for local LLM inference.  

No cloud lock-in. No API costs. Just **local docs → local vectors → local LLMs**.  

---

## ✨ Features
- 🔍 Parse documents with **Docling** (`PDF`, `DOCX`, `PPTX`, `HTML`, etc.)  
- 📑 **Intelligent chunking** for retrieval  
- 🧠 Store embeddings in **ChromaDB**  
- 🤖 Answer questions using **Ollama** (default: `llama3.2:1b`)  
- 🛡️ **Privacy-first** → all local execution  
- 🖥️ Use as a **CLI tool** or **Python library**  

---

## 📦 Installation
```bash
pip install docrag-llm
```

Requirements:
- Python 3.10+  
- [Ollama](https://ollama.com/) installed & running  
- Local models:  
  ```bash
  ollama pull llama3.2:1b
  ollama pull nomic-embed-text
  ```

---

## 🚀 Quickstart

### CLI – Ingest and Ask
```bash
# Ingest a document (default collection: demo)
python -m docrag.cli ingest https://arxiv.org/pdf/2508.20755

# Ask a question (default LLM: llama3.2:1b)
python -m docrag.cli ask "Summarize in 1 paragraph with 5 bullet points"
```

### Python API
```python
from docrag import DocragSettings, RAGPipeline

cfg = DocragSettings(
    persist_path="./.chroma",
    collection="demo",
    embed_model="nomic-embed-text",
    llm_model="llama3.2:1b",
)

pipeline = RAGPipeline(cfg)

# Ingest
n_chunks = pipeline.ingest("https://arxiv.org/pdf/2508.20755")
print(f"Ingested {n_chunks} chunks")

# Ask
answer = pipeline.ask("Give a concise bullet summary of the paper's contributions.")
print(answer)
```

---

## ⚙️ Configuration
Both CLI & Python API let you customize:
- `persist_path` → where ChromaDB stores vectors  
- `collection` → logical collection name  
- `embed_model` → embedding model (Ollama tag)  
- `llm_model` → LLM model (default: `llama3.2:1b`)  
- `chunk_chars` / `chunk_overlap` → chunking granularity  

---

## 📊 Roadmap
- [ ] `model-check` CLI → list installed Ollama models  
- [ ] Support multiple backends (Weaviate, Milvus)  
- [ ] Streaming output for long answers  
- [ ] Expanded test suite (large document regression cases)  
- [ ] Example notebooks & Hugging Face demo  

---

## 🤝 Contributing
PRs and issues welcome!  

```bash
pip install "docrag-llm[dev]"
ruff check .
pytest
```

---

## 📜 License
MIT License © 2025 [Armando Medina](https://github.com/one1cat)

---
