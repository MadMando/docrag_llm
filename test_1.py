from docrag import DocragSettings, RAGPipeline

cfg = DocragSettings(persist_path="./.chroma", collection="demo", llm_model="llama3.2:1b") # <-update llm here
pipe = RAGPipeline(cfg)

# Ingest a document
pipe.ingest("https://arxiv.org/pdf/2508.20755")

# Ask a question
print(pipe.ask("Summarize"))