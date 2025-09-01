#!/usr/bin/env bash
set -euo pipefail

# Start Ollama
echo "[start] launching ollama serve on ${OLLAMA_HOST}"
(ollama serve &) 

# Wait for Ollama to come up
echo "[start] waiting for ollama..."
for i in {1..60}; do
  if curl -fsS "http://127.0.0.1:11434/api/tags" >/dev/null; then
    echo "[start] ollama is up"
    break
  fi
  sleep 1
done

# Ensure the required models exist (safe if already pulled)
echo "[start] ensuring models exist (llama3.2:1b, nomic-embed-text)"
ollama list | grep -q "llama3.2:1b" || ollama pull llama3.2:1b
ollama list | grep -q "nomic-embed-text" || ollama pull nomic-embed-text

# Launch the FastAPI app
echo "[start] starting docrag-llm API on :8000"
exec uvicorn server:app --host 0.0.0.0 --port 8000
