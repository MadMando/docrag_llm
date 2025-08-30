from __future__ import annotations
import json
from pathlib import Path
import typer
import ollama

from .config import DocragSettings
from .retrieve import RAGPipeline
from .convert import DoclingConverter
from .utils import expand_sources

app = typer.Typer(help="DocRAG: Docling → Chroma → Ollama")

@app.command()
def model_check() -> None:
    """
    List available Ollama models and mark the current defaults.
    """
    try:
        data = ollama.list()
    except Exception as e:
        typer.echo(f"[error] Could not contact Ollama: {e}")
        raise typer.Exit(code=2)

    models = data.get("models", []) or []
    if not models:
        typer.echo("No models installed. Try `ollama pull llama3.2:1b`.")
        raise typer.Exit(code=1)

    typer.echo("Installed Ollama models:")
    for m in models:
        name = m.get("model")
        size = m.get("size")
        modified = m.get("modified_at")
        typer.echo(f"  - {name}  ({size} bytes)  modified: {modified}")

@app.command()
def ingest(
    source: str = typer.Argument(..., help="Path/URL, directory, or glob"),
    persist: str = typer.Option("./.chroma", help="Chroma persist dir", show_default=True),
    collection: str = typer.Option("demo", help="Chroma collection name", show_default=True),
    embed_model: str = typer.Option(
        "nomic-embed-text", "--embed", help="Embedding model (Ollama tag)", show_default=True
    ),
    chunk_chars: int = typer.Option(1000, help="Approx characters per chunk", show_default=True),
    chunk_overlap: int = typer.Option(200, help="Chunk overlap", show_default=True),
    fail_on_empty: bool = typer.Option(True, help="Abort if extracted text is empty", show_default=True),
) -> None:
    cfg = DocragSettings(
        persist_path=persist,
        collection=collection,
        embed_model=embed_model,
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        fail_on_empty=fail_on_empty,
    )
    pipe = RAGPipeline(cfg)

    sources = expand_sources(source)
    if not sources:
        typer.echo("[error] No matching sources found.")
        raise typer.Exit(code=2)

    if len(sources) == 1:
        added = pipe.ingest(sources[0])
        typer.echo(f"Ingested {added} chunks from: {sources[0]}")
    else:
        total = pipe.ingest_many(sources)
        typer.echo(f"Ingested {total} chunks from {len(sources)} files into '{collection}' at '{persist}'.")

@app.command()
def ask(
    question: str = typer.Argument(..., help="Your query"),
    persist: str = typer.Option("./.chroma", help="Chroma persist dir", show_default=True),
    collection: str = typer.Option("demo", help="Chroma collection name", show_default=True),
    llm_model: str = typer.Option(
        "llama3.2:1b",
        "--llm",
        help="LLM model to use (Ollama tag)",
        show_default=True,
    ),
    top_k: int = typer.Option(5, help="Top-k chunks to retrieve", show_default=True),
    stream: bool = typer.Option(False, help="Stream tokens to stdout as they arrive", show_default=True),
) -> None:
    cfg = DocragSettings(
        persist_path=persist,
        collection=collection,
        llm_model=llm_model,
        top_k=top_k,
    )
    pipe = RAGPipeline(cfg)
    ans = pipe.ask(question, stream=stream)
    if not stream:
        typer.echo("\n=== ANSWER ===\n")
        typer.echo(ans)

@app.command()
def export(source: str, out_dir: str = "./exports"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    conv = DoclingConverter()
    md, dct = conv.to_both(source)
    (out / "output.md").write_text(md, encoding="utf-8")
    (out / "output.json").write_text(json.dumps(dct, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Saved: {out/'output.md'} and {out/'output.json'}")

def main() -> None:
    app()

if __name__ == "__main__":
    main()
