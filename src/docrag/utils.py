from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

DOC_EXTS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".txt", ".xlsx"}

def expand_sources(source: str) -> List[str]:
    """
    Accepts a single path/URL, directory, or glob.
    - If URL (http/https), returns [source]
    - If file, returns [source]
    - If directory, returns all acceptable files under it (recursive)
    - If glob, expands
    """
    src = source.strip()
    if src.startswith("http://") or src.startswith("https://"):
        return [src]

    p = Path(src)

    # Glob pattern?
    if any(ch in src for ch in "*?[]"):
        return [str(pp) for pp in Path().glob(src) if pp.suffix.lower() in DOC_EXTS]

    # Single path
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(pp) for pp in p.rglob("*") if pp.suffix.lower() in DOC_EXTS]

    # Fallback: return as-is (Docling can try to open; user gets a sensible error)
    return [src]
