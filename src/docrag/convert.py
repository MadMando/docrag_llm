from __future__ import annotations
from typing import Dict, Tuple
from docling.document_converter import DocumentConverter

class DoclingConverter:
    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def to_markdown(self, source: str) -> str:
        result = self._converter.convert(source)
        return result.document.export_to_markdown()

    def to_dict(self, source: str) -> Dict:
        result = self._converter.convert(source)
        return result.document.export_to_dict()

    def to_both(self, source: str) -> Tuple[str, Dict]:
        result = self._converter.convert(source)
        return result.document.export_to_markdown(), result.document.export_to_dict()
