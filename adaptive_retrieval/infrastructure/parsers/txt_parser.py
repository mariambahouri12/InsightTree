from __future__ import annotations

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType


class TxtParser(DocumentParserPort):
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith((".txt", ".md"))

    def parse(self, file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        return Document(source_path=file_path, doc_type=DocumentType.TXT, raw_text=raw_text, metadata={})
