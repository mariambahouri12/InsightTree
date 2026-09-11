from __future__ import annotations

from pathlib import Path

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType


class TxtParser(DocumentParserPort):
    def parse(self, file_path: str) -> Document:
        path = Path(file_path)
        return Document(
            source_path=file_path,
            doc_type=DocumentType.TXT,
            raw_text=path.read_text(encoding="utf-8", errors="replace"),
        )
