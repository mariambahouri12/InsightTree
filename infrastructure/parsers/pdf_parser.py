from __future__ import annotations
from pypdf import PdfReader

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType


class PdfParser(DocumentParserPort):
    def parse(self, file_path: str) -> Document:
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return Document(
            source_path=file_path,
            doc_type=DocumentType.PDF,
            raw_text="\n\n".join(pages),
            metadata={"page_count": len(reader.pages)},
        )
