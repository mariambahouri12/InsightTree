from __future__ import annotations

from pypdf import PdfReader

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType


class PdfParser(DocumentParserPort):
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def parse(self, file_path: str) -> Document:
        reader = PdfReader(file_path)
        pages_text = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"[page {page_number}]\n{text}")

        return Document(
            source_path=file_path,
            doc_type=DocumentType.PDF,
            raw_text="\n\n".join(pages_text),
            metadata={"page_count": len(reader.pages)},
        )
