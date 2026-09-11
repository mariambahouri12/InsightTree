from __future__ import annotations
import pandas as pd

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType
from infrastructure.tools.structured_data_registry import StructuredDataRegistry


class XlsxParser(DocumentParserPort):
    def __init__(self, registry: StructuredDataRegistry) -> None:
        self._registry = registry

    def parse(self, file_path: str) -> Document:
        sheets = pd.read_excel(file_path, sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            self._registry.register(file_path, sheet_name, df)
            parts.append(f"--- Feuille: {sheet_name} ---")
            parts.append(df.to_string(index=False))
        return Document(
            source_path=file_path,
            doc_type=DocumentType.XLSX,
            raw_text="\n\n".join(parts),
            metadata={"sheet_names": list(sheets.keys())},
        )
