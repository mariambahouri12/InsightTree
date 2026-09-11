from __future__ import annotations
import pandas as pd

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType
from infrastructure.tools.structured_data_registry import StructuredDataRegistry


class CsvParser(DocumentParserPort):
    def __init__(self, registry: StructuredDataRegistry) -> None:
        self._registry = registry

    def parse(self, file_path: str) -> Document:
        df = pd.read_csv(file_path)
        self._registry.register(file_path, "csv", df)
        return Document(
            source_path=file_path,
            doc_type=DocumentType.CSV,
            raw_text=df.to_string(index=False),
            metadata={"columns": list(df.columns), "row_count": len(df)},
        )
