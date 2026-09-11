from __future__ import annotations

from pathlib import Path

from application.ports.document_parser_port import DocumentParserPort
from infrastructure.parsers.csv_parser import CsvParser
from infrastructure.parsers.docx_parser import DocxParser
from infrastructure.parsers.pdf_parser import PdfParser
from infrastructure.parsers.txt_parser import TxtParser
from infrastructure.parsers.xlsx_parser import XlsxParser
from infrastructure.tools.structured_data_registry import StructuredDataRegistry


class DocumentParserFactory:
    def __init__(self, registry: StructuredDataRegistry) -> None:
        self._registry = registry

    def get_parser(self, file_path: str) -> DocumentParserPort:
        extension = Path(file_path).suffix.lower()
        parsers = {
            ".pdf": PdfParser(),
            ".docx": DocxParser(),
            ".xlsx": XlsxParser(self._registry),
            ".csv": CsvParser(self._registry),
            ".txt": TxtParser(),
        }
        parser = parsers.get(extension)
        if parser is None:
            raise ValueError(f"Unsupported document type: {extension}")
        return parser
