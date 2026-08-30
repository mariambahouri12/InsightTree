from __future__ import annotations

from application.ports.document_parser_port import DocumentParserPort
from infrastructure.parsers.docx_parser import DocxParser
from infrastructure.parsers.pdf_parser import PdfParser
from infrastructure.parsers.txt_parser import TxtParser
from infrastructure.parsers.xlsx_csv_parser import XlsxCsvParser


class DocumentParserFactory:
    def __init__(self) -> None:
        self._parsers: list[DocumentParserPort] = [
            PdfParser(),
            DocxParser(),
            XlsxCsvParser(),
            TxtParser(),
        ]

    def get_parser(self, file_path: str) -> DocumentParserPort:
        for parser in self._parsers:
            if parser.supports(file_path):
                return parser
        raise ValueError(f"Aucun parser disponible pour le fichier: {file_path}")
