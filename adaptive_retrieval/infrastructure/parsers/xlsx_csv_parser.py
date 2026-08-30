"""Parser pour données structurées (XLSX, CSV).

Conformément au diagramme UML ("Profile structured data"), on ne se
contente pas de sérialiser les cellules brutes : on génère un profil
textuel (colonnes, types, statistiques, échantillon) exploitable par la
recherche sémantique, en plus d'une représentation tabulaire lisible.
"""
from __future__ import annotations

import pandas as pd

from application.ports.document_parser_port import DocumentParserPort
from domain.entities import Document
from domain.enums import DocumentType


class XlsxCsvParser(DocumentParserPort):
    def supports(self, file_path: str) -> bool:
        return file_path.lower().endswith((".xlsx", ".xls", ".csv"))

    def parse(self, file_path: str) -> Document:
        is_csv = file_path.lower().endswith(".csv")
        sheets: dict[str, pd.DataFrame] = (
            {"data": pd.read_csv(file_path)} if is_csv else pd.read_excel(file_path, sheet_name=None)
        )

        sections: list[str] = []
        sheet_count = 0
        for sheet_name, df in sheets.items():
            sheet_count += 1
            sections.append(self._profile_sheet(sheet_name, df))

        doc_type = DocumentType.CSV if is_csv else DocumentType.XLSX
        return Document(
            source_path=file_path,
            doc_type=doc_type,
            raw_text="\n\n".join(sections),
            metadata={"sheet_count": sheet_count},
        )

    @staticmethod
    def _profile_sheet(sheet_name: str, df: pd.DataFrame) -> str:
        columns_info = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
        numeric_summary = df.describe(include="number").to_string() if not df.select_dtypes("number").empty else "(aucune colonne numérique)"
        sample_rows = df.head(10).to_string(index=False)

        return (
            f"[Feuille: {sheet_name}]\n"
            f"Colonnes: {columns_info}\n"
            f"Nombre de lignes: {len(df)}\n\n"
            f"Statistiques numériques:\n{numeric_summary}\n\n"
            f"Échantillon de données:\n{sample_rows}"
        )
