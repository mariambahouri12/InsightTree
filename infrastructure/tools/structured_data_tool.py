"""
Outil déterministe : interroge les tables structurées (XLSX/CSV)
ingérées, via un filtre/pandas query généré par le LLM à partir du
besoin d'information et du profil des colonnes disponibles.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from application.ports.llm_port import LLMPort
from application.ports.tool_port import ToolPort
from domain.entities import Evidence, ToolResult
from infrastructure.tools.structured_data_registry import (
    StructuredDataRegistry,
)


_QUERY_SYSTEM = (
    "Tu génères une requête pandas `DataFrame.query()` "
    "(syntaxe pandas, colonnes entre backticks si besoin) "
    "pour répondre au besoin d'information, à partir du nom "
    "de la table et de ses colonnes disponibles. "
    "Réponds UNIQUEMENT en JSON strict: "
    '{"table_key": "<clé de table>", '
    '"query": "<expression pandas.query>", '
    '"columns": ["<colonnes pertinentes à afficher>"]}'
)


class StructuredDataQueryTool(ToolPort):

    name = "structured_data_query"

    description = (
        "Interroge les tableaux XLSX/CSV ingérés "
        "(filtrage, sélection de colonnes)."
    )

    requires_data = False

    def __init__(
        self,
        llm: LLMPort,
        registry: StructuredDataRegistry,
    ) -> None:
        self._llm = llm
        self._registry = registry

    def execute(
        self,
        information_need: str,
        input_data: Optional[list[Evidence]] = None,
        tool_input: Optional[dict[str, Any]] = None,
    ) -> ToolResult:

        del input_data
        del tool_input

        tables = self._registry.list_tables()

        if not tables:
            return ToolResult(
                success=False,
                output_text="",
                error="No structured data ingested.",
            )

        tables_description = "\n".join(
            f"- {key}: colonnes = {list(df.columns)}"
            for key, df in tables.items()
        )

        prompt = (
            f"Besoin d'information:\n"
            f"{information_need}\n\n"
            f"Tables disponibles:\n"
            f"{tables_description}\n\n"
            f"Requête:"
        )

        result = self._llm.generate_json(
            prompt,
            system=_QUERY_SYSTEM,
        )

        table_key = str(
            result.get("table_key", "")
        ).strip()

        query = str(
            result.get("query", "")
        ).strip()

        columns = result.get("columns") or None

        df = tables.get(table_key)

        if df is None or not query:
            return ToolResult(
                success=False,
                output_text="",
                error="Invalid table_key or empty query.",
            )

        try:
            filtered = df.query(query)

            if columns:
                valid_columns = [
                    column
                    for column in columns
                    if column in filtered.columns
                ]

                if valid_columns:
                    filtered = filtered[valid_columns]

        except Exception as exc:
            return ToolResult(
                success=False,
                output_text="",
                error=f"Query error: {exc}",
            )

        if filtered.empty:
            return ToolResult(
                success=True,
                output_text=(
                    "Aucune ligne ne correspond à ce filtre."
                ),
                data={
                    "row_count": 0,
                },
            )

        output_text = filtered.head(20).to_string(
            index=False
        )

        return ToolResult(
            success=True,
            output_text=(
                f"Résultat de la requête sur {table_key} "
                f"({query}):\n{output_text}"
            ),
            data={
                "table_key": table_key,
                "query": query,
                "row_count": len(filtered),
            },
        )