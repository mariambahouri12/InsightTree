from __future__ import annotations

from typing import Optional

import pandas as pd

from application.ports.llm_port import LLMPort
from application.ports.tool_port import ToolPort
from domain.entities import Evidence, ToolResult
from infrastructure.tools.structured_data_registry import StructuredDataRegistry

_SYSTEM = """
Génère une requête pandas DataFrame.query() pour une table donnée.
Réponds uniquement en JSON:
{"table_key":"...", "query":"...", "columns":["..."]}.
""".strip()


class StructuredDataQueryTool(ToolPort):
    name = "structured_data_query"
    description = "Interroge les tables XLSX/CSV ingérées."
    requires_data = False

    def __init__(self, llm: LLMPort, registry: StructuredDataRegistry) -> None:
        self._llm = llm
        self._registry = registry

    def execute(
        self,
        information_need: str,
        input_data: Optional[list[Evidence]] = None,
    ) -> ToolResult:
        tables = self._registry.list_tables()
        if not tables:
            return ToolResult(False, "", error="No structured data ingested.")

        description = "\n".join(
            f"- {key}: columns={list(df.columns)}"
            for key, df in tables.items()
        )
        result = self._llm.generate_json(
            f"Besoin:\n{information_need}\n\nTables:\n{description}",
            system=_SYSTEM,
        )
        table_key = str(result.get("table_key", "")).strip()
        query = str(result.get("query", "")).strip()
        columns = result.get("columns") or []

        df = tables.get(table_key)
        if df is None or not query:
            return ToolResult(False, "", error="Invalid table or empty query.")

        if not isinstance(columns, list):
            columns = []

        try:
            filtered = df.query(query)
            valid_columns = [c for c in columns if c in filtered.columns]
            if valid_columns:
                filtered = filtered[valid_columns]
        except Exception as exc:
            return ToolResult(False, "", error=f"Query error: {exc}")

        if filtered.empty:
            return ToolResult(
                True,
                "Aucune ligne ne correspond à ce filtre.",
                {"table_key": table_key, "query": query, "row_count": 0},
            )

        output = filtered.head(20).to_string(index=False)
        return ToolResult(
            True,
            f"Résultat ({table_key}):\n{output}",
            {"table_key": table_key, "query": query, "row_count": len(filtered)},
        )
