from __future__ import annotations

import pandas as pd


class StructuredDataRegistry:
    def __init__(self) -> None:
        self._tables: dict[str, pd.DataFrame] = {}

    def register(self, source_path: str, sheet_name: str, df: pd.DataFrame) -> None:
        self._tables[f"{source_path}::{sheet_name}"] = df

    def list_tables(self) -> dict[str, pd.DataFrame]:
        return dict(self._tables)

    def is_empty(self) -> bool:
        return not self._tables
