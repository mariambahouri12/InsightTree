from __future__ import annotations

from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    UNKNOWN = "unknown"


class RetrievalMethod(str, Enum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    TOOL = "tool"
    REUSED = "reused"


class ActionType(str, Enum):
    REUSE_EVIDENCE = "REUSE_EVIDENCE"
    SEARCH_DATA = "SEARCH_DATA"
    USE_TOOL = "USE_TOOL"
    SEARCH_DATA_AND_USE_TOOL = "SEARCH_DATA_AND_USE_TOOL"


class BranchStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    EXPANDED = "expanded"
    INVALID = "invalid"
    LOW_VALUE = "low_value"
    TERMINATED = "terminated"

    @property
    def is_pruned(self) -> bool:
        return self in {
            BranchStatus.INVALID,
            BranchStatus.LOW_VALUE,
            BranchStatus.TERMINATED,
        }
