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


class BranchStatus(str, Enum):
    ACTIVE = "active"
    SUPPORTED = "supported"
    PRUNED_IRRELEVANT = "pruned_irrelevant"
    PRUNED_LOW_INFO_GAIN = "pruned_low_info_gain"
    PRUNED_NO_CHILDREN = "pruned_no_children"
