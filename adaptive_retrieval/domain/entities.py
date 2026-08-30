"""Entités métier. Aucune dépendance vers l'infrastructure (clean architecture)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from domain.enums import DocumentType, RetrievalMethod


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Document:
    source_path: str
    doc_type: DocumentType
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)


@dataclass
class Chunk:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    id: str = field(default_factory=new_id)


@dataclass
class Evidence:
    """Un élément de preuve retourné par la couche de récupération (retrieval)."""

    chunk_id: str
    text: str
    score: float
    source_metadata: dict[str, Any]
    retrieval_method: RetrievalMethod
    id: str = field(default_factory=new_id)

    def citation_label(self) -> str:
        source = self.source_metadata.get("source_path", "unknown")
        page = self.source_metadata.get("page")
        return f"{source}" + (f" (p.{page})" if page is not None else "")


@dataclass
class Hypothesis:
    statement: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    id: str = field(default_factory=new_id)


@dataclass
class FinalAnswer:
    text: str
    hypotheses: list[Hypothesis]
    citations: list[Evidence]
    confidence: float
