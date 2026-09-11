from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from domain.enums import ActionType, DocumentType, RetrievalMethod


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
    text: str
    score: float
    source_metadata: dict[str, Any]
    retrieval_method: RetrievalMethod
    chunk_id: Optional[str] = None
    embedding: Optional[list[float]] = None
    source_reliability: float = 0.5
    id: str = field(default_factory=new_id)

    def citation_label(self) -> str:
        source = self.source_metadata.get(
            "source_path",
            self.source_metadata.get("tool_name", "unknown"),
        )
        page = self.source_metadata.get("page")
        if page is not None:
            return f"{source} (p.{page})"
        return str(source)


@dataclass
class Hypothesis:
    statement: str
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    remaining_uncertainty: float = 1.0
    id: str = field(default_factory=new_id)


@dataclass
class FinalAnswer:
    text: str
    hypotheses: list[Hypothesis]
    citations: list[Evidence]
    confidence: float


@dataclass
class ActionPlanStep:
    action_type: ActionType
    tool_name: Optional[str]
    requires_data: bool
    more_steps: bool
    rationale: str = ""


@dataclass
class ActionExecutionResult:
    evidence: list[Evidence]
    step_count: int
    step_log: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.evidence


@dataclass
class ToolResult:
    success: bool
    output_text: str
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
