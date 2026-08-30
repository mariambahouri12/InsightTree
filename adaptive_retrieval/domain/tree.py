"""Structure d'arbre représentant l'exploration adaptative des branches
(questions -> preuves -> sous-questions) décrite dans le diagramme UML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from domain.entities import Evidence, new_id
from domain.enums import BranchStatus


@dataclass
class Branch:
    question: str
    parent_id: Optional[str]
    depth: int
    id: str = field(default_factory=new_id)
    status: BranchStatus = BranchStatus.ACTIVE
    evidence: list[Evidence] = field(default_factory=list)
    context_summary: str = ""
    evidence_strength: float = 0.0
    information_gain: float = 0.0
    priority: float = 1.0
    children_ids: list[str] = field(default_factory=list)
    conclusion: str = ""

    def is_explorable(self) -> bool:
        return self.status == BranchStatus.ACTIVE

    def add_evidence(self, evidence: list[Evidence]) -> None:
        self.evidence.extend(evidence)


class EvidenceTree:
    """Agrégat racine : l'arbre complet + le pool global de preuves."""

    def __init__(self, root_question: str) -> None:
        self.root = Branch(question=root_question, parent_id=None, depth=0)
        self.nodes: dict[str, Branch] = {self.root.id: self.root}
        self.global_evidence_pool: list[Evidence] = []
        self.discovered_facts: list[str] = []

    def add_branch(self, branch: Branch) -> None:
        self.nodes[branch.id] = branch
        if branch.parent_id is not None:
            self.nodes[branch.parent_id].children_ids.append(branch.id)

    def get(self, branch_id: str) -> Branch:
        return self.nodes[branch_id]

    def unexplored_branches(self) -> list[Branch]:
        return [b for b in self.nodes.values() if b.is_explorable()]

    def select_next_branch(self) -> Optional[Branch]:
        candidates = self.unexplored_branches()
        if not candidates:
            return None
        return max(candidates, key=lambda b: b.priority)

    def supported_branches(self) -> list[Branch]:
        return [b for b in self.nodes.values() if b.status == BranchStatus.SUPPORTED]

    def register_evidence(self, evidence: list[Evidence]) -> None:
        existing_ids = {e.id for e in self.global_evidence_pool}
        for e in evidence:
            if e.id not in existing_ids:
                self.global_evidence_pool.append(e)

    def strongest_branches(self, limit: int) -> list[Branch]:
        supported = sorted(
            self.supported_branches(), key=lambda b: b.evidence_strength, reverse=True
        )
        return supported[:limit]
