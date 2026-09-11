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
    relevance: float = 0.0
    priority: float = 0.0
    children_ids: list[str] = field(default_factory=list)
    conclusion: str = ""
    failure_reason: str = ""

    def is_explorable(self) -> bool:
        return self.status == BranchStatus.ACTIVE

    def add_evidence(self, evidence: list[Evidence]) -> None:
        self.evidence.extend(evidence)


class EvidenceTree:
    def __init__(self, root_question: str) -> None:
        self.root = Branch(question=root_question, parent_id=None, depth=0)
        self.nodes: dict[str, Branch] = {self.root.id: self.root}
        self.global_evidence_pool: list[Evidence] = []
        self.discovered_facts: list[str] = []

    def add_branch(self, branch: Branch) -> None:
        if branch.parent_id is not None and branch.parent_id not in self.nodes:
            raise KeyError(f"Unknown parent branch: {branch.parent_id}")
        self.nodes[branch.id] = branch
        if branch.parent_id is not None:
            self.nodes[branch.parent_id].children_ids.append(branch.id)

    def get(self, branch_id: str) -> Branch:
        return self.nodes[branch_id]

    def unexplored_branches(self) -> list[Branch]:
        return [b for b in self.nodes.values() if b.is_explorable()]

    def has_promising_branches(self) -> bool:
        return bool(self.unexplored_branches())

    def select_next_branch(self) -> Optional[Branch]:
        candidates = self.unexplored_branches()
        return max(candidates, key=lambda b: b.priority) if candidates else None

    def resolved_branches(self) -> list[Branch]:
        return [b for b in self.nodes.values() if b.status == BranchStatus.RESOLVED]

    def register_evidence(self, evidence: list[Evidence]) -> None:
        seen = {
            (e.chunk_id, e.text.strip())
            for e in self.global_evidence_pool
        }
        for item in evidence:
            key = (item.chunk_id, item.text.strip())
            if key not in seen:
                self.global_evidence_pool.append(item)
                seen.add(key)

    def strongest_branches(self, limit: int) -> list[Branch]:
        resolved = sorted(
            self.resolved_branches(),
            key=lambda b: b.priority + b.evidence_strength,
            reverse=True,
        )
        return resolved[:limit]

    def all_questions(self) -> list[str]:
        return [b.question for b in self.nodes.values()]
