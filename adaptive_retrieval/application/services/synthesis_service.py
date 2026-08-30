"""Synthèse finale : compare les hypothèses, produit la réponse et
rattache les citations des preuves utilisées."""
from __future__ import annotations

from application.ports.llm_port import LLMPort
from domain.entities import Evidence, FinalAnswer, Hypothesis
from domain.tree import Branch

_SYNTHESIS_SYSTEM = (
    "Tu es un analyste qui rédige une réponse finale rigoureuse à une "
    "question, en te basant STRICTEMENT sur les preuves et hypothèses "
    "fournies. Compare les hypothèses (preuves à l'appui / contre), "
    "indique le niveau de confiance global, et conclus clairement. "
    "Réponds UNIQUEMENT en JSON strict: "
    '{"answer": "...", "confidence": <0..1>}'
)


class SynthesisService:
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def synthesize(
        self,
        original_question: str,
        strongest_branches: list[Branch],
        hypotheses: list[Hypothesis],
    ) -> FinalAnswer:
        branches_repr = "\n\n".join(
            f"Branche: {b.question}\nConclusion: {b.conclusion or '(non conclue)'}\n"
            f"Preuves clés:\n"
            + "\n".join(f"  - [{e.citation_label()}] {e.text[:400]}" for e in b.evidence[:4])
            for b in strongest_branches
        )
        hypotheses_repr = "\n".join(
            f"- {h.statement} (confiance: {h.confidence:.2f}, "
            f"{len(h.supporting_evidence_ids)} preuve(s) à l'appui)"
            for h in hypotheses
        ) or "(aucune hypothèse formulée)"

        prompt = (
            f"Question originale:\n{original_question}\n\n"
            f"Branches les plus fortes:\n{branches_repr}\n\n"
            f"Hypothèses:\n{hypotheses_repr}\n\nRéponse finale:"
        )
        result = self._llm.generate_json(prompt, system=_SYNTHESIS_SYSTEM)
        answer_text = str(result.get("answer", "")).strip()
        confidence = float(result.get("confidence", 0.0))

        citations: list[Evidence] = []
        seen_chunk_ids: set[str] = set()
        for b in strongest_branches:
            for e in b.evidence:
                if e.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(e.chunk_id)
                    citations.append(e)

        return FinalAnswer(
            text=answer_text,
            hypotheses=hypotheses,
            citations=citations,
            confidence=confidence,
        )
