from __future__ import annotations

import re

from application.ports.llm_port import LLMPort
from domain.entities import Evidence, FinalAnswer, Hypothesis
from domain.tree import Branch


_SYNTHESIS_SYSTEM = """
Tu es l'analyste final d'un système d'investigation fondé sur des preuves.

Réponds à la question originale uniquement à partir des preuves fournies.

RÈGLES:
1. Ne crée aucune information absente des preuves.
2. Donne les chiffres exacts lorsqu'ils sont disponibles.
3. Explique les facteurs qui sont explicitement présents dans les preuves.
4. Lorsqu'il existe une contradiction entre deux sources, signale-la
   explicitement au lieu de choisir arbitrairement une version.
5. Distingue clairement:
   - faits observés;
   - interprétations;
   - incertitudes ou contradictions.
6. Pour une évolution temporelle, calcule correctement la variation
   absolue et/ou relative lorsque les chiffres nécessaires sont présents.
7. Les calculs doivent être vérifiables.
8. Ne présente jamais une hypothèse comme un fait.

JSON strict:
{"answer":"...", "confidence":0..1}
""".strip()


class SynthesisService:
    def __init__(
        self,
        llm: LLMPort,
    ) -> None:
        self._llm = llm

    def synthesize(
        self,
        original_question: str,
        strongest_branches: list[Branch],
        hypotheses: list[Hypothesis],
        fallback_evidence: list[Evidence] | None = None,
    ) -> FinalAnswer:

        evidence = self._select_strongest_evidence(
            strongest_branches,
            fallback_evidence or [],
        )

        evidence_by_id = {
            e.id: e
            for e in evidence
        }

        hypothesis_blocks = []

        for hypothesis in hypotheses:

            supporting = [
                evidence_by_id[eid]
                for eid in hypothesis.supporting_evidence_ids
                if eid in evidence_by_id
            ]

            contradicting = [
                evidence_by_id[eid]
                for eid in hypothesis.contradicting_evidence_ids
                if eid in evidence_by_id
            ]

            hypothesis_blocks.append(
                f"Hypothèse: {hypothesis.statement}\n"
                f"Confiance: {hypothesis.confidence:.2f}\n"
                f"Incertainité restante: "
                f"{hypothesis.remaining_uncertainty:.2f}\n"
                f"Soutien:\n"
                f"{self._format_evidence(supporting) or '(aucun)'}\n"
                f"Contradiction:\n"
                f"{self._format_evidence(contradicting) or '(aucune)'}"
            )

        branch_blocks = []

        for branch in strongest_branches:

            branch_blocks.append(
                f"Branche: {branch.question}\n"
                f"Conclusion: "
                f"{branch.conclusion or '(aucune)'}\n"
                f"Force: {branch.evidence_strength:.2f}\n"
                f"Priorité: {branch.priority:.2f}\n"
                f"Preuves:\n"
                f"{self._format_evidence(branch.evidence[:8])}"
            )

        prompt = (
            f"Question originale:\n"
            f"{original_question}\n\n"

            f"Branches fortes:\n"
            f"{chr(10).join(branch_blocks) or '(aucune)'}\n\n"

            f"Hypothèses:\n"
            f"{chr(10).join(hypothesis_blocks) or '(aucune)'}\n\n"

            f"Preuves sélectionnées:\n"
            f"{self._format_evidence(evidence)}"
        )

        result = self._llm.generate_json(
            prompt,
            system=_SYNTHESIS_SYSTEM,
        )

        answer = str(
            result.get("answer", "")
        ).strip()

        confidence = self._bounded(
            result.get("confidence"),
            0.0,
        )

        if not answer:
            answer = self._fallback_answer(
                original_question,
                evidence,
            )

        return FinalAnswer(
            text=answer,
            hypotheses=hypotheses,
            citations=evidence,
            confidence=confidence,
        )

    @staticmethod
    def _select_strongest_evidence(
        branches: list[Branch],
        fallback: list[Evidence],
        limit: int = 12,
    ) -> list[Evidence]:

        candidates = []

        for branch in branches:
            candidates.extend(
                branch.evidence
            )

        if not candidates:
            candidates = list(fallback)

        unique = {}

        for evidence in candidates:

            key = (
                evidence.chunk_id
                or evidence.text.strip()
            )

            if (
                key not in unique
                or evidence.score
                > unique[key].score
            ):
                unique[key] = evidence

        return sorted(
            unique.values(),
            key=lambda e: (
                e.score
                * e.source_reliability
            ),
            reverse=True,
        )[:limit]

    @staticmethod
    def _format_evidence(
        evidence: list[Evidence],
    ) -> str:

        if not evidence:
            return "(aucune)"

        return "\n---\n".join(
            f"[{e.citation_label()} | "
            f"score={e.score:.2f} | "
            f"reliability={e.source_reliability:.2f}] "
            f"{e.text[:700]}"
            for e in evidence
        )

    @staticmethod
    def _bounded(
        value,
        default,
    ):

        try:
            return max(
                0.0,
                min(1.0, float(value)),
            )
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _fallback_answer(
        question: str,
        evidence: list[Evidence],
    ) -> str:

        if not evidence:
            return (
                "Aucune preuve suffisante n'a été trouvée "
                f"pour répondre à : {question}"
            )

        return (
            "Les éléments disponibles sont : "
            + " ".join(
                f"[{e.citation_label()}] "
                f"{e.text[:250]}"
                for e in evidence[:3]
            )
        )
    