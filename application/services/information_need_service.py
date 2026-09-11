from __future__ import annotations

from application.ports.llm_port import LLMPort


_INITIAL_NEED_SYSTEM = """
Transforme la question utilisateur en un besoin d'information Q0
autonome, précis et orienté preuve.

Le besoin doit conserver:
- l'objet principal de la question;
- les périodes concernées;
- les valeurs ou métriques à rechercher;
- les facteurs explicatifs demandés.

Ne rends pas le besoin inutilement général.

Réponds uniquement avec le besoin d'information.
""".strip()


_BRANCH_NEED_SYSTEM = """
Transforme la question d'une branche et son contexte en un besoin
d'information précis, testable et orienté preuve.

Ne reformule pas simplement la question.
Conserve les éléments numériques, temporels et causaux importants.

Réponds uniquement avec le besoin d'information.
""".strip()


class InformationNeedService:
    def __init__(
        self,
        llm: LLMPort,
    ) -> None:
        self._llm = llm

    def generate_initial_need(
        self,
        user_question: str,
    ) -> str:

        result = self._llm.generate(
            f"Question utilisateur:\n"
            f"{user_question}\n\n"
            f"Besoin Q0:",
            system=_INITIAL_NEED_SYSTEM,
        ).strip()

        return result or user_question.strip()

    def generate_branch_need(
        self,
        branch_question: str,
        context_summary: str,
    ) -> str:

        prompt = (
            f"Contexte:\n"
            f"{context_summary or '(aucun)'}\n\n"
            f"Question de branche:\n"
            f"{branch_question}\n\n"
            f"Besoin:"
        )

        result = self._llm.generate(
            prompt,
            system=_BRANCH_NEED_SYSTEM,
        ).strip()

        return result or branch_question.strip()