from __future__ import annotations

from application.ports.llm_port import LLMPort


_INITIAL_NEED_SYSTEM = """
Transform the user's question into a self-contained, precise,
and evidence-oriented Q0 information need.

The information need must preserve:
- the main subject of the question;
- the relevant periods;
- the values or metrics to be investigated;
- the requested explanatory factors.

Do not make the information need unnecessarily broad.

Respond only with the information need.
""".strip()


_BRANCH_NEED_SYSTEM = """
Transform the branch question and its context into a precise,
testable, and evidence-oriented information need.

Do not simply rephrase the question.
Preserve the important numerical, temporal, and causal elements.

Respond only with the information need.
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
            f"User question:\n"
            f"{user_question}\n\n"
            f"Q0 information need:",
            system=_INITIAL_NEED_SYSTEM,
        ).strip()

        return result or user_question.strip()

    def generate_branch_need(
        self,
        branch_question: str,
        context_summary: str,
    ) -> str:

        prompt = (
            f"Context:\n"
            f"{context_summary or '(none)'}\n\n"
            f"Branch question:\n"
            f"{branch_question}\n\n"
            f"Information need:"
        )

        result = self._llm.generate(
            prompt,
            system=_BRANCH_NEED_SYSTEM,
        ).strip()

        return result or branch_question.strip()