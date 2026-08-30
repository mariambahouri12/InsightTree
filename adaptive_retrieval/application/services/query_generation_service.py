"""Transforme la question utilisateur / les questions de branche en
requêtes de recherche optimisées. L'utilisateur ne va jamais directement
au retriever : c'est toujours le LLM qui reformule."""
from __future__ import annotations

from application.ports.llm_port import LLMPort

_INITIAL_QUESTION_SYSTEM = (
    "Tu es un assistant de recherche documentaire. Ta tâche est de transformer "
    "la question d'un utilisateur en UNE question de recherche initiale (Q0), "
    "claire, autonome et orientée vers la recherche de preuves factuelles. "
    "Réponds uniquement avec la question, sans préambule."
)

_REFORMULATE_SYSTEM = (
    "Tu es un moteur de reformulation de requêtes pour un système de recherche "
    "hybride (dense + lexical). Transforme la question de branche fournie en "
    "une requête de recherche courte et efficace (mots-clés + concepts), en "
    "tenant compte du contexte déjà découvert. Réponds uniquement avec la "
    "requête, sans préambule ni guillemets."
)


class QueryGenerationService:
    def __init__(self, llm: LLMPort) -> None:
        self._llm = llm

    def generate_initial_question(self, user_question: str) -> str:
        prompt = f"Question utilisateur:\n{user_question}\n\nQuestion de recherche initiale (Q0):"
        return self._llm.generate(prompt, system=_INITIAL_QUESTION_SYSTEM).strip()

    def reformulate_branch_question(self, branch_question: str, context_summary: str) -> str:
        prompt = (
            f"Contexte déjà découvert:\n{context_summary or '(aucun)'}\n\n"
            f"Question de branche à reformuler:\n{branch_question}\n\n"
            "Requête de recherche optimisée:"
        )
        return self._llm.generate(prompt, system=_REFORMULATE_SYSTEM).strip()
