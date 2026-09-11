from __future__ import annotations

from application.ports.llm_port import LLMPort
from application.ports.tool_port import ToolRegistryPort
from domain.entities import ActionPlanStep
from domain.enums import ActionType
from infrastructure.logging.verbose_logger import vlog


_PLANNER_SYSTEM = """
Tu es le planner de l'Evidence Action Engine.

Ta tâche est de choisir UNE seule action pour satisfaire le besoin
d'information actuel.

Actions possibles:
- REUSE_EVIDENCE
- SEARCH_DATA
- USE_TOOL
- SEARCH_DATA_AND_USE_TOOL

RÈGLES IMPORTANTES:

1. Si une preuve réutilisable est disponible et qu'elle répond déjà
   au besoin, choisis REUSE_EVIDENCE.

2. Si aucune preuve pertinente n'est disponible et qu'il faut rechercher
   des documents, choisis SEARCH_DATA.

3. Si des données structurées doivent être calculées ou interrogées,
   utilise un outil approprié.

4. NE choisis PAS plusieurs fois SEARCH_DATA si le contexte contient
   déjà des preuves pertinentes pour le besoin.

5. `more_steps` doit être false lorsque les preuves disponibles sont
   déjà suffisantes pour répondre au besoin.

6. `more_steps` doit être true uniquement lorsqu'une information
   réellement manquante nécessite une nouvelle action.

7. Ne demande jamais une nouvelle recherche simplement pour obtenir
   davantage de preuves similaires.

JSON strict:
{
  "action_type":"...",
  "tool_name":null,
  "requires_data":false,
  "more_steps":false,
  "rationale":"..."
}
""".strip()


class EvidenceActionPlannerService:
    def __init__(
        self,
        llm: LLMPort,
        tool_registry: ToolRegistryPort,
    ) -> None:
        self._llm = llm
        self._tool_registry = tool_registry

    def plan_next_step(
        self,
        information_need: str,
        context_summary: str,
        executed_steps: list[str],
        has_reusable_evidence: bool,
    ) -> ActionPlanStep:

        tools = "\n".join(
            f"- {t.name}: {t.description} "
            f"(requires_data={t.requires_data})"
            for t in self._tool_registry.list_tools()
        ) or "(aucun outil)"

        prompt = (
            f"Besoin d'information:\n"
            f"{information_need}\n\n"

            f"Preuves déjà obtenues:\n"
            f"{context_summary or '(aucune)'}\n\n"

            f"Preuve réutilisable disponible: "
            f"{has_reusable_evidence}\n\n"

            f"Outils disponibles:\n"
            f"{tools}\n\n"

            f"Historique des actions:\n"
            f"{chr(10).join(executed_steps) or '(aucune)'}"
        )

        result = self._llm.generate_json(
            prompt,
            system=_PLANNER_SYSTEM,
        )

        plan_step = self._to_plan_step(
            result,
            has_reusable_evidence,
            context_summary,
            executed_steps,
        )

        vlog(
            f"  [Planner] action={plan_step.action_type.value} "
            f"tool={plan_step.tool_name} "
            f"more_steps={plan_step.more_steps} "
            f"| motif: {plan_step.rationale[:150]}"
        )

        return plan_step

    def _to_plan_step(
        self,
        result: dict,
        has_reusable_evidence: bool,
        context_summary: str,
        executed_steps: list[str],
    ) -> ActionPlanStep:

        try:
            action = ActionType(
                str(
                    result.get(
                        "action_type",
                        ActionType.SEARCH_DATA.value,
                    )
                )
            )
        except ValueError:
            action = ActionType.SEARCH_DATA

        if (
            action == ActionType.REUSE_EVIDENCE
            and not has_reusable_evidence
        ):
            action = ActionType.SEARCH_DATA

        tool_name = (
            str(result.get("tool_name", "")).strip()
            or None
        )

        if action in {
            ActionType.USE_TOOL,
            ActionType.SEARCH_DATA_AND_USE_TOOL,
        }:
            tool = (
                self._tool_registry.get_tool(tool_name)
                if tool_name
                else None
            )

            if tool is None:
                action = ActionType.SEARCH_DATA
                tool_name = None
                requires_data = False
            else:
                requires_data = tool.requires_data

        else:
            tool_name = None
            requires_data = False

        more_steps = bool(
            result.get("more_steps", False)
        )

        # Safety rule:
        # if SEARCH_DATA has already been executed and we already have
        # context, do not blindly repeat it. The engine will evaluate
        # sufficiency after the current retrieval.
        search_already_done = any(
            "SEARCH_DATA: OK" in step
            or "SEARCH_DATA_AND_USE_TOOL: OK" in step
            for step in executed_steps
        )

        if (
            action == ActionType.SEARCH_DATA
            and search_already_done
            and context_summary.strip()
        ):
            more_steps = False

        return ActionPlanStep(
            action_type=action,
            tool_name=tool_name,
            requires_data=requires_data,
            more_steps=more_steps,
            rationale=str(
                result.get("rationale", "")
            ).strip(),
        )