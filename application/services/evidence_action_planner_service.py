from __future__ import annotations

from application.ports.llm_port import LLMPort
from application.ports.tool_port import ToolRegistryPort
from domain.entities import ActionPlanStep
from domain.enums import ActionType
from infrastructure.logging.verbose_logger import vlog


_PLANNER_SYSTEM = """
You are the planner of the Evidence Action Engine.

Your task is to choose ONE single action to satisfy the current
information need.

Possible actions:
- REUSE_EVIDENCE
- SEARCH_DATA
- USE_TOOL
- SEARCH_DATA_AND_USE_TOOL

IMPORTANT RULES:

1. If reusable evidence is available and already addresses
   the information need, choose REUSE_EVIDENCE.

2. If no relevant evidence is available and documents need to be
   searched, choose SEARCH_DATA.

3. If structured data needs to be calculated or queried,
   use an appropriate tool.

4. Do NOT choose SEARCH_DATA multiple times if the context already
   contains relevant evidence for the information need.

5. `more_steps` must be false when the available evidence is
   already sufficient to answer the information need.

6. `more_steps` must be true only when genuinely missing information
   requires a new action.

7. Never request a new search simply to obtain
   more similar evidence.

Strict JSON:
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
        ) or "(no tools)"

        prompt = (
            f"Information need:\n"
            f"{information_need}\n\n"

            f"Evidence already obtained:\n"
            f"{context_summary or '(none)'}\n\n"

            f"Reusable evidence available: "
            f"{has_reusable_evidence}\n\n"

            f"Available tools:\n"
            f"{tools}\n\n"

            f"Action history:\n"
            f"{chr(10).join(executed_steps) or '(none)'}"
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
            f"| reason: {plan_step.rationale[:150]}"
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