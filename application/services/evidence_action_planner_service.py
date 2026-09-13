from __future__ import annotations

from application.ports.llm_port import LLMPort
from application.ports.tool_port import ToolRegistryPort
from domain.entities import ActionPlanStep
from domain.enums import ActionType
from infrastructure.logging.verbose_logger import vlog


_PLANNER_SYSTEM = """
You are the planner of the Evidence Action Engine.

Your task is to choose the next action for the current information need.

Possible actions:

- REUSE_EVIDENCE
- SEARCH_DATA
- USE_TOOL
- SEARCH_DATA_AND_USE_TOOL

IMPORTANT DECISION RULES:

1. REUSE_EVIDENCE

Choose REUSE_EVIDENCE only when the existing evidence directly
contains the answer and NO computation, transformation, aggregation,
comparison, conversion, parsing, or other tool operation is required.

Examples:

Question:
"What is the revenue in Q1 2024?"

Evidence:
"Q1 2024 revenue was 1,250 thousand euros."

Action:
REUSE_EVIDENCE


2. USE_TOOL

Choose USE_TOOL when the required information is already present
in the evidence but the answer requires a computation or another
operation that should be performed by a tool.

IMPORTANT:
Do NOT choose REUSE_EVIDENCE merely because the evidence contains
all the raw values.

If the question asks for:

- a sum
- an average
- a percentage
- a difference
- a ratio
- a multiplication
- a division
- a total
- an aggregation
- a numerical comparison

and the required input values are already available in the evidence,
choose USE_TOOL.

Example:

Question:
"What is the total revenue for Q1, Q2, Q3 and Q4 of 2024?"

Evidence:
"Q1 = 1250"
"Q2 = 1320"
"Q3 = 1410"
"Q4 = 1370"

Action:
USE_TOOL

Tool:
calculator


3. SEARCH_DATA

Choose SEARCH_DATA when relevant information is missing from the
current context and documents or data need to be searched.

Example:

Question:
"What was the revenue in Q2 2024?"

No relevant Q2 evidence exists.

Action:
SEARCH_DATA


4. SEARCH_DATA_AND_USE_TOOL

Choose SEARCH_DATA_AND_USE_TOOL when the answer requires both:

- retrieving missing input data
- executing a tool using those inputs

Example:

Question:
"What was the total revenue for 2024?"

Only Q1 and Q2 are currently available.

Action:
SEARCH_DATA_AND_USE_TOOL

Tool:
calculator


5. REUSE_EVIDENCE HAS LOWER PRIORITY THAN TOOL EXECUTION

When both conditions are true:

- evidence is sufficient
- computation is required

choose USE_TOOL, NOT REUSE_EVIDENCE.


6. TOOL SELECTION

The tool_name must be null unless the selected action requires
a tool.

For arithmetic calculations, use the calculator tool when available.


7. MORE_STEPS

Set more_steps to true only when another action is genuinely
required after the current action.

If the selected action is USE_TOOL and that tool can complete the
current information need, set more_steps to false.

If SEARCH_DATA is required before another action, set more_steps
to true.

Return strict JSON:

{
  "action_type": "...",
  "tool_name": null,
  "more_steps": false,
  "rationale": "..."
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

    # =================================================================
    # PLAN NEXT STEP
    # =================================================================

    def plan_next_step(
        self,
        information_need: str,
        context_summary: str,
        executed_steps: list[str],
        has_reusable_evidence: bool,
    ) -> ActionPlanStep:

        tools = "\n".join(
            f"- {tool.name}: "
            f"{tool.description} "
            f"(requires_data={tool.requires_data})"
            for tool in self._tool_registry.list_tools()
        )

        if not tools:
            tools = "(no tools)"

        history = (
            "\n".join(executed_steps)
            if executed_steps
            else "(none)"
        )

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
            f"{history}"
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
            f"  [Planner] "
            f"action={plan_step.action_type.value} "
            f"tool={plan_step.tool_name} "
            f"more_steps={plan_step.more_steps} "
            f"| reason: "
            f"{plan_step.rationale[:150]}"
        )

        return plan_step

    # =================================================================
    # CONVERT LLM RESULT
    # =================================================================

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

        # -------------------------------------------------------------
        # REUSE_EVIDENCE VALIDATION
        # -------------------------------------------------------------

        if (
            action == ActionType.REUSE_EVIDENCE
            and not has_reusable_evidence
        ):
            action = ActionType.SEARCH_DATA

        # -------------------------------------------------------------
        # TOOL VALIDATION
        # -------------------------------------------------------------

        tool_name = (
            str(
                result.get(
                    "tool_name",
                    "",
                )
            ).strip()
            or None
        )

        requires_data = False

        if action in {
            ActionType.USE_TOOL,
            ActionType.SEARCH_DATA_AND_USE_TOOL,
        }:

            tool = (
                self._tool_registry.get_tool(
                    tool_name
                )
                if tool_name
                else None
            )

            if tool is None:

                # Try to find calculator automatically
                # when the LLM forgot the tool name.
                calculator = (
                    self._tool_registry.get_tool(
                        "calculator"
                    )
                )

                if calculator is not None:

                    tool = calculator
                    tool_name = calculator.name

                else:

                    action = ActionType.SEARCH_DATA
                    tool_name = None
                    requires_data = False

            if tool is not None:
                requires_data = bool(
                    tool.requires_data
                )

        else:

            tool_name = None
            requires_data = False

        # -------------------------------------------------------------
        # MORE STEPS
        # -------------------------------------------------------------

        more_steps = bool(
            result.get(
                "more_steps",
                False,
            )
        )

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

        # A tool that can directly complete the request does not need
        # another planner iteration.
        if action == ActionType.USE_TOOL:
            more_steps = False

        return ActionPlanStep(
            action_type=action,
            tool_name=tool_name,
            requires_data=requires_data,
            more_steps=more_steps,
            rationale=str(
                result.get(
                    "rationale",
                    "",
                )
            ).strip(),
        )