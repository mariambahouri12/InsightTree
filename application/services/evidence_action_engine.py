from __future__ import annotations

from copy import deepcopy

from application.services.evidence_action_planner_service import (
    EvidenceActionPlannerService,
)
from application.services.evidence_evaluator_service import (
    EvidenceEvaluatorService,
)
from domain.entities import ActionExecutionResult, Evidence
from domain.enums import RetrievalMethod
from infrastructure.logging.verbose_logger import vlog


class EvidenceActionEngine:
    def __init__(
        self,
        retrieval_service,
        planner: EvidenceActionPlannerService,
        evaluator: EvidenceEvaluatorService,
        evidence_pool,
        tool_registry,
        settings,
    ) -> None:
        self._retrieval = retrieval_service
        self._planner = planner
        self._evaluator = evaluator
        self._evidence_pool = evidence_pool
        self._tool_registry = tool_registry
        self._settings = settings

    # ================================================================
    # DIRECT SEARCH
    # ================================================================
    #
    # This method is intentionally separate from execute().
    #
    # It does NOT:
    # - generate a Q0
    # - call the planner
    # - call a tool
    # - perform multiple action steps
    #
    # It simply searches the indexed documents using the exact
    # question provided by the user.
    #
    # ================================================================

    def search_direct(
        self,
        user_question: str,
    ) -> ActionExecutionResult:

        question = user_question.strip()

        if not question:
            return ActionExecutionResult(
                evidence=[],
                logs=["Question vide."],
            )

        vlog(
            f"[Direct Retrieval] Recherche exacte: {question}"
        )

        evidence = self._retrieval.retrieve(
            question
        )

        if not evidence:
            vlog(
                "[Direct Retrieval] Aucune preuve trouvée."
            )

            return ActionExecutionResult(
                evidence=[],
                logs=[
                    "Recherche directe: aucune preuve trouvée."
                ],
            )

        # ------------------------------------------------------------
        # Remove duplicate evidence.
        # ------------------------------------------------------------

        evidence = self._only_new_evidence(
            evidence,
            [],
        )

        # ------------------------------------------------------------
        # Keep the direct retrieval result in the global evidence pool.
        # ------------------------------------------------------------

        if evidence:
            self._evidence_pool.add(
                evidence
            )

        vlog(
            f"[Direct Retrieval] "
            f"{len(evidence)} preuve(s) trouvée(s)."
        )

        return ActionExecutionResult(
            evidence=evidence,
            logs=[
                f"Recherche directe: "
                f"{len(evidence)} preuve(s) trouvée(s)."
            ],
        )

    # ================================================================
    # NORMAL ACTION EXECUTION
    # ================================================================

    def execute(
        self,
        information_need: str,
    ) -> ActionExecutionResult:

        collected: list[Evidence] = []
        logs: list[str] = []

        context = ""

        previous_keys: set[str] = set()

        for step_number in range(
            1,
            self._settings.max_action_steps + 1,
        ):

            # --------------------------------------------------------
            # Find reusable evidence.
            # --------------------------------------------------------

            reusable = self._find_reusable_evidence(
                information_need
            )

            # --------------------------------------------------------
            # Ask the planner what to do next.
            # --------------------------------------------------------

            plan = self._planner.plan_next_step(
                information_need=information_need,
                context=context,
                history=logs,
                has_reusable_evidence=bool(reusable),
            )

            vlog(
                f"  [Planner] "
                f"action={plan.action_type.value} "
                f"tool={plan.tool_name} "
                f"more_steps={plan.more_steps} "
                f"| motif: {plan.rationale}"
            )

            # --------------------------------------------------------
            # REUSE EXISTING EVIDENCE
            # --------------------------------------------------------

            if plan.action_type.value == "REUSE_EVIDENCE":

                if not reusable:

                    logs.append(
                        f"Step {step_number}: "
                        f"REUSE_EVIDENCE demandé mais "
                        f"aucune preuve réutilisable."
                    )

                    break

                evidence = [
                    deepcopy(e)
                    for e in reusable
                ]

                for item in evidence:
                    item.retrieval_method = (
                        RetrievalMethod.REUSED
                    )

            # --------------------------------------------------------
            # SEARCH DATA
            # --------------------------------------------------------

            elif plan.action_type.value == "SEARCH_DATA":

                evidence = self._retrieval.retrieve(
                    information_need
                )

            # --------------------------------------------------------
            # USE TOOL
            # --------------------------------------------------------

            elif plan.action_type.value == "USE_TOOL":

                input_data = None

                if plan.requires_data:
                    input_data = self._retrieval.retrieve(
                        information_need
                    )

                evidence = self._execute_tool(
                    plan.tool_name,
                    input_data,
                )

            # --------------------------------------------------------
            # SEARCH DATA + USE TOOL
            # --------------------------------------------------------

            elif (
                plan.action_type.value
                == "SEARCH_DATA_AND_USE_TOOL"
            ):

                input_data = self._retrieval.retrieve(
                    information_need
                )

                evidence = self._execute_tool(
                    plan.tool_name,
                    input_data,
                )

            # --------------------------------------------------------
            # Unknown action
            # --------------------------------------------------------

            else:

                logs.append(
                    f"Step {step_number}: "
                    f"Action inconnue."
                )

                break

            # --------------------------------------------------------
            # No evidence
            # --------------------------------------------------------

            if not evidence:

                logs.append(
                    f"Step {step_number}: "
                    f"aucune preuve récupérée."
                )

                vlog(
                    f"  [Action] Step {step_number} "
                    f"-> aucune preuve."
                )

                break

            # --------------------------------------------------------
            # Keep only novel evidence.
            # --------------------------------------------------------

            new_evidence = self._only_new_evidence(
                evidence,
                list(collected),
            )

            if not new_evidence:

                logs.append(
                    f"Step {step_number}: "
                    f"aucune nouvelle preuve."
                )

                vlog(
                    f"  [Action] Step {step_number} "
                    f"-> aucune nouvelle preuve, arrêt."
                )

                break

            # --------------------------------------------------------
            # Validate relevance.
            # --------------------------------------------------------

            if not self._evaluator.is_relevant(
                information_need,
                new_evidence,
            ):

                logs.append(
                    f"Step {step_number}: "
                    f"preuves non pertinentes."
                )

                vlog(
                    f"  [Action] Step {step_number} "
                    f"-> preuves rejetées."
                )

                break

            # --------------------------------------------------------
            # Add evidence.
            # --------------------------------------------------------

            collected.extend(
                new_evidence
            )

            self._evidence_pool.add(
                new_evidence
            )

            logs.append(
                f"Step {step_number}: "
                f"{len(new_evidence)} nouvelle(s) preuve(s)."
            )

            # --------------------------------------------------------
            # Build context for the next planner step.
            # --------------------------------------------------------

            context = self._summarize(
                collected
            )

            # --------------------------------------------------------
            # Deterministic sufficiency check.
            #
            # This is important:
            # do not blindly trust planner.more_steps.
            # --------------------------------------------------------

            sufficient, confidence = (
                self._evaluator.is_sufficient(
                    information_need,
                    collected,
                )
            )

            vlog(
                f"  [Action Sufficiency] "
                f"sufficient={sufficient} "
                f"confidence={confidence:.2f}"
            )

            if sufficient:

                logs.append(
                    f"Step {step_number}: "
                    f"preuves suffisantes."
                )

                return ActionExecutionResult(
                    evidence=collected,
                    logs=logs,
                )

            # --------------------------------------------------------
            # Stop if planner says there is no need for another step.
            # --------------------------------------------------------

            if not plan.more_steps:

                return ActionExecutionResult(
                    evidence=collected,
                    logs=logs,
                )

            # --------------------------------------------------------
            # Prevent repeated retrieval of the same evidence.
            # --------------------------------------------------------

            current_keys = {
                self._evidence_key(e)
                for e in new_evidence
            }

            if current_keys.issubset(previous_keys):

                logs.append(
                    f"Step {step_number}: "
                    f"aucun progrès détecté."
                )

                break

            previous_keys.update(
                current_keys
            )

        return ActionExecutionResult(
            evidence=collected,
            logs=logs,
        )

    # ================================================================
    # REUSABLE EVIDENCE
    # ================================================================

    def _find_reusable_evidence(
        self,
        information_need: str,
    ) -> list[Evidence]:

        query_embedding = (
            self._retrieval.embed_query(
                information_need
            )
        )

        return self._evidence_pool.search(
            query_embedding,
            top_k=self._settings.reuse_top_k,
            min_score=self._settings.reuse_similarity_threshold,
        )

    # ================================================================
    # TOOL EXECUTION
    # ================================================================

    def _execute_tool(
        self,
        tool_name: str | None,
        input_data,
    ) -> list[Evidence]:

        if not tool_name:
            return []

        tool = self._tool_registry.get(
            tool_name
        )

        if tool is None:
            return []

        try:

            result = tool.execute(
                input_data
            )

        except Exception as exc:

            vlog(
                f"  [Tool] {tool_name} "
                f"-> erreur: {exc}"
            )

            return []

        if result is None:
            return []

        if isinstance(result, list):

            return [
                Evidence(
                    text=str(item),
                    score=1.0,
                    retrieval_method=RetrievalMethod.TOOL,
                    source_reliability=1.0,
                )
                for item in result
            ]

        return [
            Evidence(
                text=str(result),
                score=1.0,
                retrieval_method=RetrievalMethod.TOOL,
                source_reliability=1.0,
            )
        ]

    # ================================================================
    # DEDUPLICATION
    # ================================================================

    @staticmethod
    def _only_new_evidence(
        evidence: list[Evidence],
        existing: list[Evidence],
    ) -> list[Evidence]:

        existing_keys = {
            EvidenceActionEngine._evidence_key(e)
            for e in existing
        }

        result: list[Evidence] = []

        for item in evidence:

            key = EvidenceActionEngine._evidence_key(
                item
            )

            if key in existing_keys:
                continue

            existing_keys.add(key)

            result.append(item)

        return result

    @staticmethod
    def _evidence_key(
        evidence: Evidence,
    ) -> str:

        chunk_id = getattr(
            evidence,
            "chunk_id",
            None,
        )

        if chunk_id:
            return f"chunk:{chunk_id}"

        evidence_id = getattr(
            evidence,
            "id",
            None,
        )

        if evidence_id:
            return f"id:{evidence_id}"

        return (
            "text:"
            + " ".join(
                evidence.text.lower().split()
            )
        )

    # ================================================================
    # CONTEXT SUMMARY
    # ================================================================

    @staticmethod
    def _summarize(
        evidence: list[Evidence],
    ) -> str:

        return "\n".join(
            f"- {e.text[:500]}"
            for e in evidence[-8:]
        )