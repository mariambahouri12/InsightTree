from __future__ import annotations

from copy import deepcopy
from typing import Any

from application.services.calculation_intent_extractor_service import (
    CalculationIntentExtractorService,
)
from application.services.evidence_action_planner_service import (
    EvidenceActionPlannerService,
)
from application.services.evidence_evaluator_service import (
    EvidenceEvaluatorService,
)
from domain.entities import (
    ActionExecutionResult,
    Evidence,
    ToolResult,
)
from domain.enums import (
    ActionType,
    RetrievalMethod,
)
from infrastructure.logging.verbose_logger import (
    vlog,
    vlog_evidence,
    vlog_evidence_list,
    vlog_kv,
    vlog_section,
    vlog_subsection,
    vlog_text,
)


class EvidenceActionEngine:

    def __init__(
        self,
        retrieval_service,
        planner: EvidenceActionPlannerService,
        evaluator: EvidenceEvaluatorService,
        evidence_pool,
        tool_registry,
        settings,
        calculation_intent_extractor: CalculationIntentExtractorService,
    ) -> None:

        self._retrieval = retrieval_service
        self._planner = planner
        self._evaluator = evaluator
        self._evidence_pool = evidence_pool
        self._tool_registry = tool_registry
        self._settings = settings
        self._calculation_intent_extractor = (
            calculation_intent_extractor
        )

    # =================================================================
    # DIRECT SEARCH
    # =================================================================

    def search_direct(
        self,
        user_question: str,
    ) -> ActionExecutionResult:

        question = user_question.strip()

        vlog_section(
            "[DIRECT SEARCH]"
        )

        vlog_kv(
            "query",
            question,
        )

        if not question:

            vlog(
                "Question vide."
            )

            return ActionExecutionResult(
                evidence=[],
                step_count=0,
                step_log=[
                    "Question vide."
                ],
            )

        vlog(
            "[DIRECT SEARCH] Lancement du retrieval..."
        )

        evidence = (
            self._retrieval.retrieve(
                question
            )
        )

        if not evidence:

            vlog(
                "[DIRECT SEARCH] "
                "Aucune preuve trouvée."
            )

            return ActionExecutionResult(
                evidence=[],
                step_count=1,
                step_log=[
                    "Recherche directe: "
                    "aucune preuve trouvée."
                ],
            )

        evidence = (
            self._only_new_evidence(
                evidence,
                [],
            )
        )

        if evidence:

            self._evidence_pool.add(
                evidence
            )

        vlog_kv(
            "evidence_count",
            len(evidence),
        )

        vlog_evidence_list(
            evidence,
            title="DIRECT RETRIEVED EVIDENCE",
        )

        return ActionExecutionResult(
            evidence=evidence,
            step_count=1,
            step_log=[
                f"Recherche directe: "
                f"{len(evidence)} preuve(s) trouvée(s)."
            ],
        )

    # =================================================================
    # NORMAL ACTION EXECUTION
    # =================================================================

    def execute(
        self,
        information_need: str,
        initial_evidence: list[Evidence] | None = None,
    ) -> ActionExecutionResult:

        vlog_section(
            "[EVIDENCE ACTION ENGINE]"
        )

        vlog_kv(
            "information_need",
            information_need,
        )

        vlog_kv(
            "max_action_steps",
            self._settings.max_action_steps,
        )

        logs: list[str] = []

        collected: list[Evidence] = []

        if initial_evidence:

            collected = [
                deepcopy(e)
                for e in initial_evidence
            ]

            vlog_evidence_list(
                collected,
                title="INITIAL EVIDENCE",
            )

            logs.append(
                f"Initial evidence: "
                f"{len(collected)} preuve(s)."
            )

        else:

            vlog(
                "Aucune evidence initiale."
            )

            logs.append(
                "Initial evidence: 0 preuve."
            )

        context = self._summarize(
            collected
        )

        previous_keys: set[str] = set()

        # Nombre réel d'étapes exécutées.
        executed_step_count = 0

        # =================================================================
        # ACTION LOOP
        # =================================================================

        for step_number in range(
            1,
            self._settings.max_action_steps + 1,
        ):

            executed_step_count = step_number

            logs.append(
                f"Step {step_number}: début."
            )

            vlog_section(
                f"[ACTION STEP {step_number}]"
            )

            vlog_kv(
                "step",
                step_number,
            )

            vlog_kv(
                "collected_before_step",
                len(collected),
            )

            # ---------------------------------------------------------
            # REUSABLE EVIDENCE
            # ---------------------------------------------------------

            vlog_subsection(
                "1. FIND REUSABLE EVIDENCE"
            )

            reusable = (
                self._find_reusable_evidence(
                    information_need
                )
            )

            vlog_kv(
                "pool_results",
                len(reusable),
            )

            if reusable:

                vlog_evidence_list(
                    reusable,
                    title="POOL RETRIEVED EVIDENCE",
                )

            if collected:

                reusable = (
                    self._merge_unique_evidence(
                        reusable,
                        collected,
                    )
                )

            vlog_kv(
                "reusable_after_merge",
                len(reusable),
            )

            if reusable:

                vlog_evidence_list(
                    reusable,
                    title="REUSABLE EVIDENCE",
                )

            # ---------------------------------------------------------
            # PLANNER
            # ---------------------------------------------------------

            vlog_subsection(
                "2. PLANNER DECISION"
            )

            plan = (
                self._planner.plan_next_step(
                    information_need=information_need,
                    context_summary=context,
                    executed_steps=logs,
                    has_reusable_evidence=bool(
                        reusable
                    ),
                )
            )

            vlog_kv(
                "action",
                plan.action_type.value,
            )

            vlog_kv(
                "tool",
                plan.tool_name,
            )

            vlog_kv(
                "more_steps",
                plan.more_steps,
            )

            vlog_text(
                "rationale",
                plan.rationale,
                max_chars=2000,
            )

            logs.append(
                f"Step {step_number}: "
                f"planner={plan.action_type.value} "
                f"tool={plan.tool_name}."
            )

            evidence: list[Evidence] = []

            # =========================================================
            # REUSE_EVIDENCE
            # =========================================================

            if (
                plan.action_type
                == ActionType.REUSE_EVIDENCE
            ):

                vlog_subsection(
                    "3. REUSE EVIDENCE"
                )

                if not reusable:

                    logs.append(
                        f"Step {step_number}: "
                        f"REUSE_EVIDENCE demandé mais "
                        f"aucune preuve réutilisable."
                    )

                    vlog(
                        "REUSE_EVIDENCE requested "
                        "but no reusable evidence."
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

                vlog_evidence_list(
                    evidence,
                    title="REUSED EVIDENCE",
                )

            # =========================================================
            # SEARCH_DATA
            # =========================================================

            elif (
                plan.action_type
                == ActionType.SEARCH_DATA
            ):

                vlog_subsection(
                    "3. SEARCH DATA"
                )

                vlog(
                    "[Action] Retrieval lancé pour "
                    "l'information need."
                )

                evidence = (
                    self._retrieval.retrieve(
                        information_need
                    )
                )

                vlog_kv(
                    "results",
                    len(evidence),
                )

                vlog_evidence_list(
                    evidence,
                    title="SEARCH RESULTS",
                )

            # =========================================================
            # USE_TOOL
            # =========================================================

            elif (
                plan.action_type
                == ActionType.USE_TOOL
            ):

                vlog_subsection(
                    "3. USE TOOL"
                )

                input_data = (
                    collected
                    if collected
                    else None
                )

                vlog_kv(
                    "tool",
                    plan.tool_name,
                )

                vlog_kv(
                    "input_evidence_count",
                    len(input_data)
                    if input_data
                    else 0,
                )

                if input_data:

                    vlog_evidence_list(
                        input_data,
                        title="TOOL INPUT EVIDENCE",
                    )

                else:

                    vlog(
                        "[Tool] Aucun evidence fourni au tool."
                    )

                evidence = (
                    self._execute_tool(
                        tool_name=plan.tool_name,
                        information_need=information_need,
                        input_data=input_data,
                    )
                )

                vlog_kv(
                    "tool_output_evidence_count",
                    len(evidence),
                )

                vlog_evidence_list(
                    evidence,
                    title="TOOL OUTPUT AS EVIDENCE",
                )

            # =========================================================
            # SEARCH_DATA_AND_USE_TOOL
            # =========================================================

            elif (
                plan.action_type
                == ActionType.SEARCH_DATA_AND_USE_TOOL
            ):

                vlog_subsection(
                    "3. SEARCH DATA + USE TOOL"
                )

                searched_evidence = (
                    self._retrieval.retrieve(
                        information_need
                    )
                )

                vlog_kv(
                    "searched_results",
                    len(searched_evidence),
                )

                vlog_evidence_list(
                    searched_evidence,
                    title="SEARCHED EVIDENCE",
                )

                tool_input = (
                    self._merge_unique_evidence(
                        collected,
                        searched_evidence,
                    )
                )

                vlog_kv(
                    "combined_tool_input_count",
                    len(tool_input),
                )

                vlog_evidence_list(
                    tool_input,
                    title="COMBINED TOOL INPUT",
                )

                evidence = (
                    self._execute_tool(
                        tool_name=plan.tool_name,
                        information_need=information_need,
                        input_data=tool_input,
                    )
                )

                vlog_kv(
                    "tool_output_evidence_count",
                    len(evidence),
                )

                vlog_evidence_list(
                    evidence,
                    title="TOOL OUTPUT AS EVIDENCE",
                )

            # =========================================================
            # UNKNOWN ACTION
            # =========================================================

            else:

                logs.append(
                    f"Step {step_number}: "
                    f"Action inconnue."
                )

                vlog(
                    "Unknown action."
                )

                break

            # =========================================================
            # NO RESULT
            # =========================================================

            if not evidence:

                logs.append(
                    f"Step {step_number}: "
                    f"aucune preuve récupérée."
                )

                vlog(
                    "No evidence returned."
                )

                break

            # =========================================================
            # NEW EVIDENCE
            # =========================================================

            vlog_subsection(
                "4. NEW EVIDENCE DETECTION"
            )

            new_evidence = (
                self._only_new_evidence(
                    evidence,
                    collected,
                )
            )

            vlog_kv(
                "returned",
                len(evidence),
            )

            vlog_kv(
                "new",
                len(new_evidence),
            )

            if new_evidence:

                vlog_evidence_list(
                    new_evidence,
                    title="NEW EVIDENCE",
                )

            if not new_evidence:

                logs.append(
                    f"Step {step_number}: "
                    f"aucune nouvelle preuve."
                )

                vlog(
                    "No new evidence -> stop."
                )

                break

            # =========================================================
            # RELEVANCE CHECK
            # =========================================================

            vlog_subsection(
                "5. EVIDENCE RELEVANCE CHECK"
            )

            vlog(
                "[Action] Evaluation des nouvelles preuves..."
            )

            vlog_kv(
                "information_need",
                information_need,
            )

            vlog_evidence_list(
                new_evidence,
                title="EVIDENCE SENT TO EVALUATOR",
            )

            is_relevant = (
                self._evaluator.is_relevant(
                    information_need,
                    new_evidence,
                )
            )

            vlog_kv(
                "decision",
                "ACCEPT"
                if is_relevant
                else "REJECT",
            )

            if not is_relevant:

                logs.append(
                    f"Step {step_number}: "
                    f"preuves non pertinentes."
                )

                vlog(
                    "❌ Evidence rejected."
                )

                break

            logs.append(
                f"Step {step_number}: "
                f"preuves pertinentes."
            )

            vlog(
                "✅ Evidence accepted."
            )

            # =========================================================
            # ADD EVIDENCE
            # =========================================================

            vlog_subsection(
                "6. ADD EVIDENCE"
            )

            collected.extend(
                new_evidence
            )

            self._evidence_pool.add(
                new_evidence
            )

            vlog_kv(
                "added",
                len(new_evidence),
            )

            vlog_kv(
                "collected_total",
                len(collected),
            )

            context = self._summarize(
                collected
            )

            # =========================================================
            # SUFFICIENCY
            # =========================================================

            vlog_subsection(
                "7. SUFFICIENCY CHECK"
            )

            vlog(
                "[Action] Vérification de la suffisance..."
            )

            sufficient, confidence = (
                self._evaluator.is_sufficient(
                    information_need,
                    collected,
                )
            )

            vlog_kv(
                "sufficient",
                sufficient,
            )

            vlog_kv(
                "confidence",
                f"{confidence:.2f}",
            )

            if sufficient:

                logs.append(
                    f"Step {step_number}: "
                    f"preuves suffisantes."
                )

                vlog(
                    "✅ Evidence sufficient -> "
                    "Action Engine complete."
                )

                return ActionExecutionResult(
                    evidence=collected,
                    step_count=step_number,
                    step_log=logs,
                )

            logs.append(
                f"Step {step_number}: "
                f"preuves insuffisantes "
                f"(confidence={confidence:.2f})."
            )

            # =========================================================
            # NO MORE STEPS
            # =========================================================

            if not plan.more_steps:

                vlog(
                    "Planner says more_steps=False."
                )

                logs.append(
                    f"Step {step_number}: "
                    f"planner indique qu'aucune étape "
                    f"supplémentaire n'est nécessaire."
                )

                return ActionExecutionResult(
                    evidence=collected,
                    step_count=step_number,
                    step_log=logs,
                )

            # =========================================================
            # PROGRESS DETECTION
            # =========================================================

            vlog_subsection(
                "8. PROGRESS DETECTION"
            )

            current_keys = {
                self._evidence_key(e)
                for e in new_evidence
            }

            vlog_kv(
                "new_evidence_keys",
                len(current_keys),
            )

            vlog_kv(
                "previous_keys",
                len(previous_keys),
            )

            if current_keys.issubset(
                previous_keys
            ):

                logs.append(
                    f"Step {step_number}: "
                    f"aucun progrès détecté."
                )

                vlog(
                    "No progress detected -> stop."
                )

                break

            previous_keys.update(
                current_keys
            )

            vlog(
                "Progress detected -> "
                "continuing action loop."
            )

        # =================================================================
        # ACTION ENGINE TERMINATION
        # =================================================================

        vlog_section(
            "[EVIDENCE ACTION ENGINE] END"
        )

        vlog_kv(
            "total_steps",
            executed_step_count,
        )

        vlog_kv(
            "collected_evidence",
            len(collected),
        )

        vlog(
            "Action Engine terminé."
        )

        return ActionExecutionResult(
            evidence=collected,
            step_count=executed_step_count,
            step_log=logs,
        )

    # =================================================================
    # FIND REUSABLE EVIDENCE
    # =================================================================

    def _find_reusable_evidence(
        self,
        information_need: str,
    ) -> list[Evidence]:

        vlog(
            "[Reusable Evidence] "
            "Recherche dans l'evidence pool..."
        )

        query_embedding = (
            self._retrieval.embed_query(
                information_need
            )
        )

        results = (
            self._evidence_pool.find_relevant(
                query_embedding,
                top_k=self._settings.reuse_top_k,
                min_score=(
                    self._settings
                    .reuse_similarity_threshold
                ),
            )
        )

        vlog_kv(
            "reusable_results",
            len(results),
        )

        return results

    # =================================================================
    # EXECUTE TOOL
    # =================================================================

    def _execute_tool(
        self,
        tool_name: str | None,
        information_need: str,
        input_data: list[Evidence] | None,
    ) -> list[Evidence]:

        vlog_subsection(
            "[TOOL] EXECUTION"
        )

        vlog_kv(
            "tool",
            tool_name,
        )

        vlog_kv(
            "information_need",
            information_need,
        )

        vlog_kv(
            "input_count",
            len(input_data)
            if input_data
            else 0,
        )

        if not tool_name:

            vlog(
                "❌ No tool specified."
            )

            return []

        tool = (
            self._tool_registry.get_tool(
                tool_name
            )
        )

        if tool is None:

            vlog(
                f"❌ Tool not found: {tool_name}"
            )

            return []

        vlog_kv(
            "tool_class",
            type(tool).__name__,
        )

        # -------------------------------------------------------------
        # INPUT EVIDENCE
        # -------------------------------------------------------------

        if input_data:

            vlog_evidence_list(
                input_data,
                title="TOOL INPUT",
            )

        else:

            vlog(
                "[TOOL] Input: None"
            )

        # -------------------------------------------------------------
        # STRUCTURED TOOL INPUT
        #
        # Calculator:
        #
        # Evidence
        #     ↓
        # CalculationIntentExtractorService
        #     ↓
        # {
        #     "operation": "subtract",
        #     "values": [1250, 1180]
        # }
        #     ↓
        # CalculatorTool
        #
        # CalculatorTool does NOT call the LLM.
        # -------------------------------------------------------------

        tool_input: dict[str, Any] | None = None

        if tool_name == "calculator":

            vlog_subsection(
                "[TOOL] CALCULATION INTENT EXTRACTION"
            )

            if not input_data:

                vlog(
                    "❌ Calculator requires evidence "
                    "to extract the calculation intent."
                )

                return []

            vlog(
                "[Calculator] Calling "
                "CalculationIntentExtractorService..."
            )

            try:

                tool_input = (
                    self._calculation_intent_extractor.extract(
                        information_need=information_need,
                        evidence=input_data,
                    )
                )

            except Exception as exc:

                vlog(
                    f"❌ Calculation intent extraction failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                return []

            vlog_kv(
                "calculation_operation",
                tool_input.get("operation"),
            )

            vlog_kv(
                "calculation_values",
                tool_input.get("values"),
            )

            vlog(
                "[Calculator] Structured calculation "
                "intent successfully extracted."
            )

        # -------------------------------------------------------------
        # EXECUTE TOOL
        # -------------------------------------------------------------

        vlog_subsection(
            "[TOOL] EXECUTION"
        )

        vlog(
            "[TOOL] Calling tool.execute(...)"
        )

        try:

            result = tool.execute(
                information_need=information_need,
                input_data=input_data,
                tool_input=tool_input,
            )

        except Exception as exc:

            vlog(
                f"❌ Tool error: "
                f"{type(exc).__name__}: {exc}"
            )

            return []

        vlog(
            "[TOOL] tool.execute(...) terminé."
        )

        vlog_kv(
            "result_type",
            type(result).__name__,
        )

        # =============================================================
        # TOOL RESULT
        # =============================================================

        if isinstance(
            result,
            ToolResult,
        ):

            vlog_subsection(
                "[TOOL] TOOL RESULT"
            )

            vlog_kv(
                "success",
                result.success,
            )

            if not result.success:

                vlog(
                    f"❌ Tool failure: "
                    f"{result.error}"
                )

                return []

            output_text = (
                result.output_text
                or ""
            ).strip()

            vlog_text(
                "raw_output",
                output_text,
                max_chars=3000,
            )

            if not output_text:

                vlog(
                    "[TOOL] Empty output -> "
                    "no evidence created."
                )

                return []

            if result.data:

                vlog_kv(
                    "tool_result_data",
                    result.data,
                )

            # ---------------------------------------------------------
            # CONVERT TOOL OUTPUT TO EVIDENCE
            # ---------------------------------------------------------

            evidence = Evidence(
                text=output_text,
                score=1.0,
                source_metadata={
                    "tool_name": tool_name,
                },
                retrieval_method=(
                    RetrievalMethod.TOOL
                ),
                source_reliability=1.0,
            )

            vlog(
                "[TOOL] Tool output converted "
                "into Evidence."
            )

            vlog_evidence(
                evidence,
                max_chars=3000,
            )

            vlog_kv(
                "evidence_method",
                evidence.retrieval_method,
            )

            vlog_kv(
                "evidence_score",
                evidence.score,
            )

            vlog_kv(
                "evidence_reliability",
                evidence.source_reliability,
            )

            return [evidence]

        # =============================================================
        # LIST[Evidence]
        # =============================================================

        if isinstance(
            result,
            list,
        ):

            vlog_subsection(
                "[TOOL] LIST RESULT"
            )

            vlog_kv(
                "raw_items",
                len(result),
            )

            normalized: list[Evidence] = []

            for index, item in enumerate(
                result,
                start=1,
            ):

                vlog(
                    f"[TOOL] Normalizing result #{index} "
                    f"type={type(item).__name__}"
                )

                if isinstance(
                    item,
                    Evidence,
                ):

                    normalized.append(
                        item
                    )

                    continue

                normalized.append(
                    Evidence(
                        text=str(item),
                        score=1.0,
                        source_metadata={
                            "tool_name": tool_name,
                        },
                        retrieval_method=(
                            RetrievalMethod.TOOL
                        ),
                        source_reliability=1.0,
                    )
                )

            vlog_evidence_list(
                normalized,
                title="NORMALIZED TOOL RESULTS",
            )

            return normalized

        # =============================================================
        # GENERIC RESULT
        # =============================================================

        if result is None:

            vlog(
                "[TOOL] Result is None -> "
                "no evidence."
            )

            return []

        vlog(
            "[TOOL] Generic result converted "
            "into Evidence."
        )

        evidence = Evidence(
            text=str(result),
            score=1.0,
            source_metadata={
                "tool_name": tool_name,
            },
            retrieval_method=(
                RetrievalMethod.TOOL
            ),
            source_reliability=1.0,
        )

        vlog_evidence(
            evidence,
            max_chars=3000,
        )

        return [evidence]

    # =================================================================
    # MERGE UNIQUE EVIDENCE
    # =================================================================

    @staticmethod
    def _merge_unique_evidence(
        first: list[Evidence],
        second: list[Evidence],
    ) -> list[Evidence]:

        result: list[Evidence] = []

        seen: set[str] = set()

        for evidence in [
            *first,
            *second,
        ]:

            key = (
                EvidenceActionEngine
                ._evidence_key(evidence)
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(evidence)

        return result

    # =================================================================
    # ONLY NEW EVIDENCE
    # =================================================================

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

            key = (
                EvidenceActionEngine
                ._evidence_key(item)
            )

            if key in existing_keys:
                continue

            existing_keys.add(key)
            result.append(item)

        return result

    # =================================================================
    # EVIDENCE KEY
    # =================================================================

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

    # =================================================================
    # SUMMARY
    # =================================================================

    @staticmethod
    def _summarize(
        evidence: list[Evidence],
    ) -> str:

        return "\n".join(
            f"- {e.text[:1500]}"
            for e in evidence[-8:]
        )