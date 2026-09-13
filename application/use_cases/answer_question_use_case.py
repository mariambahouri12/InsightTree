from __future__ import annotations

from application.services.branch_exploration_service import (
    BranchExplorationService,
)
from application.services.cross_branch_reasoning_service import (
    CrossBranchReasoningService,
)
from application.services.evidence_action_engine import (
    EvidenceActionEngine,
)
from application.services.evidence_evaluator_service import (
    EvidenceEvaluatorService,
)
from application.services.hypothesis_service import (
    HypothesisService,
)
from application.services.information_need_service import (
    InformationNeedService,
)
from application.services.synthesis_service import (
    SynthesisService,
)
from config.settings import TreeSearchSettings
from domain.entities import FinalAnswer, Hypothesis
from domain.enums import BranchStatus
from domain.tree import Branch, EvidenceTree
from infrastructure.logging.verbose_logger import (
    reset_trace_timer,
    vlog,
    vlog_evidence_list,
    vlog_kv,
    vlog_section,
    vlog_subsection,
)


class AnswerQuestionUseCase:
    def __init__(
        self,
        information_need_service: InformationNeedService,
        action_engine: EvidenceActionEngine,
        evaluator: EvidenceEvaluatorService,
        branch_exploration_service: BranchExplorationService,
        hypothesis_service: HypothesisService,
        cross_branch_service: CrossBranchReasoningService,
        synthesis_service: SynthesisService,
        settings: TreeSearchSettings,
    ) -> None:

        self._information_need = (
            information_need_service
        )

        self._action_engine = (
            action_engine
        )

        self._evaluator = evaluator

        self._branch_exploration = (
            branch_exploration_service
        )

        self._hypothesis_service = (
            hypothesis_service
        )

        self._cross_branch = (
            cross_branch_service
        )

        self._synthesis = (
            synthesis_service
        )

        self._settings = settings

    def execute(
        self,
        user_question: str,
    ) -> FinalAnswer:

        reset_trace_timer()

        user_question = (
            user_question.strip()
        )

        vlog_section(
            "INSIGHTTREE — QUERY TRACE"
        )

        vlog_kv(
            "question",
            user_question,
        )

        # ============================================================
        # STEP 1 — DIRECT RETRIEVAL
        # ============================================================

        vlog_section(
            "[STEP 1] DIRECT RETRIEVAL"
        )

        vlog(
            f"Question: {user_question}"
        )

        direct_result = (
            self._action_engine.search_direct(
                user_question
            )
        )

        direct_evidence = (
            direct_result.evidence
        )

        vlog_kv(
            "retrieved",
            len(direct_evidence),
        )

        vlog_evidence_list(
            direct_evidence,
            title="DIRECT EVIDENCE",
        )

        # ============================================================
        # STEP 2 — NO DIRECT EVIDENCE
        # ============================================================

        if not direct_evidence:

            vlog_section(
                "[FLOW] NO DIRECT EVIDENCE"
            )

            vlog(
                "Aucune preuve directe -> "
                "activation de Q0."
            )

            return self._run_adaptive_reasoning(
                user_question=user_question,
                direct_evidence=[],
            )

        # ============================================================
        # STEP 3 — DIRECT SUFFICIENCY
        # ============================================================

        vlog_section(
            "[STEP 2] DIRECT SUFFICIENCY"
        )

        sufficient_direct, confidence_direct = (
            self._evaluator.is_sufficient(
                user_question,
                direct_evidence,
            )
        )

        vlog_kv(
            "sufficient",
            sufficient_direct,
        )

        vlog_kv(
            "confidence",
            f"{confidence_direct:.2f}",
        )

        # ============================================================
        # STEP 4 — ACTION ENGINE
        # ============================================================

        if sufficient_direct:

            vlog_section(
                "[STEP 3] EVIDENCE ACTION ENGINE"
            )

            vlog(
                "Preuves directes suffisantes -> "
                "passage par Evidence Action Engine."
            )

            action_result = (
                self._action_engine.execute(
                    information_need=user_question,
                    initial_evidence=direct_evidence,
                )
            )

            vlog_subsection(
                "ACTION ENGINE RESULT"
            )

            vlog_kv(
                "step_count",
                action_result.step_count,
            )

            vlog_kv(
                "evidence_count",
                len(action_result.evidence),
            )

            vlog_evidence_list(
                action_result.evidence,
                title="COLLECTED EVIDENCE",
            )

            if action_result.step_log:

                vlog_subsection(
                    "ACTION ENGINE STEP LOG"
                )

                for item in (
                    action_result.step_log
                ):
                    vlog(
                        f"  {item}"
                    )

            if action_result.evidence:

                vlog_section(
                    "[STEP 4] FINAL SYNTHESIS"
                )

                return self._synthesis.synthesize(
                    user_question,
                    [],
                    [],
                    action_result.evidence,
                )

            vlog(
                "Action Engine n'a produit "
                "aucune preuve -> activation de Q0."
            )

        # ============================================================
        # STEP 5 — ADAPTIVE REASONING
        # ============================================================

        vlog_section(
            "[FLOW] ADAPTIVE REASONING"
        )

        vlog(
            "Preuves directes insuffisantes -> "
            "activation de Q0."
        )

        return self._run_adaptive_reasoning(
            user_question=user_question,
            direct_evidence=direct_evidence,
        )

    # =================================================================
    # ADAPTIVE REASONING
    # =================================================================

    def _run_adaptive_reasoning(
        self,
        user_question: str,
        direct_evidence,
    ) -> FinalAnswer:

        # ============================================================
        # STEP 1 — GENERATE Q0
        # ============================================================

        vlog_section(
            "[ADAPTIVE] Q0 INFORMATION NEED"
        )

        q0 = (
            self._information_need
            .generate_initial_need(
                user_question
            )
        )

        vlog_kv(
            "Q0",
            q0,
        )

        # ============================================================
        # STEP 2 — EXECUTE Q0
        # ============================================================

        vlog_section(
            "[ADAPTIVE] EXECUTE Q0"
        )

        result0 = (
            self._action_engine.execute(
                information_need=q0,
                initial_evidence=direct_evidence,
            )
        )

        vlog_kv(
            "Q0_step_count",
            result0.step_count,
        )

        vlog_kv(
            "Q0_evidence_count",
            len(result0.evidence),
        )

        vlog_evidence_list(
            result0.evidence,
            title="Q0 EVIDENCE",
        )

        sufficient0, confidence0 = (
            self._evaluator.is_sufficient(
                user_question,
                result0.evidence,
            )
        )

        vlog_subsection(
            "[ADAPTIVE] Q0 SUFFICIENCY"
        )

        vlog_kv(
            "sufficient",
            sufficient0,
        )

        vlog_kv(
            "confidence",
            f"{confidence0:.2f}",
        )

        # ============================================================
        # STEP 3 — Q0 ALONE IS ENOUGH
        # ============================================================

        if sufficient0:

            vlog_section(
                "[FLOW] Q0 SUFFICIENT"
            )

            vlog(
                "Q0 a fourni suffisamment de "
                "preuves -> synthèse directe."
            )

            return self._synthesis.synthesize(
                user_question,
                [],
                [],
                result0.evidence,
            )

        # ============================================================
        # STEP 4 — CREATE TREE
        # ============================================================

        vlog_section(
            "[ADAPTIVE] CREATE EVIDENCE TREE"
        )

        tree = EvidenceTree(
            user_question
        )

        if direct_evidence:

            tree.root.add_evidence(
                direct_evidence
            )

            tree.register_evidence(
                direct_evidence
            )

        if result0.evidence:

            tree.root.add_evidence(
                result0.evidence
            )

            tree.register_evidence(
                result0.evidence
            )

        tree.root.evidence_strength = (
            self._evaluator.evidence_strength(
                tree.root.evidence
            )
        )

        vlog_kv(
            "root_evidence_count",
            len(tree.root.evidence),
        )

        vlog_kv(
            "root_evidence_strength",
            f"{tree.root.evidence_strength:.2f}",
        )

        # ============================================================
        # STEP 5 — INITIAL HYPOTHESES
        # ============================================================

        vlog_section(
            "[ADAPTIVE] INITIAL HYPOTHESES"
        )

        initial_evidence = (
            tree.root.evidence
        )

        hypotheses: list[Hypothesis] = (
            self._hypothesis_service
            .generate_initial_hypotheses(
                user_question,
                initial_evidence,
            )
        )

        vlog_kv(
            "hypothesis_count",
            len(hypotheses),
        )

        for index, hypothesis in enumerate(
            hypotheses,
            start=1,
        ):
            vlog(
                f"  H{index}: "
                f"{hypothesis.statement}"
            )

            vlog_kv(
                "confidence",
                f"{hypothesis.confidence:.2f}",
                indent=6,
            )

            vlog_kv(
                "remaining_uncertainty",
                f"{hypothesis.remaining_uncertainty:.2f}",
                indent=6,
            )

        # ============================================================
        # STEP 6 — INITIAL BRANCH QUESTIONS
        # ============================================================

        vlog_section(
            "[ADAPTIVE] INITIAL BRANCH QUESTIONS"
        )

        questions = (
            self._branch_exploration
            .generate_initial_branch_questions(
                user_question,
                initial_evidence,
            )
        )

        questions = questions[
            : self._settings.max_initial_branches
        ]

        vlog_kv(
            "branch_count",
            len(questions),
        )

        for index, question in enumerate(
            questions,
            start=1,
        ):
            vlog(
                f"  Branch {index}: {question}"
            )

        self._create_branches(
            tree,
            questions,
            tree.root.id,
            1,
        )

        # ============================================================
        # STEP 7 — ADAPTIVE EXPLORATION
        # ============================================================

        vlog_section(
            "[ADAPTIVE] TREE EXPLORATION"
        )

        iterations_left = (
            self._run_adaptive_exploration(
                tree,
                hypotheses,
                self._settings.max_iterations,
            )
        )

        vlog_kv(
            "iterations_left",
            iterations_left,
        )

        vlog_kv(
            "branches_total",
            len(tree.branches),
        )

        # ============================================================
        # STEP 8 — GLOBAL SUFFICIENCY
        # ============================================================

        vlog_section(
            "[ADAPTIVE] GLOBAL SUFFICIENCY"
        )

        sufficient, confidence = (
            self._evaluator.is_sufficient(
                user_question,
                tree.global_evidence_pool,
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

        # ============================================================
        # STEP 9 — CROSS-BRANCH EXPANSION
        # ============================================================

        expansion_round = 0

        while (
            not sufficient
            and confidence
            < self._settings.global_sufficiency_confidence
            and expansion_round
            < self._settings.max_expansion_rounds
            and iterations_left > 0
        ):

            vlog_section(
                f"[CROSS-BRANCH] ROUND "
                f"{expansion_round + 1}"
            )

            gaps = (
                self._cross_branch
                .identify_information_gaps(
                    user_question,
                    tree,
                    hypotheses,
                )
            )

            vlog_kv(
                "information_gaps",
                gaps,
            )

            new_questions = (
                self._cross_branch
                .generate_new_questions(
                    user_question,
                    tree,
                    gaps,
                )
            )

            vlog_kv(
                "new_questions",
                new_questions,
            )

            if not new_questions:

                vlog(
                    "Aucune nouvelle question -> "
                    "arrêt du cross-branch."
                )

                break

            self._create_branches(
                tree,
                new_questions,
                tree.root.id,
                1,
                max_count=(
                    self._settings
                    .max_initial_branches
                ),
            )

            expansion_round += 1

            iterations_left = (
                self._run_adaptive_exploration(
                    tree,
                    hypotheses,
                    iterations_left,
                )
            )

            sufficient, confidence = (
                self._evaluator.is_sufficient(
                    user_question,
                    tree.global_evidence_pool,
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

        # ============================================================
        # STEP 10 — FINAL SYNTHESIS
        # ============================================================

        vlog_section(
            "[FINAL] SYNTHESIS"
        )

        strongest = (
            tree.strongest_branches(
                max(
                    self._settings
                    .max_children_per_branch * 2,
                    4,
                )
            )
        )

        vlog_kv(
            "selected_branches",
            len(strongest),
        )

        for index, branch in enumerate(
            strongest,
            start=1,
        ):
            vlog(
                f"  Branch #{index}: "
                f"{branch.question}"
            )

            vlog_kv(
                "status",
                branch.status.value,
                indent=6,
            )

            vlog_kv(
                "priority",
                f"{branch.priority:.2f}",
                indent=6,
            )

            vlog_kv(
                "information_gain",
                f"{branch.information_gain:.2f}",
                indent=6,
            )

            vlog_kv(
                "evidence_count",
                len(branch.evidence),
                indent=6,
            )

        vlog_kv(
            "discovered_facts",
            tree.discovered_facts,
        )

        vlog_evidence_list(
            tree.global_evidence_pool,
            title="GLOBAL EVIDENCE SENT TO SYNTHESIS",
        )

        answer = (
            self._synthesis.synthesize(
                user_question,
                strongest,
                hypotheses,
                tree.global_evidence_pool,
            )
        )

        vlog_section(
            "[FINAL] ANSWER"
        )

        vlog_kv(
            "confidence",
            f"{answer.confidence:.2f}",
        )

        vlog(
            f"Answer: {answer.text}"
        )

        vlog_section(
            "INSIGHTTREE — END"
        )

        return answer

    # =================================================================
    # ADAPTIVE EXPLORATION
    # =================================================================

    def _run_adaptive_exploration(
        self,
        tree: EvidenceTree,
        hypotheses: list[Hypothesis],
        iterations_left: int,
    ) -> int:

        while (
            tree.has_promising_branches()
            and iterations_left > 0
        ):

            branch = (
                tree.select_next_branch()
            )

            if branch is None:
                break

            vlog_section(
                "[TREE] BRANCH SELECTION"
            )

            vlog_kv(
                "question",
                branch.question,
            )

            vlog_kv(
                "depth",
                branch.depth,
            )

            vlog_kv(
                "priority",
                f"{branch.priority:.2f}",
            )

            vlog_kv(
                "relevance",
                f"{branch.relevance:.2f}",
            )

            vlog_kv(
                "information_gain",
                f"{branch.information_gain:.2f}",
            )

            iterations_left -= 1

            self._explore_branch(
                tree,
                branch,
                hypotheses,
            )

        return iterations_left

    # =================================================================
    # EXPLORE ONE BRANCH
    # =================================================================

    def _explore_branch(
        self,
        tree: EvidenceTree,
        branch: Branch,
        hypotheses: list[Hypothesis],
    ) -> None:

        vlog_section(
            f"[BRANCH] {branch.question}"
        )

        information_need = (
            self._information_need
            .generate_branch_need(
                branch.question,
                branch.context_summary,
            )
        )

        vlog_kv(
            "information_need",
            information_need,
        )

        result = (
            self._action_engine.execute(
                information_need
            )
        )

        vlog_kv(
            "evidence_count",
            len(result.evidence),
        )

        vlog_evidence_list(
            result.evidence,
            title="BRANCH EVIDENCE",
        )

        if not result.evidence:

            branch.status = (
                BranchStatus.INVALID
            )

            branch.failure_reason = (
                "Aucune preuve récupérée."
            )

            vlog(
                f"BRANCH -> INVALID | "
                f"{branch.failure_reason}"
            )

            return

        if not self._evaluator.is_relevant(
            information_need,
            result.evidence,
        ):

            branch.status = (
                BranchStatus.INVALID
            )

            branch.failure_reason = (
                "Résultat non pertinent."
            )

            vlog(
                f"BRANCH -> INVALID | "
                f"{branch.failure_reason}"
            )

            return

        previous_pool = list(
            tree.global_evidence_pool
        )

        gain = (
            self._evaluator.information_gain(
                result.evidence,
                previous_pool,
            )
        )

        vlog_kv(
            "information_gain",
            f"{gain:.4f}",
        )

        branch.add_evidence(
            result.evidence
        )

        tree.register_evidence(
            result.evidence
        )

        branch.information_gain = gain

        branch.evidence_strength = (
            self._evaluator.evidence_strength(
                branch.evidence
            )
        )

        branch.context_summary = (
            self._summarize(
                branch.evidence
            )
        )

        sufficient, _ = (
            self._evaluator.is_sufficient(
                branch.question,
                branch.evidence,
            )
        )

        self._hypothesis_service.update_confidence(
            hypotheses,
            branch,
        )

        branch.relevance = min(
            1.0,
            max(
                [
                    e.score
                    for e in result.evidence
                ]
                or [0.0]
            ),
        )

        branch.priority = (
            self._branch_exploration.priority(
                branch.relevance,
                branch.information_gain,
            )
        )

        vlog_kv(
            "relevance",
            f"{branch.relevance:.2f}",
        )

        vlog_kv(
            "priority",
            f"{branch.priority:.2f}",
        )

        if sufficient:

            branch.status = (
                BranchStatus.RESOLVED
            )

            branch.conclusion = (
                branch.context_summary
            )

            tree.discovered_facts.append(
                f"{branch.question} -> "
                f"{branch.conclusion}"
            )

            vlog(
                "BRANCH -> RESOLVED"
            )

            return

        if self._evaluator.is_information_gain_too_low(
            gain
        ):

            branch.status = (
                BranchStatus.LOW_VALUE
            )

            branch.failure_reason = (
                "Gain d'information insuffisant."
            )

            vlog(
                "BRANCH -> LOW_VALUE"
            )

            return

        child_questions = (
            self._branch_exploration
            .generate_child_questions(
                branch,
                tree,
            )
        )

        vlog_kv(
            "child_questions",
            child_questions,
        )

        if child_questions:

            self._create_branches(
                tree,
                child_questions,
                branch.id,
                branch.depth + 1,
                max_count=(
                    self._settings
                    .max_children_per_branch
                ),
            )

            branch.status = (
                BranchStatus.EXPANDED
            )

        else:

            branch.status = (
                BranchStatus.TERMINATED
            )

            branch.failure_reason = (
                "Aucune question enfant prometteuse."
            )

            vlog(
                "BRANCH -> TERMINATED"
            )

        self._recalculate_priorities(
            tree
        )

    # =================================================================
    # CREATE BRANCHES
    # =================================================================

    @staticmethod
    def _create_branches(
        tree: EvidenceTree,
        questions: list[str],
        parent_id: str,
        depth: int,
        max_count: int | None = None,
    ) -> None:

        limit = (
            max_count
            if max_count is not None
            else len(questions)
        )

        seen = {
            q.strip().lower()
            for q in tree.all_questions()
        }

        added = 0

        for question in questions:

            if added >= limit:
                break

            normalized = (
                question.strip().lower()
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            tree.add_branch(
                Branch(
                    question=question.strip(),
                    parent_id=parent_id,
                    depth=depth,
                )
            )

            seen.add(normalized)
            added += 1

            vlog(
                f"[TREE] Branch created: "
                f"{question.strip()}"
            )

    # =================================================================
    # PRIORITIES
    # =================================================================

    @staticmethod
    def _recalculate_priorities(
        tree: EvidenceTree,
    ) -> None:

        for branch in (
            tree.unexplored_branches()
        ):

            branch.priority = (
                0.6 * branch.relevance
                + 0.4 * branch.information_gain
            )

    # =================================================================
    # SUMMARY
    # =================================================================

    @staticmethod
    def _summarize(
        evidence,
    ) -> str:

        return "\n".join(
            f"- {e.text[:220]}"
            for e in evidence[-8:]
        )