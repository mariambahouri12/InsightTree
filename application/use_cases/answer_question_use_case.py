from __future__ import annotations

from application.services.branch_exploration_service import BranchExplorationService
from application.services.cross_branch_reasoning_service import CrossBranchReasoningService
from application.services.evidence_action_engine import EvidenceActionEngine
from application.services.evidence_evaluator_service import EvidenceEvaluatorService
from application.services.hypothesis_service import HypothesisService
from application.services.information_need_service import InformationNeedService
from application.services.synthesis_service import SynthesisService
from config.settings import TreeSearchSettings
from domain.entities import FinalAnswer, Hypothesis
from domain.enums import BranchStatus
from domain.tree import Branch, EvidenceTree
from infrastructure.logging.verbose_logger import vlog


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
        self._information_need = information_need_service
        self._action_engine = action_engine
        self._evaluator = evaluator
        self._branch_exploration = branch_exploration_service
        self._hypothesis_service = hypothesis_service
        self._cross_branch = cross_branch_service
        self._synthesis = synthesis_service
        self._settings = settings

    def execute(self, user_question: str) -> FinalAnswer:

        # ============================================================
        # STEP 1 — DIRECT SEARCH
        # ============================================================
        # IMPORTANT:
        # We use the user's original question exactly as written.
        # No Q0 generation, no rewriting, no planner.
        # ============================================================

        user_question = user_question.strip()

        vlog(f"[Direct Search] {user_question}")

        direct_result = self._action_engine.search_direct(user_question)

        direct_evidence = direct_result.evidence

        vlog(
            f"[Direct Search] {len(direct_evidence)} preuve(s) récupérée(s)"
        )

        # ============================================================
        # STEP 2 — CHECK IF THE DIRECT EVIDENCE IS SUFFICIENT
        # ============================================================

        sufficient_direct, confidence_direct = (
            self._evaluator.is_sufficient(
                user_question,
                direct_evidence,
            )
        )

        vlog(
            f"[Direct Sufficiency] "
            f"sufficient={sufficient_direct} "
            f"confidence={confidence_direct:.2f}"
        )

        # ============================================================
        # STEP 3 — SIMPLE QUESTION
        # ============================================================
        # If the original question can already be answered from the
        # retrieved documents, stop here.
        #
        # No Q0.
        # No hypotheses.
        # No tree.
        # No branches.
        # ============================================================

        if sufficient_direct:
            vlog(
                "[Flow] Réponse trouvée directement -> "
                "pas de Q0, pas d'exploration."
            )

            return self._synthesis.synthesize(
                user_question,
                [],
                [],
                direct_evidence,
            )

        # ============================================================
        # STEP 4 — DIRECT SEARCH WAS NOT SUFFICIENT
        # ============================================================
        # Now, and only now, we activate the adaptive reasoning system.
        # ============================================================

        vlog(
            "[Flow] Preuves directes insuffisantes -> génération de Q0."
        )

        q0 = self._information_need.generate_initial_need(
            user_question
        )

        vlog(f"[Q0] {q0}")

        # ============================================================
        # STEP 5 — SEARCH Q0 USING THE ACTION ENGINE
        # ============================================================

        result0 = self._action_engine.execute(q0)

        sufficient0, confidence0 = self._evaluator.is_sufficient(
            user_question,
            result0.evidence,
        )

        vlog(
            f"[Sufficiency Q0] "
            f"sufficient={sufficient0} "
            f"confidence={confidence0:.2f}"
        )

        # ============================================================
        # STEP 6 — Q0 ALONE IS ENOUGH
        # ============================================================

        if sufficient0:
            vlog(
                "[Flow] Preuves suffisantes après Q0 -> synthèse directe."
            )

            return self._synthesis.synthesize(
                user_question,
                [],
                [],
                result0.evidence,
            )

        # ============================================================
        # STEP 7 — CREATE THE EVIDENCE TREE
        # ============================================================

        tree = EvidenceTree(user_question)

        # The evidence found during the direct search is also useful.
        # We keep it as initial global context.
        if direct_evidence:
            tree.root.add_evidence(direct_evidence)
            tree.register_evidence(direct_evidence)

        # Add Q0 evidence as well.
        if result0.evidence:
            tree.root.add_evidence(result0.evidence)
            tree.register_evidence(result0.evidence)

        tree.root.evidence_strength = self._evaluator.evidence_strength(
            tree.root.evidence
        )

        # ============================================================
        # STEP 8 — INITIAL HYPOTHESES
        # ============================================================

        initial_evidence = tree.root.evidence

        hypotheses: list[Hypothesis] = (
            self._hypothesis_service.generate_initial_hypotheses(
                user_question,
                initial_evidence,
            )
        )

        vlog(
            f"[Hypothèses initiales] "
            f"{[h.statement for h in hypotheses]}"
        )

        # ============================================================
        # STEP 9 — GENERATE INITIAL BRANCH QUESTIONS
        # ============================================================

        questions = (
            self._branch_exploration.generate_initial_branch_questions(
                user_question,
                initial_evidence,
            )
        )

        questions = questions[
            : self._settings.max_initial_branches
        ]

        vlog(f"[Questions initiales] {questions}")

        self._create_branches(
            tree,
            questions,
            tree.root.id,
            1,
        )

        # ============================================================
        # STEP 10 — ADAPTIVE EXPLORATION
        # ============================================================

        iterations_left = self._run_adaptive_exploration(
            tree,
            hypotheses,
            self._settings.max_iterations,
        )

        # ============================================================
        # STEP 11 — GLOBAL SUFFICIENCY CHECK
        # ============================================================

        sufficient, confidence = self._evaluator.is_sufficient(
            user_question,
            tree.global_evidence_pool,
        )

        vlog(
            f"[Sufficiency après exploration] "
            f"sufficient={sufficient} "
            f"confidence={confidence:.2f}"
        )

        # ============================================================
        # STEP 12 — CROSS-BRANCH EXPANSION
        # ============================================================

        expansion_round = 0

        while (
            not sufficient
            and confidence < self._settings.global_sufficiency_confidence
            and expansion_round < self._settings.max_expansion_rounds
            and iterations_left > 0
        ):
            gaps = self._cross_branch.identify_information_gaps(
                user_question,
                tree,
                hypotheses,
            )

            vlog(
                f"[Cross-branch] Lacunes identifiées: {gaps}"
            )

            new_questions = self._cross_branch.generate_new_questions(
                user_question,
                tree,
                gaps,
            )

            vlog(
                f"[Cross-branch] Nouvelles questions: {new_questions}"
            )

            if not new_questions:
                break

            self._create_branches(
                tree,
                new_questions,
                tree.root.id,
                1,
                max_count=self._settings.max_initial_branches,
            )

            expansion_round += 1

            iterations_left = self._run_adaptive_exploration(
                tree,
                hypotheses,
                iterations_left,
            )

            sufficient, confidence = self._evaluator.is_sufficient(
                user_question,
                tree.global_evidence_pool,
            )

            vlog(
                f"[Sufficiency round {expansion_round}] "
                f"sufficient={sufficient} "
                f"confidence={confidence:.2f}"
            )

        # ============================================================
        # STEP 13 — FINAL SYNTHESIS
        # ============================================================

        strongest = tree.strongest_branches(
            max(
                self._settings.max_children_per_branch * 2,
                4,
            )
        )

        vlog(
            f"[Synthèse] Branches retenues: "
            f"{[b.question for b in strongest]}"
        )

        vlog(
            f"[Faits découverts] {tree.discovered_facts}"
        )

        return self._synthesis.synthesize(
            user_question,
            strongest,
            hypotheses,
            tree.global_evidence_pool,
        )

    # ================================================================
    # ADAPTIVE EXPLORATION
    # ================================================================

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
            branch = tree.select_next_branch()

            if branch is None:
                break

            vlog(
                f"[Branche sélectionnée] "
                f"'{branch.question}' "
                f"(priority={branch.priority:.2f})"
            )

            iterations_left -= 1

            self._explore_branch(
                tree,
                branch,
                hypotheses,
            )

        return iterations_left

    # ================================================================
    # EXPLORE ONE BRANCH
    # ================================================================

    def _explore_branch(
        self,
        tree: EvidenceTree,
        branch: Branch,
        hypotheses: list[Hypothesis],
    ) -> None:

        information_need = (
            self._information_need.generate_branch_need(
                branch.question,
                branch.context_summary,
            )
        )

        result = self._action_engine.execute(
            information_need
        )

        if not result.evidence:
            branch.status = BranchStatus.INVALID
            branch.failure_reason = (
                "Aucune preuve récupérée."
            )

            vlog(
                f"  [Branche] '{branch.question}' "
                f"-> INVALID ({branch.failure_reason})"
            )

            return

        if not self._evaluator.is_relevant(
            information_need,
            result.evidence,
        ):
            branch.status = BranchStatus.INVALID
            branch.failure_reason = (
                "Résultat non pertinent."
            )

            vlog(
                f"  [Branche] '{branch.question}' "
                f"-> INVALID ({branch.failure_reason})"
            )

            return

        # ------------------------------------------------------------
        # Information gain
        # ------------------------------------------------------------

        previous_pool = list(
            tree.global_evidence_pool
        )

        gain = self._evaluator.information_gain(
            result.evidence,
            previous_pool,
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

        branch.context_summary = self._summarize(
            branch.evidence
        )

        # ------------------------------------------------------------
        # Branch sufficiency
        # ------------------------------------------------------------

        sufficient, _ = self._evaluator.is_sufficient(
            branch.question,
            branch.evidence,
        )

        self._hypothesis_service.update_confidence(
            hypotheses,
            branch,
        )

        branch.relevance = min(
            1.0,
            max(
                [e.score for e in result.evidence]
                or [0.0]
            ),
        )

        branch.priority = (
            self._branch_exploration.priority(
                branch.relevance,
                branch.information_gain,
            )
        )

        # ------------------------------------------------------------
        # Resolved
        # ------------------------------------------------------------

        if sufficient:
            branch.status = BranchStatus.RESOLVED

            branch.conclusion = (
                branch.context_summary
            )

            tree.discovered_facts.append(
                f"{branch.question} -> "
                f"{branch.conclusion}"
            )

            vlog(
                f"  [Branche] '{branch.question}' "
                f"-> RESOLVED "
                f"(gain={gain:.2f})"
            )

            return

        # ------------------------------------------------------------
        # Low information gain
        # ------------------------------------------------------------

        if self._evaluator.is_information_gain_too_low(
            gain
        ):
            branch.status = BranchStatus.LOW_VALUE

            branch.failure_reason = (
                "Gain d'information insuffisant."
            )

            vlog(
                f"  [Branche] '{branch.question}' "
                f"-> LOW_VALUE "
                f"(gain={gain:.2f})"
            )

            return

        # ------------------------------------------------------------
        # Generate child questions
        # ------------------------------------------------------------

        child_questions = (
            self._branch_exploration.generate_child_questions(
                branch,
                tree,
            )
        )

        vlog(
            f"  [Branche] '{branch.question}' "
            f"-> questions enfants: "
            f"{child_questions}"
        )

        if child_questions:
            self._create_branches(
                tree,
                child_questions,
                branch.id,
                branch.depth + 1,
                max_count=self._settings.max_children_per_branch,
            )

            branch.status = BranchStatus.EXPANDED

        else:
            branch.status = BranchStatus.TERMINATED

            branch.failure_reason = (
                "Aucune question enfant prometteuse."
            )

        self._recalculate_priorities(tree)

    # ================================================================
    # CREATE BRANCHES
    # ================================================================

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

            normalized = question.strip().lower()

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

    # ================================================================
    # PRIORITIES
    # ================================================================

    @staticmethod
    def _recalculate_priorities(
        tree: EvidenceTree,
    ) -> None:

        for branch in tree.unexplored_branches():
            branch.priority = (
                0.6 * branch.relevance
                + 0.4 * branch.information_gain
            )

    # ================================================================
    # EVIDENCE SUMMARY
    # ================================================================

    @staticmethod
    def _summarize(evidence) -> str:
        return "\n".join(
            f"- {e.text[:220]}"
            for e in evidence[-8:]
        )