"""Use case principal : implémente fidèlement le diagramme UML
"Adaptive Evidence-Driven Tree Retrieval".

Ce use case est l'orchestrateur (chef d'orchestre) : il ne contient aucune
logique d'infrastructure, uniquement l'enchaînement des services du domaine
et de l'application, via leurs ports.
"""
from __future__ import annotations

from config.settings import TreeSearchSettings
from domain.entities import Evidence, FinalAnswer, Hypothesis
from domain.enums import BranchStatus
from domain.tree import Branch, EvidenceTree

from application.services.branch_exploration_service import BranchExplorationService
from application.services.evidence_evaluator_service import EvidenceEvaluatorService
from application.services.hypothesis_service import HypothesisService
from application.services.query_generation_service import QueryGenerationService
from application.services.retrieval_service import RetrievalService
from application.services.synthesis_service import SynthesisService


class AnswerQuestionUseCase:
    def __init__(
        self,
        query_generation_service: QueryGenerationService,
        retrieval_service: RetrievalService,
        evidence_evaluator: EvidenceEvaluatorService,
        branch_exploration_service: BranchExplorationService,
        hypothesis_service: HypothesisService,
        synthesis_service: SynthesisService,
        settings: TreeSearchSettings,
    ) -> None:
        self._query_gen = query_generation_service
        self._retrieval = retrieval_service
        self._evaluator = evidence_evaluator
        self._branch_explorer = branch_exploration_service
        self._hypotheses_service = hypothesis_service
        self._synthesis = synthesis_service
        self._settings = settings

    def execute(self, user_question: str) -> FinalAnswer:
        # ------------------------------------------------------------
        # 1) Génération de la question de recherche initiale Q0
        # ------------------------------------------------------------
        q0 = self._query_gen.generate_initial_question(user_question)
        e0 = self._retrieval.retrieve(q0)

        sufficient, confidence = self._evaluator.is_sufficient(user_question, e0)
        if sufficient:
            root = Branch(question=q0, parent_id=None, depth=0)
            root.evidence = e0
            root.status = BranchStatus.SUPPORTED
            root.conclusion = "Réponse directe : les preuves initiales suffisent."
            return self._synthesis.synthesize(user_question, [root], hypotheses=[])

        # ------------------------------------------------------------
        # 2) Création de l'arbre + branches candidates initiales
        # ------------------------------------------------------------
        tree = EvidenceTree(root_question=q0)
        tree.root.evidence = e0
        tree.register_evidence(e0)

        initial_questions = self._branch_explorer.generate_initial_branch_questions(user_question, e0)
        for question in initial_questions:
            tree.add_branch(Branch(question=question, parent_id=tree.root.id, depth=1))

        hypotheses: list[Hypothesis] = []
        iterations = 0

        # ------------------------------------------------------------
        # 3) Boucle d'exploration adaptative des branches
        # ------------------------------------------------------------
        while True:
            iterations += 1
            branch = tree.select_next_branch()
            if branch is None or iterations > self._settings.max_iterations:
                break

            self._explore_branch(branch, tree)

            # --- Mise à jour globale ---
            tree.register_evidence(branch.evidence)
            if branch.conclusion:
                tree.discovered_facts.append(branch.conclusion)

            new_evidence = branch.evidence if branch.status != BranchStatus.PRUNED_IRRELEVANT else []
            hypotheses = self._hypotheses_service.update_hypotheses(user_question, hypotheses, new_evidence)
            _contradictions = self._hypotheses_service.check_contradictions(hypotheses)

            for b in tree.unexplored_branches():
                self._branch_explorer.recalculate_priority(b)

            if self._is_globally_sufficient(user_question, tree):
                break

        # ------------------------------------------------------------
        # 4) Synthèse finale
        # ------------------------------------------------------------
        strongest = tree.strongest_branches(limit=self._settings.min_independent_branches_supported + 2)
        if not strongest:
            strongest = [tree.root]
        return self._synthesis.synthesize(user_question, strongest, hypotheses)

    # ------------------------------------------------------------------
    # Exploration d'une branche unique
    # ------------------------------------------------------------------
    def _explore_branch(self, branch: Branch, tree: EvidenceTree) -> None:
        search_query = self._query_gen.reformulate_branch_question(branch.question, branch.context_summary)
        evidence: list[Evidence] = self._retrieval.retrieve(search_query)

        if not self._evaluator.is_relevant(branch.question, evidence):
            branch.status = BranchStatus.PRUNED_IRRELEVANT
            return

        branch.add_evidence(evidence)
        branch.evidence_strength = self._evaluator.evidence_strength(branch.evidence)
        gain = self._evaluator.information_gain(evidence, tree.global_evidence_pool)
        branch.information_gain = gain

        if self._evaluator.is_information_gain_too_low(gain):
            branch.status = BranchStatus.PRUNED_LOW_INFO_GAIN
            return

        branch.context_summary = self._build_context_summary(branch)

        sufficient, _confidence = self._evaluator.is_sufficient(branch.question, branch.evidence)
        if sufficient:
            branch.status = BranchStatus.SUPPORTED
            branch.conclusion = self._build_conclusion(branch)
            return

        candidate_questions = self._branch_explorer.generate_child_questions(branch, tree)
        if not candidate_questions:
            branch.status = BranchStatus.PRUNED_NO_CHILDREN
            return

        for question in candidate_questions:
            child = Branch(question=question, parent_id=branch.id, depth=branch.depth + 1)
            tree.add_branch(child)

        # Le nœud parent a fait son travail (a produit des enfants) : il
        # n'est plus une branche "active" à explorer telle quelle.
        branch.status = BranchStatus.SUPPORTED
        branch.conclusion = self._build_conclusion(branch)

    # ------------------------------------------------------------------
    # Condition d'arrêt globale
    # ------------------------------------------------------------------
    def _is_globally_sufficient(self, user_question: str, tree: EvidenceTree) -> bool:
        supported_count = len(tree.supported_branches())
        if supported_count < self._settings.min_independent_branches_supported:
            return False
        sufficient, confidence = self._evaluator.is_sufficient(user_question, tree.global_evidence_pool)
        return sufficient and confidence >= self._settings.global_sufficiency_confidence

    @staticmethod
    def _build_context_summary(branch: Branch) -> str:
        excerpts = "\n".join(f"- {e.text[:200]}" for e in branch.evidence[-8:])
        return f"Question: {branch.question}\nPreuves clés:\n{excerpts}"

    @staticmethod
    def _build_conclusion(branch: Branch) -> str:
        top_evidence = branch.evidence[0].text[:200] if branch.evidence else ""
        return f"Branche '{branch.question}' étayée par {len(branch.evidence)} preuve(s). Ex: {top_evidence}"
