"""
Point d'entrée CLI + composition root.

Usage:
    python main.py session

Exemples:
    > ingest fichier1.pdf fichier2.xlsx
    > ask "Pourquoi les ventes du produit B ont-elles baissé en 2026 ?"
    > quit
"""

from __future__ import annotations

import argparse

from application.services.branch_exploration_service import (
    BranchExplorationService,
)
from application.services.calculation_intent_extractor_service import (
    CalculationIntentExtractorService,
)
from application.services.cross_branch_reasoning_service import (
    CrossBranchReasoningService,
)
from application.services.evidence_action_engine import (
    EvidenceActionEngine,
)
from application.services.evidence_action_planner_service import (
    EvidenceActionPlannerService,
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
from application.services.retrieval_service import (
    RetrievalService,
)
from application.services.synthesis_service import (
    SynthesisService,
)
from application.use_cases.answer_question_use_case import (
    AnswerQuestionUseCase,
)
from application.use_cases.ingest_documents_use_case import (
    IngestDocumentsUseCase,
)

from config.settings import get_settings

from infrastructure.chunking.text_chunker import (
    TextChunker,
)
from infrastructure.embeddings.ollama_embedding_adapter import (
    OllamaEmbeddingAdapter,
)
from infrastructure.lexical.bm25_lexical_index import (
    BM25LexicalIndex,
)
from infrastructure.llm.ollama_llm_adapter import (
    OllamaLLMAdapter,
)
from infrastructure.parsers.parser_factory import (
    DocumentParserFactory,
)
from infrastructure.repositories.in_memory_document_repository import (
    InMemoryDocumentRepository,
)
from infrastructure.repositories.in_memory_evidence_pool import (
    InMemoryEvidencePool,
)
from infrastructure.tools.calculator_tool import (
    CalculatorTool,
)
from infrastructure.tools.structured_data_registry import (
    StructuredDataRegistry,
)
from infrastructure.tools.structured_data_tool import (
    StructuredDataQueryTool,
)
from infrastructure.tools.tool_registry import (
    ToolRegistry,
)
from infrastructure.vector_store.in_memory_vector_store import (
    InMemoryVectorStore,
)


class Application:
    """
    Composition root.

    Toutes les dépendances de l'application sont construites ici.
    """

    def __init__(self) -> None:

        self.settings = get_settings()

        # ==============================================================
        # INFRASTRUCTURE : LLM
        # ==============================================================

        self.llm = OllamaLLMAdapter(
            self.settings.ollama
        )

        # ==============================================================
        # INFRASTRUCTURE : EMBEDDINGS
        # ==============================================================

        self.embedding_port = OllamaEmbeddingAdapter(
            self.settings.ollama
        )

        # ==============================================================
        # INFRASTRUCTURE : STOCKAGE / INDEX
        # ==============================================================

        self.vector_store = InMemoryVectorStore()

        self.lexical_index = BM25LexicalIndex()

        self.chunker = TextChunker(
            self.settings.chunking
        )

        self.document_repository = (
            InMemoryDocumentRepository()
        )

        self.evidence_pool = InMemoryEvidencePool(
            self.embedding_port
        )

        # ==============================================================
        # INFRASTRUCTURE : DONNÉES STRUCTURÉES
        # ==============================================================

        self.structured_data_registry = (
            StructuredDataRegistry()
        )

        self.parser_factory = DocumentParserFactory(
            self.structured_data_registry
        )

        # ==============================================================
        # INFRASTRUCTURE : OUTILS
        # ==============================================================

        self.tool_registry = ToolRegistry(
            [
                CalculatorTool(),

                StructuredDataQueryTool(
                    self.llm,
                    self.structured_data_registry,
                ),
            ]
        )

        # ==============================================================
        # SERVICES : INFORMATION NEED
        # ==============================================================

        self.information_need_service = (
            InformationNeedService(
                self.llm
            )
        )

        # ==============================================================
        # SERVICES : RETRIEVAL
        # ==============================================================

        self.retrieval_service = RetrievalService(
            self.embedding_port,
            self.vector_store,
            self.lexical_index,
            self.settings.retrieval,
        )

        # ==============================================================
        # SERVICES : EVIDENCE EVALUATION
        # ==============================================================

        self.evidence_evaluator = (
            EvidenceEvaluatorService(
                self.llm,
                relevance_threshold=(
                    self.settings.tree_search.relevance_threshold
                ),
                info_gain_threshold=(
                    self.settings.tree_search.information_gain_threshold
                ),
            )
        )

        # ==============================================================
        # SERVICES : ACTION PLANNER
        # ==============================================================

        self.action_planner = (
            EvidenceActionPlannerService(
                self.llm,
                self.tool_registry,
            )
        )

        # ==============================================================
        # SERVICES : CALCULATION INTENT EXTRACTION
        # ==============================================================

        self.calculation_intent_extractor = (
            CalculationIntentExtractorService(
                self.llm
            )
        )

        # ==============================================================
        # SERVICES : EVIDENCE ACTION ENGINE
        # ==============================================================

        self.action_engine = EvidenceActionEngine(
            planner=self.action_planner,
            retrieval_service=self.retrieval_service,
            tool_registry=self.tool_registry,
            evidence_pool=self.evidence_pool,
            evaluator=self.evidence_evaluator,
            settings=self.settings.action_engine,
            calculation_intent_extractor=(
                self.calculation_intent_extractor
            ),
        )

        # ==============================================================
        # SERVICES : BRANCH EXPLORATION
        # ==============================================================

        self.branch_exploration_service = (
            BranchExplorationService(
                self.llm,
                self.settings.tree_search,
            )
        )

        # ==============================================================
        # SERVICES : HYPOTHESES
        # ==============================================================

        self.hypothesis_service = (
            HypothesisService(
                self.llm
            )
        )

        # ==============================================================
        # SERVICES : CROSS-BRANCH REASONING
        # ==============================================================

        self.cross_branch_service = (
            CrossBranchReasoningService(
                self.llm
            )
        )

        # ==============================================================
        # SERVICES : FINAL SYNTHESIS
        # ==============================================================

        self.synthesis_service = (
            SynthesisService(
                self.llm
            )
        )

        # ==============================================================
        # USE CASE : INGESTION
        # ==============================================================

        self.ingest_use_case = (
            IngestDocumentsUseCase(
                parser_factory=self.parser_factory,
                chunker=self.chunker,
                embedding_port=self.embedding_port,
                vector_store=self.vector_store,
                lexical_index=self.lexical_index,
                document_repository=self.document_repository,
            )
        )

        # ==============================================================
        # USE CASE : QUESTION / ANSWER
        # ==============================================================

        self.answer_question_use_case = (
            AnswerQuestionUseCase(
                self.information_need_service,
                self.action_engine,
                self.evidence_evaluator,
                self.branch_exploration_service,
                self.hypothesis_service,
                self.cross_branch_service,
                self.synthesis_service,
                self.settings.tree_search,
            )
        )


def run_session(app: Application) -> None:
    """
    Lance une session interactive CLI.
    """

    print(
        "Session: ingest <files...> | "
        "ask <question> | quit"
    )

    while True:

        try:
            command = input("\n> ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nSession terminée.")
            break

        if not command:
            continue

        # --------------------------------------------------------------
        # QUIT
        # --------------------------------------------------------------

        if command.lower() in {
            "quit",
            "exit",
            "q",
        }:
            print("Session terminée.")
            break

        # --------------------------------------------------------------
        # INGEST
        # --------------------------------------------------------------

        if command.lower().startswith("ingest "):

            files_text = command[
                len("ingest "):
            ].strip()

            if not files_text:
                print(
                    "Usage: ingest <fichier1> "
                    "<fichier2> ..."
                )
                continue

            file_paths = files_text.split()

            try:

                result = app.ingest_use_case.execute(
                    file_paths
                )

                print(result)

            except Exception as exc:

                print(
                    f"Erreur pendant l'ingestion: {exc}"
                )

            continue

        # --------------------------------------------------------------
        # ASK
        # --------------------------------------------------------------

        if command.lower().startswith("ask "):

            question = command[
                len("ask "):
            ].strip()

            if not question:
                print(
                    "Usage: ask <question>"
                )
                continue

            try:

                result = (
                    app.answer_question_use_case.execute(
                        question
                    )
                )

                # IMPORTANT :
                # Ne pas utiliser str(result).
                #
                # FinalAnswer contient des Evidence.
                # Evidence contient les embeddings utilisés
                # en interne par le système de retrieval.
                #
                # Afficher str(result) provoquait donc
                # l'affichage des vecteurs.
                #
                # On affiche uniquement les informations
                # destinées à l'utilisateur.

                print(
                    f"\n{result.text}"
                )

                print(
                    f"Confidence: "
                    f"{result.confidence:.2f}"
                )

                if result.citations:

                    print(
                        "\nCitations:"
                    )

                    for evidence in result.citations:

                        print(
                            f"- "
                            f"{evidence.citation_label()}"
                        )

            except Exception as exc:

                print(
                    f"Erreur pendant la question: {exc}"
                )

            continue

        # --------------------------------------------------------------
        # UNKNOWN COMMAND
        # --------------------------------------------------------------

        print(
            "Commande inconnue. Utilise :\n"
            "  ingest <files...>\n"
            "  ask <question>\n"
            "  quit"
        )


def main() -> None:
    """
    Point d'entrée principal.
    """

    parser = argparse.ArgumentParser(
        description="InsightTree"
    )

    parser.add_argument(
        "command",
        choices=["session"],
        help="Commande à exécuter.",
    )

    args = parser.parse_args()

    app = Application()

    if args.command == "session":
        run_session(app)


if __name__ == "__main__":
    main()