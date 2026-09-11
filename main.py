from __future__ import annotations

import argparse

from application.services.branch_exploration_service import BranchExplorationService
from application.services.cross_branch_reasoning_service import CrossBranchReasoningService
from application.services.evidence_action_engine import EvidenceActionEngine
from application.services.evidence_action_planner_service import EvidenceActionPlannerService
from application.services.evidence_evaluator_service import EvidenceEvaluatorService
from application.services.hypothesis_service import HypothesisService
from application.services.information_need_service import InformationNeedService
from application.services.retrieval_service import RetrievalService
from application.services.synthesis_service import SynthesisService
from application.use_cases.answer_question_use_case import AnswerQuestionUseCase
from application.use_cases.ingest_documents_use_case import IngestDocumentsUseCase
from config.settings import get_settings
from infrastructure.chunking.text_chunker import TextChunker
from infrastructure.embeddings.ollama_embedding_adapter import OllamaEmbeddingAdapter
from infrastructure.lexical.bm25_lexical_index import BM25LexicalIndex
from infrastructure.llm.ollama_llm_adapter import OllamaLLMAdapter
from infrastructure.parsers.parser_factory import DocumentParserFactory
from infrastructure.repositories.in_memory_document_repository import InMemoryDocumentRepository
from infrastructure.repositories.in_memory_evidence_pool import InMemoryEvidencePool
from infrastructure.tools.calculator_tool import CalculatorTool
from infrastructure.tools.structured_data_registry import StructuredDataRegistry
from infrastructure.tools.structured_data_tool import StructuredDataQueryTool
from infrastructure.tools.tool_registry import ToolRegistry
from infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore


class Application:
    def __init__(self) -> None:
        self.settings = get_settings()

        self.llm = OllamaLLMAdapter(self.settings.ollama)
        self.embedding = OllamaEmbeddingAdapter(self.settings.ollama)
        self.vector_store = InMemoryVectorStore()
        self.lexical_index = BM25LexicalIndex()
        self.chunker = TextChunker(self.settings.chunking)
        self.document_repository = InMemoryDocumentRepository()
        self.evidence_pool = InMemoryEvidencePool(self.embedding)

        self.structured_registry = StructuredDataRegistry()
        self.parser_factory = DocumentParserFactory(self.structured_registry)
        self.tool_registry = ToolRegistry([
            CalculatorTool(self.llm),
            StructuredDataQueryTool(self.llm, self.structured_registry),
        ])

        self.retrieval = RetrievalService(
            self.embedding,
            self.vector_store,
            self.lexical_index,
            self.settings.retrieval,
        )
        self.evaluator = EvidenceEvaluatorService(
            self.llm,
            self.settings.tree_search.relevance_threshold,
            self.settings.tree_search.information_gain_threshold,
        )
        self.planner = EvidenceActionPlannerService(self.llm, self.tool_registry)
        self.action_engine = EvidenceActionEngine(
            self.planner,
            self.retrieval,
            self.tool_registry,
            self.evidence_pool,
            self.evaluator,
            self.settings.action_engine,
        )
        self.information_need = InformationNeedService(self.llm)
        self.branch_exploration = BranchExplorationService(
            self.llm, self.settings.tree_search
        )
        self.hypothesis = HypothesisService(self.llm)
        self.cross_branch = CrossBranchReasoningService(self.llm)
        self.synthesis = SynthesisService(self.llm)

        self.ingest = IngestDocumentsUseCase(
            self.parser_factory,
            self.chunker,
            self.embedding,
            self.vector_store,
            self.lexical_index,
            self.document_repository,
        )
        self.answer = AnswerQuestionUseCase(
            self.information_need,
            self.action_engine,
            self.evaluator,
            self.branch_exploration,
            self.hypothesis,
            self.cross_branch,
            self.synthesis,
            self.settings.tree_search,
        )


def run_ingest(app: Application, paths: list[str]) -> None:
    documents = app.ingest.execute(paths)
    print(f"{len(documents)} document(s) ingéré(s).")
    for doc in documents:
        print(f"- {doc.source_path} [{doc.doc_type.value}]")


def run_ask(app: Application, question: str) -> None:
    answer = app.answer.execute(question)
    print("\n=== RÉPONSE ===")
    print(answer.text)
    print(f"\nConfiance: {answer.confidence:.0%}")

    if answer.hypotheses:
        print("\n=== HYPOTHÈSES ===")
        for h in answer.hypotheses:
            print(
                f"- {h.statement} | confiance={h.confidence:.0%} | "
                f"incertitude={h.remaining_uncertainty:.0%}"
            )

    if answer.citations:
        print("\n=== SOURCES ===")
        for evidence in answer.citations:
            print(f"- {evidence.citation_label()}")


def run_session(app: Application) -> None:
    print("Session: ingest <files...> | ask <question> | quit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line == "quit":
            break
        command, *rest = line.split(maxsplit=1)
        if command == "ingest" and rest:
            run_ingest(app, rest[0].split())
        elif command == "ask" and rest:
            run_ask(app, rest[0])
        else:
            print("Commande invalide.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive Evidence-Driven Tree Retrieval"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest")
    ingest_parser.add_argument("files", nargs="+")

    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("question")

    sub.add_parser("session")
    args = parser.parse_args()

    app = Application()
    if args.command == "ingest":
        run_ingest(app, args.files)
    elif args.command == "ask":
        run_ask(app, args.question)
    else:
        run_session(app)


if __name__ == "__main__":
    main()
