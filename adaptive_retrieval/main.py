"""Point d'entrée CLI + composition root (câblage des dépendances).

Usage:
    python main.py ingest fichier1.pdf fichier2.docx data.xlsx
    python main.py ask "Pourquoi les ventes du produit B ont baissé en 2026 ?"
"""
from __future__ import annotations

import argparse
import sys

from application.services.branch_exploration_service import BranchExplorationService
from application.services.evidence_evaluator_service import EvidenceEvaluatorService
from application.services.hypothesis_service import HypothesisService
from application.services.query_generation_service import QueryGenerationService
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
from infrastructure.vector_store.in_memory_vector_store import InMemoryVectorStore


class Application:
    """Composition root : instancie et câble toutes les dépendances une seule
    fois (singletons applicatifs), en respectant l'inversion de dépendance."""

    def __init__(self) -> None:
        self.settings = get_settings()

        # --- Infrastructure ---
        self.llm = OllamaLLMAdapter(self.settings.ollama)
        self.embedding_port = OllamaEmbeddingAdapter(self.settings.ollama)
        self.vector_store = InMemoryVectorStore()
        self.lexical_index = BM25LexicalIndex()
        self.chunker = TextChunker(self.settings.chunking)
        self.parser_factory = DocumentParserFactory()
        self.document_repository = InMemoryDocumentRepository()

        # --- Services applicatifs ---
        self.query_generation_service = QueryGenerationService(self.llm)
        self.retrieval_service = RetrievalService(
            self.embedding_port, self.vector_store, self.lexical_index, self.settings.retrieval
        )
        self.evidence_evaluator = EvidenceEvaluatorService(
            self.llm,
            relevance_threshold=self.settings.tree_search.relevance_threshold,
            info_gain_threshold=self.settings.tree_search.information_gain_threshold,
        )
        self.branch_exploration_service = BranchExplorationService(self.llm, self.settings.tree_search)
        self.hypothesis_service = HypothesisService(self.llm)
        self.synthesis_service = SynthesisService(self.llm)

        # --- Use cases ---
        self.ingest_use_case = IngestDocumentsUseCase(
            self.parser_factory,
            self.chunker,
            self.embedding_port,
            self.vector_store,
            self.lexical_index,
            self.document_repository,
        )
        self.answer_question_use_case = AnswerQuestionUseCase(
            self.query_generation_service,
            self.retrieval_service,
            self.evidence_evaluator,
            self.branch_exploration_service,
            self.hypothesis_service,
            self.synthesis_service,
            self.settings.tree_search,
        )


def _run_ingest(app: Application, file_paths: list[str]) -> None:
    documents = app.ingest_use_case.execute(file_paths)
    print(f"{len(documents)} document(s) ingéré(s) et indexé(s).")
    for doc in documents:
        print(f"  - {doc.source_path} ({doc.doc_type.value})")


def _run_ask(app: Application, question: str) -> None:
    answer = app.answer_question_use_case.execute(question)

    print("\n=== RÉPONSE ===")
    print(answer.text)
    print(f"\nConfiance globale: {answer.confidence:.0%}")

    if answer.hypotheses:
        print("\n=== HYPOTHÈSES ===")
        for h in answer.hypotheses:
            print(f"- {h.statement} (confiance: {h.confidence:.0%})")

    if answer.citations:
        print("\n=== SOURCES ===")
        for c in answer.citations:
            print(f"- {c.citation_label()}")


def _run_session(app: Application) -> None:
    """Mode interactif : indispensable car le vector store / index lexical
    sont en mémoire (non persistés). `ingest` puis `ask` dans deux appels
    CLI séparés ne partageraient pas l'index. Utilisez ce mode, ou
    remplacez InMemoryVectorStore par un adaptateur persistant (Chroma...)."""
    print("Session interactive. Commandes: 'ingest <fichier1> <fichier2> ...', 'ask <question>', 'quit'")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line or line == "quit":
            break
        parts = line.split(maxsplit=1)
        command = parts[0]
        if command == "ingest" and len(parts) > 1:
            _run_ingest(app, parts[1].split())
        elif command == "ask" and len(parts) > 1:
            _run_ask(app, parts[1])
        else:
            print("Commande invalide.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive Evidence-Driven Tree Retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingérer et indexer des documents")
    ingest_parser.add_argument("files", nargs="+", help="Chemins des fichiers à ingérer")

    ask_parser = subparsers.add_parser("ask", help="Poser une question (nécessite un index déjà en mémoire)")
    ask_parser.add_argument("question", help="Question en langage naturel")

    subparsers.add_parser("session", help="Mode interactif ingest+ask dans le même processus (recommandé)")

    args = parser.parse_args()
    app = Application()

    if args.command == "ingest":
        _run_ingest(app, args.files)
    elif args.command == "ask":
        _run_ask(app, args.question)
    elif args.command == "session":
        _run_session(app)


if __name__ == "__main__":
    sys.exit(main() or 0)
