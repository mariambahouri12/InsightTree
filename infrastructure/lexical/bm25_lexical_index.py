from __future__ import annotations

import math
import re
from collections import Counter

from application.ports.lexical_index_port import LexicalIndexPort
from domain.entities import Chunk, Evidence
from domain.enums import RetrievalMethod


_TOKEN_RE = re.compile(
    r"\w+",
    re.UNICODE,
)

_K1 = 1.5
_B = 0.75


def _tokenize(
    text: str,
) -> list[str]:
    return _TOKEN_RE.findall(
        text.lower()
    )


class BM25LexicalIndex(
    LexicalIndexPort
):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_doc_length = 0.0

    def add_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:

        for chunk in chunks:

            tokens = _tokenize(
                chunk.text
            )

            freqs = Counter(tokens)

            self._chunks.append(chunk)
            self._doc_freqs.append(freqs)
            self._doc_lengths.append(
                len(tokens)
            )

            for term in freqs:
                self._df[term] += 1

        if self._doc_lengths:
            self._avg_doc_length = (
                sum(self._doc_lengths)
                / len(self._doc_lengths)
            )

    def search(
        self,
        query_text: str,
        top_k: int,
    ) -> list[Evidence]:

        if not self._chunks:
            return []

        query_terms = _tokenize(
            query_text
        )

        if not query_terms:
            return []

        n_docs = len(
            self._chunks
        )

        scores = [
            0.0
            for _ in range(n_docs)
        ]

        for term in query_terms:

            df = self._df.get(
                term,
                0,
            )

            if not df:
                continue

            idf = math.log(
                1
                + (
                    n_docs
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

            for i, freqs in enumerate(
                self._doc_freqs
            ):

                tf = freqs.get(
                    term,
                    0,
                )

                if not tf:
                    continue

                length = (
                    self._doc_lengths[i]
                )

                denom = (
                    tf
                    + _K1
                    * (
                        1
                        - _B
                        + _B
                        * length
                        / (
                            self._avg_doc_length
                            or 1.0
                        )
                    )
                )

                scores[i] += (
                    idf
                    * (
                        tf
                        * (
                            _K1
                            + 1
                        )
                    )
                    / denom
                )

        ranked = sorted(
            range(n_docs),
            key=scores.__getitem__,
            reverse=True,
        )

        result = []

        for i in ranked[:top_k]:

            if scores[i] <= 0:
                continue

            chunk = self._chunks[i]

            result.append(
                Evidence(
                    text=chunk.text,
                    score=scores[i],
                    source_metadata=dict(
                        chunk.metadata
                    ),
                    retrieval_method=(
                        RetrievalMethod.LEXICAL
                    ),
                    chunk_id=chunk.id,
                    embedding=chunk.embedding,
                )
            )

        return result