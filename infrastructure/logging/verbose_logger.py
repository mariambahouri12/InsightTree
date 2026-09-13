from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any


_TRACE_START = time.perf_counter()


def _is_verbose() -> bool:
    return (
        os.getenv("VERBOSE", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )


def _elapsed() -> str:
    elapsed = time.perf_counter() - _TRACE_START
    return f"+{elapsed:8.2f}s"


def vlog(message: str) -> None:
    """
    Verbose logger.

    Enabled when:
        VERBOSE=1
    """

    if not _is_verbose():
        return

    timestamp = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{timestamp}] [{_elapsed()}] {message}"
    )


def vlog_section(title: str) -> None:
    """
    Print a clearly separated tracing section.
    """

    if not _is_verbose():
        return

    print()
    print(
        "════════════════════════════════════════════════════════════"
    )
    print(f" {title}")
    print(
        "════════════════════════════════════════════════════════════"
    )


def vlog_subsection(title: str) -> None:
    """
    Print a smaller subsection.
    """

    if not _is_verbose():
        return

    print()
    print(
        "────────────────────────────────────────────────────────────"
    )
    print(f" {title}")
    print(
        "────────────────────────────────────────────────────────────"
    )


def vlog_kv(
    key: str,
    value: Any,
    indent: int = 2,
) -> None:
    """
    Print a key/value pair.
    """

    if not _is_verbose():
        return

    prefix = " " * indent

    print(
        f"{prefix}{key}: {value}"
    )


def vlog_text(
    label: str,
    text: str,
    max_chars: int = 1200,
    indent: int = 2,
) -> None:
    """
    Print a potentially long text value while keeping the
    terminal readable.
    """

    if not _is_verbose():
        return

    prefix = " " * indent

    text = str(text or "")

    if len(text) > max_chars:
        text = (
            text[:max_chars]
            + f"\n{prefix}... [truncated, "
            f"{len(text)} chars total]"
        )

    print(
        f"{prefix}{label}:"
    )

    for line in text.splitlines() or [""]:
        print(
            f"{prefix}  {line}"
        )


def vlog_json(
    label: str,
    data: Any,
    indent: int = 2,
) -> None:
    """
    Pretty-print a dictionary/list/JSON-compatible object.
    """

    if not _is_verbose():
        return

    prefix = " " * indent

    print(
        f"{prefix}{label}:"
    )

    try:
        formatted = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    except Exception:
        formatted = str(data)

    for line in formatted.splitlines():
        print(
            f"{prefix}  {line}"
        )


def vlog_evidence(
    evidence,
    index: int | None = None,
    max_chars: int = 1200,
    indent: int = 2,
) -> None:
    """
    Print an Evidence object in a consistent format.

    This function is intentionally defensive because Evidence
    can evolve without breaking the logger.
    """

    if not _is_verbose():
        return

    prefix = " " * indent

    if index is not None:
        print(
            f"{prefix}Evidence #{index}"
        )
    else:
        print(
            f"{prefix}Evidence"
        )

    chunk_id = getattr(
        evidence,
        "chunk_id",
        None,
    )

    evidence_id = getattr(
        evidence,
        "id",
        None,
    )

    score = getattr(
        evidence,
        "score",
        None,
    )

    reliability = getattr(
        evidence,
        "source_reliability",
        None,
    )

    retrieval_method = getattr(
        evidence,
        "retrieval_method",
        None,
    )

    source_metadata = getattr(
        evidence,
        "source_metadata",
        {},
    )

    text = getattr(
        evidence,
        "text",
        "",
    )

    if hasattr(retrieval_method, "value"):
        retrieval_method = retrieval_method.value

    vlog_kv(
        "chunk_id",
        chunk_id,
        indent + 2,
    )

    vlog_kv(
        "evidence_id",
        evidence_id,
        indent + 2,
    )

    vlog_kv(
        "method",
        retrieval_method,
        indent + 2,
    )

    if score is not None:
        try:
            score = f"{float(score):.4f}"
        except (TypeError, ValueError):
            pass

    vlog_kv(
        "score",
        score,
        indent + 2,
    )

    if reliability is not None:
        try:
            reliability = f"{float(reliability):.4f}"
        except (TypeError, ValueError):
            pass

    vlog_kv(
        "reliability",
        reliability,
        indent + 2,
    )

    if source_metadata:
        vlog_json(
            "source_metadata",
            source_metadata,
            indent + 2,
        )

    vlog_text(
        "text",
        text,
        max_chars=max_chars,
        indent=indent + 2,
    )


def vlog_evidence_list(
    evidence_list,
    title: str = "Evidence",
    max_chars: int = 1200,
) -> None:
    """
    Print a complete evidence list.
    """

    if not _is_verbose():
        return

    vlog_subsection(
        f"{title} ({len(evidence_list)} item(s))"
    )

    if not evidence_list:
        vlog(
            "  No evidence."
        )
        return

    for index, evidence in enumerate(
        evidence_list,
        start=1,
    ):
        vlog_evidence(
            evidence,
            index=index,
            max_chars=max_chars,
        )


def reset_trace_timer() -> None:
    """
    Reset the elapsed-time counter.

    Useful when starting a new interactive question.
    """

    global _TRACE_START

    _TRACE_START = time.perf_counter()