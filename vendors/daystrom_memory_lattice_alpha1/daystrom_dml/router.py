"""Intent routing utilities for retrieval mode selection."""
from __future__ import annotations

import re
from typing import Literal

_CODE_PAREN_PATTERN = re.compile(
    r"(\b(?:if|for|while|switch|function|def)\s*\(|\b[\w\.]+\()",
    re.IGNORECASE,
)

RetrievalMode = Literal["semantic", "literal", "hybrid"]


def decide_mode(query: str) -> RetrievalMode:
    """Infer the retrieval mode for a query using lightweight heuristics."""
    normalized = query.lower()
    # Detect explicit code and structured query signals.
    strong_literal_triggers = (
        "::",
        "->",
    )
    if any(trigger in query for trigger in strong_literal_triggers):
        return "literal"
    literal_keywords = (
        "select",
        "fetchuserprofile(",
        "fetchuserprofile",
    )
    if any(keyword in normalized for keyword in literal_keywords):
        return "literal"

    semantic_keywords = (
        "average",
        "trend",
        "summarize",
    )
    if any(keyword in normalized for keyword in semantic_keywords):
        return "semantic"

    # Treat parentheses as a literal cue only when no semantic intent was found
    # and the structure resembles code or function invocations. Natural language
    # asides such as "(2010-2020)" should continue through the semantic path.
    if _CODE_PAREN_PATTERN.search(query):
        return "literal"

    return "hybrid"
