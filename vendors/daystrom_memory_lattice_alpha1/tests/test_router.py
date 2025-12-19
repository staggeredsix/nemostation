import pytest

from daystrom_dml.router import decide_mode


@pytest.mark.parametrize(
    "query",
    [
        "fetchUserProfile(userId)",
        "SELECT * FROM users",
        "Look up item(id)",
    ],
)
def test_literal_queries(query: str) -> None:
    assert decide_mode(query) == "literal"


@pytest.mark.parametrize(
    "query",
    [
        "What is the average response time?",
        "Show me the trend over the last week",
        "Summarize the incidents today",
        "Summarize incidents (last week)",
    ],
)
def test_semantic_queries(query: str) -> None:
    assert decide_mode(query) == "semantic"


@pytest.mark.parametrize(
    "query",
    [
        "How are the users grouped?",
        "List user feedback",  # no summary verbs or code cues
    ],
)
def test_hybrid_queries(query: str) -> None:
    assert decide_mode(query) == "hybrid"
