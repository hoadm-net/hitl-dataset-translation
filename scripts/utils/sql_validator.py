"""
SQL pattern extraction and rule-based validation utilities.

Used to:
- Extract the set of SQL operators present in a query (for few-shot selection
  in Phase 3 and stratified sampling in Phase 2).
- Validate that a translated sample preserves the same SQL operators as the
  original, catching hallucinated or dropped SQL components.
"""

from __future__ import annotations

import re

# Ordered list of SQL operator tokens to track.
# Kept simple and dataset-agnostic — covers all constructs in Spider and BIRD.
SQL_OPERATORS = [
    "SELECT",
    "FROM",
    "WHERE",
    "JOIN",
    "LEFT JOIN",
    "INNER JOIN",
    "GROUP BY",
    "ORDER BY",
    "HAVING",
    "LIMIT",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "DISTINCT",
    "COUNT",
    "SUM",
    "AVG",
    "MAX",
    "MIN",
    "NOT IN",
    "IN",
    "LIKE",
    "BETWEEN",
    "EXISTS",
    "CASE",
    "CAST",
    "SUBSTR",
    "STRFTIME",
]

# Build patterns once at module load time (longest-first to avoid partial matches)
_PATTERNS = [
    (op, re.compile(r"\b" + re.escape(op) + r"\b", re.IGNORECASE))
    for op in sorted(SQL_OPERATORS, key=len, reverse=True)
]


def extract_sql_patterns(query: str) -> list[str]:
    """
    Return the sorted list of SQL operator tokens present in *query*.

    Args:
        query: A SQL query string.

    Returns:
        Sorted list of matched operator names (uppercase), e.g. ["COUNT", "FROM",
        "GROUP BY", "SELECT", "WHERE"].
    """
    found = []
    for op, pattern in _PATTERNS:
        if pattern.search(query):
            found.append(op)
    return sorted(found)


def patterns_match(query_a: str, query_b: str) -> bool:
    """
    Return True if both queries share the same set of SQL operators.

    Useful for verifying that a translation prompt did not accidentally alter
    the SQL component set (e.g., the model adding a spurious COUNT).

    Args:
        query_a: First SQL query.
        query_b: Second SQL query.
    """
    return set(extract_sql_patterns(query_a)) == set(extract_sql_patterns(query_b))


def get_pattern_signature(query: str) -> str:
    """
    Return a canonical string representation of the SQL operator set
    for grouping and stratified sampling.

    Example: "COUNT|FROM|GROUP_BY|SELECT|WHERE"
    """
    return "|".join(p.replace(" ", "_") for p in extract_sql_patterns(query))
