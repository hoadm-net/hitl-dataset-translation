"""
SQL pattern extraction, classification, and validation utilities.

Used to:
- Extract the set of SQL operators present in a query (for validation in Phase 3).
- Classify each query into a structural SQL class (for stratified sampling in Phase 1).
- Validate that a translated sample preserves the same SQL operators as the
  original, catching hallucinated or dropped SQL components.

SQL Class Taxonomy
------------------
Ten mutually exclusive classes assigned by priority (highest wins):

  SET_OP       — UNION / INTERSECT / EXCEPT
  NESTED       — subquery (more than one SELECT keyword)
  GROUP_HAVING — GROUP BY + HAVING
  GROUP_BY     — GROUP BY without HAVING
  JOIN_ORDER   — JOIN + (ORDER BY or LIMIT)
  JOIN         — any JOIN, no GROUP BY, no ORDER BY
  ORDER_LIMIT  — ORDER BY / LIMIT, no JOIN, no GROUP BY
  AGG_ONLY     — aggregation function (COUNT/SUM/AVG/MAX/MIN), no JOIN, no GROUP BY
  SELECT_WHERE — WHERE clause, no JOIN, no GROUP BY, no aggregation
  SIMPLE       — SELECT … FROM … only

These classes capture the structural complexity that most affects translation
difficulty (e.g., multi-hop reasoning for NESTED, domain arithmetic for AGG_ONLY).
Together with the hardness/difficulty label, they define the two-dimensional
strata used for greedy coverage sampling.
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
    Return a canonical string representation of the SQL operator set.
    Used for fine-grained grouping and debugging.

    Example: "COUNT|FROM|GROUP_BY|SELECT|WHERE"
    """
    return "|".join(p.replace(" ", "_") for p in extract_sql_patterns(query))


# ---------------------------------------------------------------------------
# SQL Class Taxonomy
# ---------------------------------------------------------------------------

_AGGREGATION_FUNCS = re.compile(r"\b(COUNT|SUM|AVG|MAX|MIN)\b", re.IGNORECASE)
_JOIN = re.compile(r"\bJOIN\b", re.IGNORECASE)
_GROUP_BY = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_HAVING = re.compile(r"\bHAVING\b", re.IGNORECASE)
_ORDER_LIMIT = re.compile(r"\b(ORDER\s+BY|LIMIT)\b", re.IGNORECASE)
_WHERE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_SET_OPS = re.compile(r"\b(UNION|INTERSECT|EXCEPT)\b", re.IGNORECASE)
_SELECT = re.compile(r"\bSELECT\b", re.IGNORECASE)


def classify_sql(query: str) -> str:
    """
    Classify a SQL query into one of ten structural classes.

    Classes (assigned by priority — highest wins):
      SET_OP       UNION / INTERSECT / EXCEPT
      NESTED       subquery (>1 SELECT keyword)
      GROUP_HAVING GROUP BY + HAVING
      GROUP_BY     GROUP BY without HAVING
      JOIN_ORDER   JOIN + (ORDER BY or LIMIT)
      JOIN         any JOIN, no GROUP BY, no ORDER BY/LIMIT
      ORDER_LIMIT  ORDER BY / LIMIT, no JOIN, no GROUP BY
      AGG_ONLY     aggregation function, no JOIN, no GROUP BY
      SELECT_WHERE WHERE clause only
      SIMPLE       SELECT … FROM … only

    Args:
        query: A SQL query string.

    Returns:
        One of the ten class name strings above.
    """
    has_set_op   = bool(_SET_OPS.search(query))
    is_nested    = len(_SELECT.findall(query)) > 1
    has_group    = bool(_GROUP_BY.search(query))
    has_having   = bool(_HAVING.search(query))
    has_join     = bool(_JOIN.search(query))
    has_order    = bool(_ORDER_LIMIT.search(query))
    has_agg      = bool(_AGGREGATION_FUNCS.search(query))
    has_where    = bool(_WHERE.search(query))

    if has_set_op:
        return "SET_OP"
    if is_nested:
        return "NESTED"
    if has_group and has_having:
        return "GROUP_HAVING"
    if has_group:
        return "GROUP_BY"
    if has_join and has_order:
        return "JOIN_ORDER"
    if has_join:
        return "JOIN"
    if has_order:
        return "ORDER_LIMIT"
    if has_agg:
        return "AGG_ONLY"
    if has_where:
        return "SELECT_WHERE"
    return "SIMPLE"


# Ordered list of all SQL classes (simple → complex) for display/reporting
SQL_CLASSES = [
    "SIMPLE",
    "SELECT_WHERE",
    "AGG_ONLY",
    "ORDER_LIMIT",
    "JOIN",
    "JOIN_ORDER",
    "GROUP_BY",
    "GROUP_HAVING",
    "NESTED",
    "SET_OP",
]
