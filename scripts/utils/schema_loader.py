"""
Database schema loading utilities.

Provides helpers to load and query schema information from Spider-format
`tables.json` files (used by both Spider and BIRD).
"""

from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=4)
def load_schemas(tables_json_path: str) -> dict[str, dict]:
    """
    Load a tables.json file and return a dict keyed by db_id.

    Args:
        tables_json_path: Absolute or relative path to tables.json.

    Returns:
        Dict mapping db_id -> full schema dict.
    """
    path = Path(tables_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, encoding="utf-8") as f:
        schemas = json.load(f)

    return {s["db_id"]: s for s in schemas}


def get_schema(tables_json_path: str, db_id: str) -> dict:
    """
    Return the schema dict for a specific database.

    Args:
        tables_json_path: Path to tables.json.
        db_id: Database identifier.

    Returns:
        Schema dict with keys: table_names_original, table_names,
        column_names_original, column_names, column_types,
        primary_keys, foreign_keys.

    Raises:
        KeyError: If db_id is not found in the schema file.
    """
    schemas = load_schemas(tables_json_path)
    if db_id not in schemas:
        raise KeyError(f"Database '{db_id}' not found in {tables_json_path}")
    return schemas[db_id]


def get_table_names(tables_json_path: str, db_id: str) -> list[str]:
    """Return the list of original table names for a database."""
    schema = get_schema(tables_json_path, db_id)
    return schema["table_names_original"]


def get_column_names(tables_json_path: str, db_id: str) -> list[tuple[int, str]]:
    """
    Return column names as (table_index, column_name) tuples.

    table_index == -1 denotes the special '*' column.
    """
    schema = get_schema(tables_json_path, db_id)
    return [tuple(c) for c in schema["column_names_original"]]


def format_schema_text(tables_json_path: str, db_id: str) -> str:
    """
    Format the schema as a compact text string for use in prompts.

    Example output:
        Table: singer (singer_id, name, birth_year, ...)
        Table: concert (concert_id, concert_name, theme, ...)

    Args:
        tables_json_path: Path to tables.json.
        db_id: Database identifier.

    Returns:
        Multi-line string describing the schema.
    """
    schema = get_schema(tables_json_path, db_id)
    tables = schema["table_names_original"]
    cols = schema["column_names_original"]

    # Group columns by table index
    table_cols: dict[int, list[str]] = {i: [] for i in range(len(tables))}
    for tbl_idx, col_name in cols:
        if tbl_idx >= 0:
            table_cols[tbl_idx].append(col_name)

    lines = []
    for i, tbl in enumerate(tables):
        col_str = ", ".join(table_cols.get(i, []))
        lines.append(f"Table: {tbl} ({col_str})")

    return "\n".join(lines)
