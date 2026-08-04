"""
Runs every query in sql/queries.sql against sql/churn.db, printing a preview
of each result and saving the full result set to sql/query_results/<name>.csv.

Queries are split out of queries.sql using the "-- @name: <id>" header
convention documented at the top of that file, so adding a new query there
is automatically picked up here with no code changes.

Run with:  python sql/run_queries.py
(requires sql/churn.db to already exist -- run load_to_sqlite.py first)
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parent
DB_PATH = SQL_DIR / "churn.db"
QUERIES_PATH = SQL_DIR / "queries.sql"
RESULTS_DIR = SQL_DIR / "query_results"

NAME_HEADER_RE = re.compile(r"--\s*@name:\s*(\w+)")


def parse_queries(sql_text: str) -> list[tuple[str, str]]:
    """Split queries.sql into (name, sql) pairs using the @name headers."""
    matches = list(NAME_HEADER_RE.finditer(sql_text))
    queries = []
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sql_text)
        sql = sql_text[start:end].strip().rstrip(";")
        queries.append((name, sql))
    return queries


def main():
    if not DB_PATH.exists():
        raise SystemExit(
            f"{DB_PATH} not found. Run 'python sql/load_to_sqlite.py' first."
        )

    sql_text = QUERIES_PATH.read_text(encoding="utf-8")
    queries = parse_queries(sql_text)
    if not queries:
        raise SystemExit(f"No '-- @name:' queries found in {QUERIES_PATH}.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        for name, sql in queries:
            print(f"\n=== {name} ===")
            df = pd.read_sql_query(sql, conn)
            print(df.to_string(index=False))

            out_path = RESULTS_DIR / f"{name}.csv"
            df.to_csv(out_path, index=False)
            print(f"-> saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
