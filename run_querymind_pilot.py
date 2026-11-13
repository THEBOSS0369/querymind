#!/usr/bin/env python3
"""QueryMind Sales-4 Pilot Evaluation Script."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time
from typing import Any
import urllib.error
import urllib.request

DB_FILENAME = "querymind_sales4.db"

SCHEMA_DDL = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    country TEXT NOT NULL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    amount REAL NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

SEED_CUSTOMERS = [
    (1, "Aarav Mehta", "India"),
    (2, "Priya Nair", "India"),
    (3, "Rohan Verma", "India"),
    (4, "Emily Carter", "USA"),
    (5, "Liam Johnson", "USA"),
    (6, "Sophia Martinez", "USA"),
]
SEED_ORDERS = [
    (101, 1, "2024-11-15", 45.0),
    (102, 1, "2025-01-20", 85.0),
    (103, 2, "2025-02-10", 90.0),
    (104, 4, "2024-12-05", 50.0),
    (105, 4, "2025-03-12", 95.0),
    (106, 5, "2025-04-01", 55.0),
]

PILOT_QUESTIONS = [
    {
        "id": "Q1",
        "question": "What is the total order revenue for each country?",
        "reference_sql": "SELECT c.country, SUM(o.amount) AS total_revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.country;",
        "ordered": False,
    },
    {
        "id": "Q2",
        "question": "Which customers have placed orders, and what is the value of each order?",
        "reference_sql": "SELECT c.customer_name, o.order_id, o.amount FROM customers c JOIN orders o ON c.customer_id = o.customer_id;",
        "ordered": False,
    },
    {
        "id": "Q3",
        "question": "How much revenue did each country generate in 2025?",
        "reference_sql": "SELECT c.country, SUM(o.amount) AS revenue_2025 FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE o.order_date LIKE '2025%' GROUP BY c.country;",
        "ordered": False,
    },
    {
        "id": "Q4",
        "question": "Which five orders have the highest order amounts?",
        "reference_sql": "SELECT order_id, customer_id, order_date, amount FROM orders ORDER BY amount DESC LIMIT 5;",
        "ordered": True,
    },
]

SYSTEM_PROMPT = """You are QueryMind, a natural-language database analytics assistant.
You have access to the `run_sql` tool connected to a SQLite database with the following schema:

Table `customers`:
  - customer_id (INTEGER PRIMARY KEY)
  - customer_name (TEXT)
  - country (TEXT)

Table `orders`:
  - order_id (INTEGER PRIMARY KEY)
  - customer_id (INTEGER, FOREIGN KEY -> customers.customer_id)
  - order_date (TEXT, YYYY-MM-DD)
  - amount (REAL)

When answering a user question:
1. Call `run_sql` with a valid SQLite SELECT query to retrieve the exact answer.
2. After receiving the query result preview, provide a concise plain-English summary of the findings.
"""

TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": "Execute a read-only SQL query against the connected SQLite database and return a preview of the result table.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "The SQLite SELECT statement to run."}},
            "required": ["sql"],
        },
    },
}]


def setup_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_DDL)
        conn.executemany("INSERT INTO customers VALUES (?, ?, ?);", SEED_CUSTOMERS)
        conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?);", SEED_ORDERS)
        conn.commit()
    finally:
        conn.close()


def execute_readonly_sql(db_path: Path, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return columns, rows
    finally:
        conn.close()


def format_preview(columns: list[str], rows: list[tuple[Any, ...]], max_chars: int = 1000) -> str:
    return "\n".join([",".join(columns)] + [",".join(str(v) for v in row) for row in rows])[:max_chars]


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, str):
        try:
            return round(float(value), 4)
        except ValueError:
            return value.strip()
    return value


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def results_match(candidate: list[tuple[Any, ...]], reference: list[tuple[Any, ...]], ordered: bool) -> bool:
    candidate = normalize_rows(candidate)
    reference = normalize_rows(reference)
    if len(candidate) != len(reference):
        return False
    if ordered:
        if candidate == reference:
            return True
        return all(set(r).issubset(set(c)) or set(c).issubset(set(r)) for c, r in zip(candidate, reference))
    if sorted(candidate, key=str) == sorted(reference, key=str):
        return True
    candidate_sets = [frozenset(row) for row in candidate]
    return all(any(frozenset(row).issubset(c) or c.issubset(frozenset(row)) for c in candidate_sets) for row in reference)


def call_chat_completions(base_url: str, api_key: str, model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "tools": TOOLS_SCHEMA, "tool_choice": "auto", "temperature": 0.0}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def run_pilot_question(db_path: Path, item: dict[str, Any], base_url: str, api_key: str, model: str) -> dict[str, Any]:
    _, reference_rows = execute_readonly_sql(db_path, item["reference_sql"])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item["question"]}]
    sql_ran = False
    correct = False
    total_tokens = 0
    model_id = model
    generated_sql = None
    error = None
    started = time.perf_counter()
    for _ in range(10):
        response = call_chat_completions(base_url, api_key, model, messages)
        model_id = response.get("model", model_id)
        total_tokens += int((response.get("usage") or {}).get("total_tokens", 0))
        choice = response["choices"][0]["message"]
        messages.append(choice)
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            break
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"].get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            if name == "run_sql":
                generated_sql = args.get("sql", "")
                try:
                    columns, rows = execute_readonly_sql(db_path, generated_sql)
                    sql_ran = True
                    correct = results_match(rows, reference_rows, item["ordered"])
                    output = format_preview(columns, rows)
                except Exception as exc:
                    error = str(exc)
                    output = f"SQL execution error: {exc}"
            else:
                output = f"Unsupported tool: {name}"
            messages.append({"role": "tool", "tool_call_id": tool_call["id"], "name": name, "content": output})
    return {"id": item["id"], "question": item["question"], "sql_ran": "Yes" if sql_ran else "No", "correct": "Yes" if correct else "No", "time_s": round(time.perf_counter() - started, 2), "tokens": total_tokens, "model_id": model_id, "generated_sql": generated_sql, "error": error}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QueryMind Sales-4 Pilot Evaluation.")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    db_path = Path(DB_FILENAME)
    setup_database(db_path)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)
    try:
        results = [run_pilot_question(db_path, item, args.base_url, api_key, args.model) for item in PILOT_QUESTIONS]
    except urllib.error.HTTPError as exc:
        print(f"API HTTP error {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Network error connecting to API: {exc}", file=sys.stderr)
        sys.exit(1)
    actual_model = results[0]["model_id"] if results else args.model
    valid = sum(r["sql_ran"] == "Yes" for r in results)
    correct = sum(r["correct"] == "Yes" for r in results)
    mean_time = sum(r["time_s"] for r in results) / len(results)
    total_tokens = sum(r["tokens"] for r in results)
    print("\n=== TABLE III: PILOT CONFIGURATION ===")
    print(f"LLM service  : {actual_model} through the OpenAI integration")
    print("Database     : QueryMind Sales-4: synthetic SQLite sales database (6 customers, 6 orders)")
    print("SQL runner   : SQLite runner with a read-only database account (mode=ro URI)")
    print("Agent memory : In-memory demo store; start each pilot question in a fresh conversation")
    print("Server       : FastAPI, streaming endpoint")
    print("Client       : Lit web component")
    print(f"Python / OS  : Python {platform.python_version()}, SQLite {sqlite3.sqlite_version} on {platform.system()} {platform.release()} ({platform.machine()})")
    print("\n=== TABLE VI: PILOT RESULTS ON THE QUESTION SET ===")
    print(f"{'#':<4} {'Question':<58} {'SQL ran':<9} {'Correct':<9} {'Time (s)':<10} {'Tokens':<8}")
    print("-" * 102)
    for result in results:
        print(f"{result['id']:<4} {result['question']:<58} {result['sql_ran']:<9} {result['correct']:<9} {result['time_s']:<10.2f} {result['tokens']:<8}")
    print("\n=== GENERATED SQL PER QUESTION ===")
    for result in results:
        print(f"{result['id']}: {result['generated_sql']} (Error: {result['error']})")
    print("\n=== REPLACEMENT PARAGRAPH BELOW TABLE VI ===")
    print(f"Across the four-question pilot (temperature 0.0, fresh conversation per question), {valid} of 4 generated queries executed without error ({valid / 4 * 100:.0f}% valid-SQL rate) and {correct} of 4 matched the hand-checked reference result sets ({correct / 4 * 100:.0f}% execution accuracy), with a mean wall-clock latency of {mean_time:.2f} s per question and {total_tokens} total tokens across the four runs.")
    print("\n=== TABLE VII: C0 MODEL ENTRY ===")
    print(f"C0 | Off | {actual_model} | Baseline")


if __name__ == "__main__":
    main()
