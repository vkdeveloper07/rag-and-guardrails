"""Evaluate retrieved chunks against the labeled evaluation set."""

import json
import sys
from pathlib import Path

# Make the fitness_agent package in src/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from fitness_agent.config import settings
from fitness_agent.evaluation.metrics import (
    f1_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from fitness_agent.rag.query_rewriter import rewrite_query
from fitness_agent.rag.retriever import retrieve

console = Console()

EVAL_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_set.json"
KS = [1, 3, 5, 10]


def load_eval_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def compute_metrics_for_question(question: str, relevant: dict[str, int], ks: list[int]) -> tuple[dict[int, dict[str, float]], float, str]:
    search_query = rewrite_query(question)
    retrieved = retrieve(search_query, max(ks))
    ids = [chunk.id for chunk in retrieved]

    summary: dict[int, dict[str, float]] = {}
    avg_f1 = 0.0
    question_id = ""

    for k in ks:
        top_ids = ids[:k]
        metrics = {
            "precision": precision_at_k(top_ids, relevant, k),
            "recall": recall_at_k(top_ids, relevant, k),
            "f1": f1_at_k(top_ids, relevant, k),
            "mrr": mrr(top_ids, relevant),
            "map": map_at_k(top_ids, relevant, k),
            "ndcg": ndcg_at_k(top_ids, relevant, k),
        }
        summary[k] = metrics
        avg_f1 += metrics["f1"]

    avg_f1 /= len(ks)
    if ids:
        question_id = ids[0]
    return summary, avg_f1, search_query


def print_metrics_table(results: list[dict]) -> None:
    table = Table(title="Retrieval evaluation", show_lines=True)
    table.add_column("K")
    table.add_column("Precision@K")
    table.add_column("Recall@K")
    table.add_column("F1@K")
    table.add_column("MRR")
    table.add_column("MAP@K")
    table.add_column("NDCG@K")

    for k in KS:
        row = [str(k)]
        values = [
            sum(item["metrics"][k]["precision"] for item in results) / len(results),
            sum(item["metrics"][k]["recall"] for item in results) / len(results),
            sum(item["metrics"][k]["f1"] for item in results) / len(results),
            sum(item["metrics"][k]["mrr"] for item in results) / len(results),
            sum(item["metrics"][k]["map"] for item in results) / len(results),
            sum(item["metrics"][k]["ndcg"] for item in results) / len(results),
        ]
        row.extend(f"{value:.3f}" for value in values)
        table.add_row(*row)

    console.print(table)


def print_worst_questions(results: list[dict]) -> None:
    ranked = sorted(results, key=lambda item: item["avg_f1"])
    worst = ranked[:3]

    table = Table(title="Three worst questions (by mean F1@K)", show_lines=True)
    table.add_column("Question")
    table.add_column("Avg F1@K")
    table.add_column("Best K")
    table.add_column("Search query")

    for item in worst:
        best_k = min(KS, key=lambda k: item["metrics"][k]["f1"])
        table.add_row(
            item["question"][:110],
            f"{item['avg_f1']:.3f}",
            str(best_k),
            item["search_query"][:80],
        )

    console.print(table)


def main():
    eval_set = load_eval_set(EVAL_PATH)
    results = []

    for item in eval_set:
        question = item["question"]
        relevant = item["relevant"]
        metrics, avg_f1, search_query = compute_metrics_for_question(question, relevant, KS)
        results.append({
            "question": question,
            "relevant": relevant,
            "metrics": metrics,
            "avg_f1": avg_f1,
            "search_query": search_query,
        })

    print_metrics_table(results)
    print_worst_questions(results)


if __name__ == "__main__":
    main()
