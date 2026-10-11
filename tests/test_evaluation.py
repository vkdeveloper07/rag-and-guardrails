import math

from fitness_agent.evaluation.metrics import (
    average_precision,
    f1_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_example():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a": 2, "c": 1, "e": 3}

    assert precision_at_k(retrieved, relevant, 3) == 2 / 3


def test_recall_at_k_example():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a": 2, "c": 1, "e": 3}

    assert recall_at_k(retrieved, relevant, 3) == 2 / 3


def test_f1_at_k_example():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a": 2, "c": 1, "e": 3}

    assert f1_at_k(retrieved, relevant, 3) == 2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3))


def test_mrr_example():
    retrieved = ["b", "a", "c"]
    relevant = {"a": 2, "d": 1}

    assert mrr(retrieved, relevant) == 1 / 2


def test_average_precision_example():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 2, "c": 1}

    # hits at ranks 1 and 3 => (1/1 + 2/3) / 2 = 5/6
    assert math.isclose(average_precision(retrieved, relevant, 3), 5 / 6)


def test_map_at_k_example():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 2, "c": 1}

    assert math.isclose(map_at_k(retrieved, relevant, 3), 5 / 6)


def test_ndcg_at_k_example():
    retrieved = ["a", "b", "c"]
    relevant = {"a": 2, "c": 1}

    dcg = 2 / math.log2(2) + 1 / math.log2(4)
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert math.isclose(ndcg_at_k(retrieved, relevant, 3), dcg / idcg)


def test_edge_cases_no_relevant_results():
    retrieved = ["x", "y"]
    relevant = {"a": 2, "b": 1}

    assert precision_at_k(retrieved, relevant, 2) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.0
    assert f1_at_k(retrieved, relevant, 2) == 0.0
    assert mrr(retrieved, relevant) == 0.0
    assert average_precision(retrieved, relevant, 2) == 0.0
    assert ndcg_at_k(retrieved, relevant, 2) == 0.0


def test_edge_cases_no_results():
    assert precision_at_k([], {"a": 2}, 3) == 0.0
    assert recall_at_k([], {"a": 2}, 3) == 0.0
    assert f1_at_k([], {"a": 2}, 3) == 0.0
    assert mrr([], {"a": 2}) == 0.0
    assert average_precision([], {"a": 2}, 3) == 0.0
    assert ndcg_at_k([], {"a": 2}, 3) == 0.0
