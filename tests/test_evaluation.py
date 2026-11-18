from scripts.evaluate_rag import run_offline_benchmark


def test_offline_rag_evaluation_metrics() -> None:
    summary = run_offline_benchmark()
    assert summary.total_cases == 4
    assert summary.recall_at_k == 1.0
    assert summary.mrr == 1.0
    assert summary.avg_keyword_coverage == 1.0
    assert summary.citation_accuracy == 1.0
    assert len(summary.results) == 4
