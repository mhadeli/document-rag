import argparse
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.db.repositories import ChunkRepository, DocumentRepository
from app.rag.embeddings import EmbeddingService
from app.rag.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    StructuredLLMResponse,
)
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import RetrievedChunk, VectorRetriever


@dataclass(slots=True)
class EvalCase:
    question: str
    expected_keywords: list[str]
    expected_filename: str | None = None
    expected_page: int | None = None
    should_abstain: bool = False


@dataclass(slots=True)
class CaseResult:
    question: str
    recall_hit: bool
    reciprocal_rank: float
    keyword_coverage: float
    citation_accurate: bool
    latency_ms: float


@dataclass(slots=True)
class EvaluationSummary:
    total_cases: int
    recall_at_k: float
    mrr: float
    avg_keyword_coverage: float
    citation_accuracy: float
    avg_latency_ms: float
    results: list[CaseResult] = field(default_factory=list)


DEFAULT_EVAL_CASES: list[EvalCase] = [
    EvalCase(
        question="By what percentage did annual revenue grow in 2026?",
        expected_keywords=["24%", "$120M"],
        expected_filename="annual_report_2026.pdf",
        expected_page=1,
    ),
    EvalCase(
        question="What database extension is used for vector similarity search?",
        expected_keywords=["pgvector", "PostgreSQL"],
        expected_filename="Architecture Guide",
        expected_page=None,
    ),
    EvalCase(
        question="What is the maximum allowed PDF upload size?",
        expected_keywords=["20 MB"],
        expected_filename="annual_report_2026.pdf",
        expected_page=2,
    ),
    EvalCase(
        question="What is the company's cryptocurrency mining budget for 2027?",
        expected_keywords=["not contain", "insufficient", "could not find"],
        expected_filename=None,
        expected_page=None,
        should_abstain=True,
    ),
]


def _is_chunk_relevant(chunk: RetrievedChunk, case: EvalCase) -> bool:
    if case.should_abstain:
        return False
    if case.expected_filename and chunk.filename != case.expected_filename:
        return False
    if case.expected_page is not None and chunk.page_number != case.expected_page:
        return False
    content_lower = chunk.content.lower()
    return any(kw.lower() in content_lower for kw in case.expected_keywords)


def compute_recall_and_rr(
    chunks: list[RetrievedChunk],
    case: EvalCase,
) -> tuple[bool, float]:
    if case.should_abstain:
        return True, 1.0

    for rank, chunk in enumerate(chunks, start=1):
        if _is_chunk_relevant(chunk, case):
            return True, 1.0 / rank

    return False, 0.0


def compute_keyword_coverage(answer: str, case: EvalCase) -> float:
    if not case.expected_keywords:
        return 1.0
    lower_ans = answer.lower()
    if case.should_abstain:
        return 1.0 if any(kw.lower() in lower_ans for kw in case.expected_keywords) else 0.0

    hits = sum(1 for kw in case.expected_keywords if kw.lower() in lower_ans)
    return hits / len(case.expected_keywords)


def check_citation_accuracy(generated: GeneratedAnswer, case: EvalCase) -> bool:
    if case.should_abstain:
        return len(generated.sources) == 0

    for src in generated.sources:
        if case.expected_filename and src.filename != case.expected_filename:
            continue
        if case.expected_page is not None and src.page_number != case.expected_page:
            continue
        return True
    return False


def evaluate_cases(
    pipeline: RAGPipeline,
    cases: list[EvalCase],
    *,
    top_k: int = 5,
) -> EvaluationSummary:
    results: list[CaseResult] = []

    for case in cases:
        t0 = time.perf_counter()
        retrieved = pipeline.retriever.retrieve(case.question, top_k=top_k)
        generated = pipeline.generator.generate_answer(case.question, retrieved)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        recall_hit, rr = compute_recall_and_rr(retrieved, case)
        kw_cov = compute_keyword_coverage(generated.answer, case)
        cit_ok = check_citation_accuracy(generated, case)

        results.append(
            CaseResult(
                question=case.question,
                recall_hit=recall_hit,
                reciprocal_rank=rr,
                keyword_coverage=kw_cov,
                citation_accurate=cit_ok,
                latency_ms=round(elapsed_ms, 2),
            )
        )

    n = len(results) or 1
    return EvaluationSummary(
        total_cases=len(results),
        recall_at_k=round(sum(1 for r in results if r.recall_hit) / n, 4),
        mrr=round(sum(r.reciprocal_rank for r in results) / n, 4),
        avg_keyword_coverage=round(sum(r.keyword_coverage for r in results) / n, 4),
        citation_accuracy=round(sum(1 for r in results if r.citation_accurate) / n, 4),
        avg_latency_ms=round(sum(r.latency_ms for r in results) / n, 2),
        results=results,
    )


def _unit_vec(index: int, dim: int = 768) -> list[float]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


class BenchmarkEmbeddingClient:
    def embed_query(self, text: str, **_kwargs: Any) -> list[float]:
        lower = text.lower()
        if "revenue" in lower:
            return _unit_vec(0)
        if "vector" in lower or "database" in lower:
            return _unit_vec(1)
        if "upload" in lower or "size" in lower:
            return _unit_vec(2)
        return _unit_vec(3)


class BenchmarkLLM:
    def with_structured_output(self, _schema: Any) -> Any:
        class _Runner:
            def invoke(self, messages: list[Any]) -> StructuredLLMResponse:
                import re

                prompt = str(messages[-1].content)
                question_part = prompt.split("User Question:")[-1].lower()
                chunk_ids = re.findall(r"\[chunk_id:\s*([0-9a-fA-F-]+)\]", prompt)
                if "cryptocurrency" in question_part:
                    return StructuredLLMResponse(
                        answer="The indexed documents do not contain sufficient information.",
                        cited_chunk_ids=[],
                    )
                if "revenue" in question_part:
                    return StructuredLLMResponse(
                        answer="Annual revenue grew by 24% to $120M in 2026.",
                        cited_chunk_ids=chunk_ids[:1],
                    )
                if "database" in question_part or "vector" in question_part:
                    return StructuredLLMResponse(
                        answer="PostgreSQL with the pgvector extension is used for vector search.",
                        cited_chunk_ids=chunk_ids[:1],
                    )
                return StructuredLLMResponse(
                    answer="The maximum allowed PDF upload size is 20 MB.",
                    cited_chunk_ids=chunk_ids[:1],
                )

        return _Runner()


def run_offline_benchmark() -> EvaluationSummary:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    db: Session = session_factory()

    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    pdf_doc = doc_repo.create(
        filename="annual_report_2026.pdf",
        source_type="pdf",
        content_hash="eval_hash_1",
        status="completed",
    )
    web_doc = doc_repo.create(
        filename="Architecture Guide",
        source_type="website",
        source_url="https://example.com/arch",
        content_hash="eval_hash_2",
        status="completed",
    )

    chunk_repo.bulk_create(
        pdf_doc.id,
        [
            {
                "chunk_index": 0,
                "content": "In 2026, annual revenue grew by 24% reaching $120M.",
                "embedding": _unit_vec(0),
                "page_number": 1,
            },
            {
                "chunk_index": 1,
                "content": "System configuration limits PDF uploads to a maximum of 20 MB.",
                "embedding": _unit_vec(2),
                "page_number": 2,
            },
        ],
    )
    chunk_repo.bulk_create(
        web_doc.id,
        [
            {
                "chunk_index": 0,
                "content": "We use PostgreSQL with the pgvector extension for similarity search.",
                "embedding": _unit_vec(1),
                "source_url": "https://example.com/arch",
            }
        ],
    )

    emb_service = EmbeddingService(dimensions=768, client=BenchmarkEmbeddingClient())
    pipeline = RAGPipeline(
        db=db,
        retriever=VectorRetriever(db, embedding_service=emb_service),
        generator=AnswerGenerator(llm=BenchmarkLLM()),
    )

    summary = evaluate_cases(pipeline, DEFAULT_EVAL_CASES, top_k=3)
    db.close()
    return summary


def check_thresholds(
    summary: EvaluationSummary,
    *,
    min_recall: float,
    min_mrr: float,
    min_keyword_coverage: float,
    min_citation_accuracy: float,
) -> list[str]:
    checks = [
        ("Recall@k", summary.recall_at_k, min_recall),
        ("MRR", summary.mrr, min_mrr),
        ("Keyword Coverage", summary.avg_keyword_coverage, min_keyword_coverage),
        ("Citation Accuracy", summary.citation_accuracy, min_citation_accuracy),
    ]
    return [
        f"{name} {value:.4f} is below threshold {threshold:.4f}"
        for name, value, threshold in checks
        if value < threshold
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and generation metrics.")
    parser.add_argument("--min-recall", type=float, default=1.0)
    parser.add_argument("--min-mrr", type=float, default=1.0)
    parser.add_argument("--min-keyword-coverage", type=float, default=1.0)
    parser.add_argument("--min-citation-accuracy", type=float, default=1.0)
    args = parser.parse_args()

    summary = run_offline_benchmark()
    print("=== RAG Evaluation Summary ===")
    print(f"Total Cases          : {summary.total_cases}")
    print(f"Recall@3             : {summary.recall_at_k:.2%}")
    print(f"MRR                  : {summary.mrr:.4f}")
    print(f"Keyword Coverage     : {summary.avg_keyword_coverage:.2%}")
    print(f"Citation Accuracy    : {summary.citation_accuracy:.2%}")
    print(f"Average Latency (ms) : {summary.avg_latency_ms:.2f}")

    failures = check_thresholds(
        summary,
        min_recall=args.min_recall,
        min_mrr=args.min_mrr,
        min_keyword_coverage=args.min_keyword_coverage,
        min_citation_accuracy=args.min_citation_accuracy,
    )
    if failures:
        print("\n=== Evaluation FAILED ===")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nAll metrics meet thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
