from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import math

import pandas as pd


Query = Dict[str, Any]
Qrels = Dict[Any, Dict[Any, float]]
SearchResult = List[Tuple[Any, float]]
RelevanceMode = Literal["direct", "inverse", "binary"]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _extract_doc_ids(
    retrieved_docs: Sequence[Any],
    doc_id_normalizer: Optional[Callable[[Any], Any]] = None,
) -> List[Any]:
    """
    Extract document ids from supported retrieval result formats:
    - doc_id
    - (doc_id, score)
    - {"doc_id": doc_id, ...}
    """
    doc_ids: List[Any] = []
    for item in retrieved_docs:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            doc_id = item[0]
        elif isinstance(item, dict) and "doc_id" in item:
            doc_id = item["doc_id"]
        else:
            doc_id = item

        if doc_id_normalizer is not None:
            doc_id = doc_id_normalizer(doc_id)
        doc_ids.append(doc_id)
    return doc_ids


def transform_relevance(
    rel: float,
    *,
    mode: RelevanceMode = "direct",
    max_rel: Optional[float] = None,
) -> float:
    """
    Transform raw qrels into gain values used by NDCG and relevance filtering.

    mode="direct": larger qrel values mean more relevant. Example: 0, 1, 2.
    mode="inverse": smaller positive qrel values mean more relevant. Example: 1 = most relevant, 2 = partially relevant.
    mode="binary": every positive qrel value becomes 1.0.
    """
    if rel <= 0:
        return 0.0

    if mode == "direct":
        return float(rel)
    if mode == "binary":
        return 1.0
    if mode == "inverse":
        if max_rel is None:
            raise ValueError("max_rel is required when relevance_mode='inverse'.")
        return float(max_rel - rel + 1.0)

    raise ValueError(f"Unsupported relevance mode: {mode}")


def normalize_relevance(rel: float, max_rel: Optional[float] = None) -> float:
    """
    Backward-compatible helper.

    Important: this no longer reverses relevance order. It simply clips negative
    relevance to zero. Use transform_relevance(..., mode='inverse') explicitly
    when your qrels encode smaller positive values as more relevant.
    """
    return max(float(rel), 0.0)


def get_relevant_set(
    qrels: Qrels,
    query_id: Any,
    *,
    relevance_mode: RelevanceMode = "direct",
) -> Dict[Any, float]:
    rels = qrels.get(query_id, {})
    max_rel = max((rel for rel in rels.values() if rel > 0), default=0.0)
    relevant: Dict[Any, float] = {}
    for doc_id, rel in rels.items():
        gain = transform_relevance(rel, mode=relevance_mode, max_rel=max_rel)
        if gain > 0:
            relevant[doc_id] = gain
    return relevant


def precision_recall_f1_at_k(
    retrieved_doc_ids: Sequence[Any],
    relevant_doc_ids: Iterable[Any],
    k: int,
) -> Tuple[float, float, float]:
    top_k = retrieved_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)
    hit_count = sum(1 for doc_id in top_k if doc_id in relevant_set)
    precision = _safe_divide(hit_count, len(top_k))
    recall = _safe_divide(hit_count, len(relevant_set))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def hit_at_k(
    retrieved_doc_ids: Sequence[Any],
    relevant_doc_ids: Iterable[Any],
    k: int,
) -> float:
    relevant_set = set(relevant_doc_ids)
    return float(any(doc_id in relevant_set for doc_id in retrieved_doc_ids[:k]))


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[Any],
    relevant_doc_ids: Iterable[Any],
    k: int,
) -> float:
    relevant_set = set(relevant_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(
    retrieved_doc_ids: Sequence[Any],
    relevant_doc_ids: Iterable[Any],
    k: int,
) -> float:
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    ap_sum = 0.0
    hit_count = 0
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            hit_count += 1
            ap_sum += _safe_divide(hit_count, rank)

    # AP@k convention: denominator is min(number of relevant docs, k).
    return _safe_divide(ap_sum, min(len(relevant_set), k))


def ndcg_at_k(
    retrieved_doc_ids: Sequence[Any],
    qrels_for_query: Dict[Any, float],
    k: int,
    relevance_mode: RelevanceMode = "direct",
) -> float:
    if not qrels_for_query:
        return 0.0

    max_rel = max((rel for rel in qrels_for_query.values() if rel > 0), default=0.0)
    rel_map = {
        doc_id: transform_relevance(rel, mode=relevance_mode, max_rel=max_rel)
        for doc_id, rel in qrels_for_query.items()
    }

    def dcg(rels: Sequence[float]) -> float:
        score = 0.0
        for rank, rel in enumerate(rels, start=1):
            if rel <= 0:
                continue
            score += (2.0 ** rel - 1.0) / math.log2(rank + 1)
        return score

    retrieved_rels = [rel_map.get(doc_id, 0.0) for doc_id in retrieved_doc_ids[:k]]
    dcg_value = dcg(retrieved_rels)

    ideal_rels = sorted(rel_map.values(), reverse=True)[:k]
    idcg_value = dcg(ideal_rels)
    return _safe_divide(dcg_value, idcg_value)


def precision_recall_curve_data(
    retrieved_doc_ids: Sequence[Any],
    relevant_doc_ids: Iterable[Any],
    k_max: int,
) -> List[Dict[str, float]]:
    relevant_set = set(relevant_doc_ids)
    curve: List[Dict[str, float]] = []
    for k in range(1, k_max + 1):
        precision, recall, _ = precision_recall_f1_at_k(retrieved_doc_ids, relevant_set, k)
        curve.append({"k": k, "precision": precision, "recall": recall})
    return curve


@dataclass
class EvaluationResult:
    summary: Dict[str, float]
    per_query: Dict[Any, Dict[str, float]]
    retrieval_results: Optional[Dict[Any, SearchResult]] = None
    summary_df: Optional[pd.DataFrame] = None
    per_query_df: Optional[pd.DataFrame] = None


class Evaluator:
    def __init__(
        self,
        queries: Sequence[Query],
        qrels: Qrels,
        retriever: Any,
        relevance_mode: RelevanceMode = "direct",
        normalize_relevance_scores: Optional[bool] = None,
        query_id_normalizer: Optional[Callable[[Any], Any]] = None,
        doc_id_normalizer: Optional[Callable[[Any], Any]] = None,
        id_normalizer: Optional[Callable[[Any], Any]] = None,
        silent_errors: bool = False,
    ) -> None:
        """
        Args:
            queries: sequence of dicts with at least {'id', 'content'}.
            qrels: nested dict {query_id: {doc_id: relevance}}.
            retriever: object exposing search(query_text, top_k=...).
            relevance_mode: 'direct', 'inverse', or 'binary'.
            normalize_relevance_scores: deprecated compatibility argument.
            query_id_normalizer: optional normalizer for query ids.
            doc_id_normalizer: optional normalizer for document ids.
            id_normalizer: backward-compatible alias for query_id_normalizer.
            silent_errors: if False, retrieval exceptions are raised with query context.
        """
        if normalize_relevance_scores is not None:
            # Compatibility: previous True meant transforming relevance for NDCG.
            # The safe default is still direct ordering.
            relevance_mode = relevance_mode

        if id_normalizer is not None and query_id_normalizer is None:
            query_id_normalizer = id_normalizer

        self.queries = list(queries)
        self.retriever = retriever
        self.relevance_mode = relevance_mode
        self.query_id_normalizer = query_id_normalizer
        self.doc_id_normalizer = doc_id_normalizer
        self.silent_errors = silent_errors
        self.qrels = self._normalize_qrels(qrels)

    def _normalize_query_id(self, value: Any) -> Any:
        if self.query_id_normalizer is None:
            return value
        return self.query_id_normalizer(value)

    def _normalize_doc_id(self, value: Any) -> Any:
        if self.doc_id_normalizer is None:
            return value
        return self.doc_id_normalizer(value)

    def _normalize_qrels(self, qrels: Qrels) -> Qrels:
        normalized: Qrels = {}
        for query_id, doc_rels in qrels.items():
            qid = self._normalize_query_id(query_id)
            normalized[qid] = {}
            for doc_id, rel in doc_rels.items():
                did = self._normalize_doc_id(doc_id)
                normalized[qid][did] = float(rel)
        return normalized

    def run_retrieval(self, top_k: int) -> Dict[Any, SearchResult]:
        results: Dict[Any, SearchResult] = {}
        for query in self.queries:
            query_id = self._normalize_query_id(query.get("id"))
            query_text = query.get("content", "")
            try:
                retrieved = self.retriever.search(query_text, top_k=top_k)
            except Exception as exc:
                if not self.silent_errors:
                    raise RuntimeError(
                        f"Retrieval failed for query_id={query_id!r}, query_text={query_text[:120]!r}"
                    ) from exc
                retrieved = []
            results[query_id] = list(retrieved) if retrieved is not None else []
        return results

    def evaluate_query(
        self,
        query_id: Any,
        retrieved_docs: Sequence[Any],
        k: int,
    ) -> Dict[str, float]:
        qrels_for_query = self.qrels.get(query_id, {})
        relevant_docs = get_relevant_set(
            self.qrels,
            query_id,
            relevance_mode=self.relevance_mode,
        )
        retrieved_doc_ids = _extract_doc_ids(retrieved_docs, self.doc_id_normalizer)
        top_k_doc_ids = retrieved_doc_ids[:k]

        precision, recall, f1 = precision_recall_f1_at_k(
            retrieved_doc_ids, relevant_docs.keys(), k
        )
        ap = average_precision_at_k(retrieved_doc_ids, relevant_docs.keys(), k)
        ndcg = ndcg_at_k(
            retrieved_doc_ids,
            qrels_for_query,
            k,
            relevance_mode=self.relevance_mode,
        )
        hit = hit_at_k(retrieved_doc_ids, relevant_docs.keys(), k)
        rr = reciprocal_rank_at_k(retrieved_doc_ids, relevant_docs.keys(), k)
        hit_count = sum(1 for doc_id in top_k_doc_ids if doc_id in relevant_docs)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "ndcg": ndcg,
            "hit": hit,
            "rr": rr,
            "relevant_count": float(len(relevant_docs)),
            "retrieved_count": float(len(top_k_doc_ids)),
            "hit_count": float(hit_count),
            "has_relevance": float(len(relevant_docs) > 0),
        }

    def evaluate_all(
        self,
        top_k: int,
        return_results: bool = False,
        return_df: bool = True,
    ) -> EvaluationResult:
        retrieval_results = self.run_retrieval(top_k)
        per_query: Dict[Any, Dict[str, float]] = {}

        for query_id, retrieved_docs in retrieval_results.items():
            per_query[query_id] = self.evaluate_query(query_id, retrieved_docs, top_k)

        metric_names = [
            "precision",
            "recall",
            "f1",
            "ap",
            "ndcg",
        ]

        summary: Dict[str, float] = {}
        if per_query:
            for metric in metric_names:
                avg_value = sum(m[metric] for m in per_query.values()) / len(per_query)
                if metric == "ap":
                    summary[f"MAP@{top_k}"] = avg_value
                elif metric == "rr":
                    summary[f"MRR@{top_k}"] = avg_value
                elif metric == "hit":
                    summary[f"Hit@{top_k}"] = avg_value
                else:
                    summary[f"{metric.title()}@{top_k}"] = avg_value
        else:
            summary = {
                f"Precision@{top_k}": 0.0,
                f"Recall@{top_k}": 0.0,
                f"F1@{top_k}": 0.0,
                f"MAP@{top_k}": 0.0,
                f"NDCG@{top_k}": 0.0,
                f"Hit@{top_k}": 0.0,
                f"MRR@{top_k}": 0.0,
            }

        summary_df = None
        per_query_df = None
        if return_df:
            per_query_df = pd.DataFrame.from_dict(per_query, orient="index")
            per_query_df.index.name = "query_id"
            summary_df = pd.DataFrame([summary])

        return EvaluationResult(
            summary=summary,
            per_query=per_query,
            retrieval_results=retrieval_results if return_results else None,
            summary_df=summary_df,
            per_query_df=per_query_df,
        )


def evaluate_many(
    models: Dict[str, Any],
    queries: Sequence[Query],
    qrels: Qrels,
    top_k: int,
    relevance_mode: RelevanceMode = "direct",
    normalize_relevance_scores: Optional[bool] = None,
    query_id_normalizer: Optional[Callable[[Any], Any]] = None,
    doc_id_normalizer: Optional[Callable[[Any], Any]] = None,
    id_normalizer: Optional[Callable[[Any], Any]] = None,
    return_results: bool = False,
    return_df: bool = True,
    silent_errors: bool = False,
) -> Dict[str, EvaluationResult]:
    results: Dict[str, EvaluationResult] = {}
    for name, retriever in models.items():
        evaluator = Evaluator(
            queries=queries,
            qrels=qrels,
            retriever=retriever,
            relevance_mode=relevance_mode,
            normalize_relevance_scores=normalize_relevance_scores,
            query_id_normalizer=query_id_normalizer,
            doc_id_normalizer=doc_id_normalizer,
            id_normalizer=id_normalizer,
            silent_errors=silent_errors,
        )
        results[name] = evaluator.evaluate_all(
            top_k=top_k,
            return_results=return_results,
            return_df=return_df,
        )
    return results
