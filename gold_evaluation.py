from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from json4ai import build_structured_output, validate_structured_output
from tfidf import (
    build_chunk_corpus,
    build_tfidf_index,
    build_word2vec_index,
    load_processed_documents,
    rank_chunks,
    save_json,
)


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if value is None:
        return ""
    return str(value)


def normalize_for_compare(text: str) -> str:
    return " ".join(flatten_text(text).lower().split()).strip()


def token_f1(predicted: str, gold: str) -> dict[str, float]:
    pred_tokens = normalize_for_compare(predicted).split()
    gold_tokens = normalize_for_compare(gold).split()

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def string_similarity(predicted: str, gold: str) -> float:
    return round(SequenceMatcher(None, normalize_for_compare(predicted), normalize_for_compare(gold)).ratio(), 4)


def jaccard_list_similarity(predicted: list[str], gold: list[str]) -> float:
    pred_set = {normalize_for_compare(item) for item in predicted if normalize_for_compare(item)}
    gold_set = {normalize_for_compare(item) for item in gold if normalize_for_compare(item)}
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    return round(len(pred_set & gold_set) / len(pred_set | gold_set), 4)


def compare_structured_output(predicted: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    summary_metrics = token_f1(predicted.get("answer_summary", ""), gold.get("answer_summary", ""))
    summary_similarity = string_similarity(predicted.get("answer_summary", ""), gold.get("answer_summary", ""))

    source_title_similarity = jaccard_list_similarity(
        predicted.get("source_titles", []),
        gold.get("source_titles", []),
    )
    key_term_similarity = jaccard_list_similarity(
        predicted.get("key_terms", []),
        gold.get("key_terms", []),
    )

    evidence_similarity = token_f1(
        " ".join(item.get("snippet", "") for item in predicted.get("evidence", [])),
        " ".join(item.get("snippet", "") for item in gold.get("evidence", [])),
    )

    return {
        "summary_token_metrics": summary_metrics,
        "summary_string_similarity": summary_similarity,
        "source_title_jaccard": source_title_similarity,
        "key_term_jaccard": key_term_similarity,
        "evidence_token_metrics": evidence_similarity,
    }


def generate_prediction(
    query: str,
    corpus: list[dict[str, Any]],
    vectorizer,
    matrix,
    top_k: int,
    ranking_method: str,
    word2vec_model,
    word2vec_matrix,
    tfidf_weight: float,
    word2vec_weight: float,
) -> dict[str, Any]:
    results = rank_chunks(
        query,
        vectorizer=vectorizer,
        matrix=matrix,
        corpus=corpus,
        top_k=top_k,
        ranking_method=ranking_method,
        word2vec_model=word2vec_model,
        word2vec_matrix=word2vec_matrix,
        tfidf_weight=tfidf_weight,
        word2vec_weight=word2vec_weight,
    )
    output = build_structured_output(query, results, top_k)
    validate_structured_output(output)
    return output


def aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_query:
        return {}

    def avg(path: list[str]) -> float:
        values = per_query
        extracted = []
        for item in values:
            current = item
            for part in path:
                current = current[part]
            extracted.append(float(current))
        return round(sum(extracted) / len(extracted), 4)

    return {
        "num_queries": len(per_query),
        "schema_validity_rate": round(
            sum(1 for item in per_query if item["schema_valid"]) / len(per_query),
            4,
        ),
        "avg_summary_precision": avg(["comparison", "summary_token_metrics", "precision"]),
        "avg_summary_recall": avg(["comparison", "summary_token_metrics", "recall"]),
        "avg_summary_f1": avg(["comparison", "summary_token_metrics", "f1"]),
        "avg_summary_string_similarity": avg(["comparison", "summary_string_similarity"]),
        "avg_source_title_jaccard": avg(["comparison", "source_title_jaccard"]),
        "avg_key_term_jaccard": avg(["comparison", "key_term_jaccard"]),
        "avg_evidence_precision": avg(["comparison", "evidence_token_metrics", "precision"]),
        "avg_evidence_recall": avg(["comparison", "evidence_token_metrics", "recall"]),
        "avg_evidence_f1": avg(["comparison", "evidence_token_metrics", "f1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated structured JSON against gold JSON references.")
    parser.add_argument("--gold_file", required=True, help="JSON file containing gold examples.")
    parser.add_argument("--processed_dir", default="processed_data", help="Folder containing preprocessing JSON outputs.")
    parser.add_argument("--output_file", default="tfidf_output/gold_evaluation_results.json", help="Where to save evaluation results.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of evidence chunks to use.")
    parser.add_argument("--max_features", type=int, default=10000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--ngram_min", type=int, default=1, help="Minimum n-gram size.")
    parser.add_argument("--ngram_max", type=int, default=2, help="Maximum n-gram size.")
    parser.add_argument("--ranking_method", default="tfidf", choices=["tfidf", "word2vec", "hybrid"], help="Ranking signal to use.")
    parser.add_argument("--tfidf_weight", type=float, default=0.7, help="Hybrid ranking weight for TF-IDF.")
    parser.add_argument("--word2vec_weight", type=float, default=0.3, help="Hybrid ranking weight for Word2Vec.")
    parser.add_argument("--word2vec_vector_size", type=int, default=100, help="Word2Vec embedding size.")
    parser.add_argument("--word2vec_window", type=int, default=5, help="Word2Vec context window size.")
    parser.add_argument("--word2vec_epochs", type=int, default=30, help="Word2Vec training epochs.")
    args = parser.parse_args()

    docs = load_processed_documents(args.processed_dir)
    corpus = build_chunk_corpus(docs)
    vectorizer, matrix = build_tfidf_index(
        corpus,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )

    word2vec_model = None
    word2vec_matrix = None
    if args.ranking_method in {"word2vec", "hybrid"}:
        word2vec_model, word2vec_matrix = build_word2vec_index(
            corpus,
            vector_size=args.word2vec_vector_size,
            window=args.word2vec_window,
            epochs=args.word2vec_epochs,
        )

    with Path(args.gold_file).open("r", encoding="utf-8") as f:
        gold_examples = json.load(f)

    per_query = []
    for example in gold_examples:
        query = example["query"]
        gold_output = example["gold_output"]

        schema_valid = True
        try:
            predicted_output = generate_prediction(
                query,
                corpus,
                vectorizer,
                matrix,
                args.top_k,
                args.ranking_method,
                word2vec_model,
                word2vec_matrix,
                args.tfidf_weight,
                args.word2vec_weight,
            )
        except Exception as exc:
            schema_valid = False
            predicted_output = {
                "query": query,
                "answer_summary": "",
                "source_titles": [],
                "key_terms": [],
                "metadata": {"error": str(exc)},
                "evidence": [],
            }

        comparison = compare_structured_output(predicted_output, gold_output)
        per_query.append({
            "query": query,
            "schema_valid": schema_valid,
            "comparison": comparison,
            "predicted_output": predicted_output,
            "gold_output": gold_output,
        })

    output = {
        "aggregate": aggregate_metrics(per_query),
        "queries": per_query,
    }
    save_json(output, args.output_file)
    print(json.dumps(output["aggregate"], indent=2))
    print(f"\nSaved gold evaluation results to: {args.output_file}")


if __name__ == "__main__":
    main()
