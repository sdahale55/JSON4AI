from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tfidf import (
    build_chunk_corpus,
    build_tfidf_index,
    build_word2vec_index,
    load_processed_documents,
    normalize_space,
    rank_chunks,
    save_json,
)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with"
}


def extract_query_terms(query: str) -> list[str]:
    terms = [token.lower() for token in TOKEN_RE.findall(query)]
    return [term for term in terms if term not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    sentences = []
    for part in SENTENCE_SPLIT_RE.split(text):
        sentence = normalize_space(part)
        if not sentence:
            continue
        if len(sentence.split()) < 8:
            continue
        if not sentence[0].isalnum() or sentence[0].islower():
            continue
        if sentence[-1] not in ".!?":
            continue
        sentences.append(sentence)
    return sentences


def score_sentence(sentence: str, query_terms: list[str]) -> tuple[int, int]:
    lowered = sentence.lower()
    overlap = sum(1 for term in query_terms if term in lowered)
    return overlap, len(sentence)


def build_answer_summary(results: list[dict[str, Any]], query_terms: list[str], max_sentences: int = 3) -> str:
    candidates: list[tuple[int, int, str]] = []
    seen_sentences: set[str] = set()

    for result in results:
        for sentence in split_sentences(result["text"]):
            if sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            overlap, length = score_sentence(sentence, query_terms)
            if overlap == 0:
                continue
            candidates.append((overlap, length, sentence))

    if candidates:
        max_overlap = max(item[0] for item in candidates)
        if max_overlap > 1:
            candidates = [item for item in candidates if item[0] == max_overlap]

    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected: list[str] = []
    selected_lower: set[str] = set()
    for _, _, sentence in candidates:
        key = sentence.lower()
        if key in selected_lower:
            continue
        selected.append(sentence)
        selected_lower.add(key)
        if len(selected) == max_sentences:
            break

    if selected:
        return " ".join(selected)

    fallback = [result["text"] for result in results[:1]]
    return normalize_space(" ".join(fallback))


def build_structured_output(query: str, results: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    query_terms = extract_query_terms(query)
    answer_summary = build_answer_summary(results, query_terms)
    source_titles = []
    for result in results:
        if result["title"] not in source_titles:
            source_titles.append(result["title"])

    evidence = [
        {
            "rank": result["rank"],
            "score": round(result["score"], 4),
            "title": result["title"],
            "source_file": result["source_file"],
            "heading": result["heading"],
            "chunk_id": result["chunk_id"],
            "word_count": result["word_count"],
            "snippet": result["text"]
        }
        for result in results
    ]

    return {
        "query": query,
        "answer_summary": answer_summary,
        "source_titles": source_titles,
        "key_terms": query_terms,
        "metadata": {
            "top_k": top_k,
            "num_evidence_chunks": len(results),
            "generator": "json4ai-heuristic-structured-output"
        },
        "evidence": evidence
    }


def validate_structured_output(output: dict[str, Any]) -> None:
    required_top_level = {
        "query": str,
        "answer_summary": str,
        "source_titles": list,
        "key_terms": list,
        "metadata": dict,
        "evidence": list
    }
    for key, expected_type in required_top_level.items():
        value = output.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"Invalid structured output: {key} must be {expected_type.__name__}")

    for evidence_item in output["evidence"]:
        for field in ["rank", "score", "title", "source_file", "heading", "chunk_id", "word_count", "snippet"]:
            if field not in evidence_item:
                raise ValueError(f"Invalid structured output: missing evidence field {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate query-focused structured JSON from the top TF-IDF evidence chunks.")
    parser.add_argument("--processed_dir", default="processed_data", help="Folder containing preprocessing JSON outputs.")
    parser.add_argument("--output_file", default="tfidf_output/structured_output.json", help="File to save the structured JSON response.")
    parser.add_argument("--query", required=True, help="Query to answer with structured JSON.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top evidence chunks to include.")
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
        ngram_max=args.ngram_max
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
    results = rank_chunks(
        args.query,
        vectorizer,
        matrix,
        corpus,
        top_k=args.top_k,
        ranking_method=args.ranking_method,
        word2vec_model=word2vec_model,
        word2vec_matrix=word2vec_matrix,
        tfidf_weight=args.tfidf_weight,
        word2vec_weight=args.word2vec_weight,
    )
    structured_output = build_structured_output(args.query, results, args.top_k)
    validate_structured_output(structured_output)

    save_json(structured_output, Path(args.output_file))
    print(json.dumps(structured_output, indent=2, ensure_ascii=False))
    print(f"\nSaved structured output to: {args.output_file}")


if __name__ == "__main__":
    main()
