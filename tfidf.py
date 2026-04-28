from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from gensim.models import Word2Vec
except Exception:  # pragma: no cover - optional dependency at runtime
    Word2Vec = None


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with"
}


def normalize_space(text: str) -> str:
    return " ".join(text.split()).strip()


def chunk_fallback(text: str, max_words: int = 180, overlap: int = 30) -> list[str]:
    words = normalize_space(text).split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize_for_embeddings(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(normalize_space(text))
        if token.lower() not in STOPWORDS
    ]


def build_search_text(title: str, heading: str, text: str, title_boost: int = 3, heading_boost: int = 2) -> str:
    title_part = " ".join([normalize_space(title)] * max(title_boost, 0)).strip()
    heading_part = " ".join([normalize_space(heading)] * max(heading_boost, 0)).strip()
    parts = [part for part in [title_part, heading_part, normalize_space(text)] if part]
    return "\n".join(parts)


def load_processed_documents(processed_dir: str | Path) -> list[dict[str, Any]]:
    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_path}")

    files = sorted(
        fp for fp in processed_path.glob("*.json")
        if fp.name != "summary.json"
    )

    docs: list[dict[str, Any]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            docs.append(json.load(f))

    if not docs:
        raise ValueError(f"No processed JSON files found in {processed_path}")

    return docs


def build_chunk_corpus(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corpus: list[dict[str, Any]] = []

    for doc_idx, doc in enumerate(docs):
        source_file = doc.get("source_file", f"doc_{doc_idx}")
        title = doc.get("title", "")
        chunks = doc.get("chunks", [])

        if chunks:
            for local_idx, chunk in enumerate(chunks):
                text = normalize_space(chunk.get("text", ""))
                if not text:
                    continue

                corpus.append({
                    "doc_id": doc_idx,
                    "source_file": source_file,
                    "title": title,
                    "chunk_id": chunk.get("chunk_id", local_idx),
                    "heading": chunk.get("heading", ""),
                    "text": text,
                    "word_count": len(text.split()),
                    "search_text": build_search_text(title, chunk.get("heading", ""), text),
                    "tokens": tokenize_for_embeddings(build_search_text(title, chunk.get("heading", ""), text))
                })
            continue

        # Fallback if preprocessing JSON does not contain chunks
        cleaned_text = doc.get("cleaned_text", "")
        fallback_chunks = chunk_fallback(cleaned_text)
        for local_idx, text in enumerate(fallback_chunks):
            corpus.append({
                "doc_id": doc_idx,
                "source_file": source_file,
                "title": title,
                "chunk_id": local_idx,
                "heading": "",
                "text": text,
                "word_count": len(text.split()),
                "search_text": build_search_text(title, "", text),
                "tokens": tokenize_for_embeddings(build_search_text(title, "", text))
            })

    if not corpus:
        raise ValueError("No usable chunk text found in processed documents.")

    return corpus


def build_tfidf_index(
    corpus: list[dict[str, Any]],
    max_features: int = 10000,
    ngram_min: int = 1,
    ngram_max: int = 2
):
    texts = [item["search_text"] for item in corpus]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        lowercase=True
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_word2vec_index(
    corpus: list[dict[str, Any]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    workers: int = 1,
    epochs: int = 30,
):
    if Word2Vec is None:
        raise ImportError(
            "gensim is required for Word2Vec ranking. Install it with `pip install gensim`."
        )

    sentences = [item["tokens"] for item in corpus if item.get("tokens")]
    if not sentences:
        raise ValueError("No tokenized text available for Word2Vec training.")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,
    )
    embeddings = np.vstack([average_embedding(item["tokens"], model) for item in corpus])
    return model, embeddings


def average_embedding(tokens: list[str], model) -> np.ndarray:
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=float)
    return np.mean(vectors, axis=0)


def safe_cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vec)
    row_norms = np.linalg.norm(matrix, axis=1)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=float)

    denominator = row_norms * query_norm
    numerator = matrix @ query_vec
    scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0
    )
    return scores


def rank_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix,
    corpus: list[dict[str, Any]],
    top_k: int = 5,
    ranking_method: str = "tfidf",
    word2vec_model=None,
    word2vec_matrix: np.ndarray | None = None,
    tfidf_weight: float = 0.7,
    word2vec_weight: float = 0.3,
) -> list[dict[str, Any]]:
    query = normalize_space(query)
    if not query:
        raise ValueError("Query cannot be empty.")

    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, matrix)[0]

    word2vec_scores = None
    if ranking_method in {"word2vec", "hybrid"}:
        if word2vec_model is None or word2vec_matrix is None:
            raise ValueError("Word2Vec ranking requested, but no Word2Vec model/index was provided.")
        query_tokens = tokenize_for_embeddings(query)
        query_embedding = average_embedding(query_tokens, word2vec_model)
        word2vec_scores = safe_cosine_similarity(query_embedding, word2vec_matrix)

    if ranking_method == "tfidf":
        scores = tfidf_scores
    elif ranking_method == "word2vec":
        scores = word2vec_scores
    elif ranking_method == "hybrid":
        scores = (tfidf_weight * tfidf_scores) + (word2vec_weight * word2vec_scores)
    else:
        raise ValueError(f"Unsupported ranking_method: {ranking_method}")

    ranked_indices = scores.argsort()[::-1][:top_k]
    results: list[dict[str, Any]] = []

    for rank, idx in enumerate(ranked_indices, start=1):
        item = corpus[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "tfidf_score": float(tfidf_scores[idx]),
            "word2vec_score": float(word2vec_scores[idx]) if word2vec_scores is not None else None,
            "source_file": item["source_file"],
            "title": item["title"],
            "chunk_id": item["chunk_id"],
            "heading": item["heading"],
            "word_count": item["word_count"],
            "text": item["text"]
        })

    return results


def evaluate_queries(
    queries: list[dict[str, Any]],
    vectorizer: TfidfVectorizer,
    matrix,
    corpus: list[dict[str, Any]],
    top_k: int = 5,
    ranking_method: str = "tfidf",
    word2vec_model=None,
    word2vec_matrix: np.ndarray | None = None,
    tfidf_weight: float = 0.7,
    word2vec_weight: float = 0.3,
) -> dict[str, Any]:
    evaluated_queries: list[dict[str, Any]] = []
    precision_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    hit_sum = 0.0

    for query_entry in queries:
        query = query_entry["query"]
        relevant_docs = set(query_entry.get("relevant_titles", []))
        results = rank_chunks(
            query,
            vectorizer,
            matrix,
            corpus,
            top_k=top_k,
            ranking_method=ranking_method,
            word2vec_model=word2vec_model,
            word2vec_matrix=word2vec_matrix,
            tfidf_weight=tfidf_weight,
            word2vec_weight=word2vec_weight,
        )

        matched_titles: list[str] = []
        first_hit_rank = None
        for result in results:
            if result["title"] in relevant_docs:
                matched_titles.append(result["title"])
                if first_hit_rank is None:
                    first_hit_rank = result["rank"]

        unique_hits = len(set(matched_titles))
        precision = unique_hits / top_k if top_k else 0.0
        recall = unique_hits / len(relevant_docs) if relevant_docs else 0.0
        reciprocal_rank = 1 / first_hit_rank if first_hit_rank else 0.0
        hit_rate = 1.0 if unique_hits else 0.0

        precision_sum += precision
        recall_sum += recall
        mrr_sum += reciprocal_rank
        hit_sum += hit_rate

        evaluated_queries.append({
            "query": query,
            "relevant_titles": sorted(relevant_docs),
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "reciprocal_rank": round(reciprocal_rank, 4),
            "matched_titles": sorted(set(matched_titles)),
            "hits": unique_hits,
            "results": results
        })

    count = len(evaluated_queries)
    aggregate = {
        "num_queries": count,
        "top_k": top_k,
        "ranking_method": ranking_method,
        "mean_precision_at_k": round(precision_sum / count, 4) if count else 0.0,
        "mean_recall_at_k": round(recall_sum / count, 4) if count else 0.0,
        "mrr": round(mrr_sum / count, 4) if count else 0.0,
        "hit_rate": round(hit_sum / count, 4) if count else 0.0
    }

    return {"aggregate": aggregate, "queries": evaluated_queries}


def summarize_corpus(corpus: list[dict[str, Any]], vectorizer: TfidfVectorizer, matrix) -> dict[str, Any]:
    word_counts = [item["word_count"] for item in corpus]
    unique_docs = len({item["source_file"] for item in corpus})
    doc_chunk_counts: dict[str, int] = {}
    for item in corpus:
        doc_chunk_counts[item["title"]] = doc_chunk_counts.get(item["title"], 0) + 1
    densest_docs = sorted(doc_chunk_counts.items(), key=lambda item: (-item[1], item[0]))[:5]

    return {
        "num_documents": unique_docs,
        "num_chunks": len(corpus),
        "vocab_size": len(vectorizer.get_feature_names_out()),
        "tfidf_matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "avg_chunk_words": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0,
        "top_documents_by_chunk_count": [
            {"title": title, "chunk_count": chunk_count} for title, chunk_count in densest_docs
        ]
    }


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a TF-IDF index from preprocessing.py JSON outputs and optionally rank chunks for a query."
    )
    parser.add_argument("--processed_dir", default="processed_data", help="Folder containing preprocessing JSON outputs.")
    parser.add_argument("--output_dir", default="tfidf_output", help="Folder to save TF-IDF outputs.")
    parser.add_argument("--query", default="", help="Optional query to rank chunks against.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top chunks to return.")
    parser.add_argument("--max_features", type=int, default=10000, help="Maximum TF-IDF vocabulary size.")
    parser.add_argument("--ngram_min", type=int, default=1, help="Minimum n-gram size.")
    parser.add_argument("--ngram_max", type=int, default=2, help="Maximum n-gram size.")
    parser.add_argument("--eval_file", default="", help="Optional JSON file with benchmark queries for evaluation.")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_summary = summarize_corpus(corpus, vectorizer, matrix)
    save_json(corpus_summary, output_dir / "tfidf_summary.json")
    save_json(corpus, output_dir / "chunk_corpus.json")

    print("TF-IDF index built successfully.")
    print(json.dumps(corpus_summary, indent=2))

    if args.query.strip():
        results = rank_chunks(
            args.query,
            vectorizer=vectorizer,
            matrix=matrix,
            corpus=corpus,
            top_k=args.top_k,
            ranking_method=args.ranking_method,
            word2vec_model=word2vec_model,
            word2vec_matrix=word2vec_matrix,
            tfidf_weight=args.tfidf_weight,
            word2vec_weight=args.word2vec_weight,
        )

        query_output = {
            "query": args.query,
            "top_k": args.top_k,
            "ranking_method": args.ranking_method,
            "results": results
        }
        save_json(query_output, output_dir / "query_results.json")

        print("\nTop results:")
        for item in results:
            print(f"[{item['rank']}] score={item['score']:.4f} | {item['title']} | chunk={item['chunk_id']} | heading={item['heading']}")
        print(f"\nSaved query results to: {output_dir / 'query_results.json'}")

    if args.eval_file.strip():
        eval_path = Path(args.eval_file)
        with eval_path.open("r", encoding="utf-8") as f:
            benchmark_queries = json.load(f)
        evaluation = evaluate_queries(
            benchmark_queries,
            vectorizer=vectorizer,
            matrix=matrix,
            corpus=corpus,
            top_k=args.top_k,
            ranking_method=args.ranking_method,
            word2vec_model=word2vec_model,
            word2vec_matrix=word2vec_matrix,
            tfidf_weight=args.tfidf_weight,
            word2vec_weight=args.word2vec_weight,
        )
        save_json(evaluation, output_dir / "evaluation_results.json")
        print("\nEvaluation summary:")
        print(json.dumps(evaluation["aggregate"], indent=2))
        print(f"Saved evaluation results to: {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
