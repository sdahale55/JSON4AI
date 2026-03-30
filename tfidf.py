from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
                    "word_count": len(text.split())
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
                "word_count": len(text.split())
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
    texts = [item["text"] for item in corpus]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        lowercase=True
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def rank_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix,
    corpus: list[dict[str, Any]],
    top_k: int = 5
) -> list[dict[str, Any]]:
    query = normalize_space(query)
    if not query:
        raise ValueError("Query cannot be empty.")

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]

    ranked_indices = scores.argsort()[::-1][:top_k]
    results: list[dict[str, Any]] = []

    for rank, idx in enumerate(ranked_indices, start=1):
        item = corpus[int(idx)]
        results.append({
            "rank": rank,
            "score": float(scores[idx]),
            "source_file": item["source_file"],
            "title": item["title"],
            "chunk_id": item["chunk_id"],
            "heading": item["heading"],
            "word_count": item["word_count"],
            "text": item["text"]
        })

    return results


def summarize_corpus(corpus: list[dict[str, Any]], vectorizer: TfidfVectorizer, matrix) -> dict[str, Any]:
    word_counts = [item["word_count"] for item in corpus]
    unique_docs = len({item["source_file"] for item in corpus})

    return {
        "num_documents": unique_docs,
        "num_chunks": len(corpus),
        "vocab_size": len(vectorizer.get_feature_names_out()),
        "tfidf_matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "avg_chunk_words": round(sum(word_counts) / len(word_counts), 2) if word_counts else 0
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
    args = parser.parse_args()

    docs = load_processed_documents(args.processed_dir)
    corpus = build_chunk_corpus(docs)
    vectorizer, matrix = build_tfidf_index(
        corpus,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max
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
            top_k=args.top_k
        )

        query_output = {
            "query": args.query,
            "top_k": args.top_k,
            "results": results
        }
        save_json(query_output, output_dir / "query_results.json")

        print("\nTop results:")
        for item in results:
            print(f"[{item['rank']}] score={item['score']:.4f} | {item['title']} | chunk={item['chunk_id']} | heading={item['heading']}")
        print(f"\nSaved query results to: {output_dir / 'query_results.json'}")


if __name__ == "__main__":
    main()
