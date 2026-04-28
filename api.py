from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from json4ai import build_structured_output, validate_structured_output
from tfidf import (
    build_chunk_corpus,
    build_tfidf_index,
    build_word2vec_index,
    load_processed_documents,
    rank_chunks,
)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    ranking_method: str = "tfidf"
    tfidf_weight: float = 0.7
    word2vec_weight: float = 0.3


app = FastAPI(title="JSON4AI API", version="1.0.0")


@lru_cache(maxsize=1)
def get_index():
    docs = load_processed_documents("processed_data")
    corpus = build_chunk_corpus(docs)
    vectorizer, matrix = build_tfidf_index(corpus)
    try:
        word2vec_model, word2vec_matrix = build_word2vec_index(corpus)
    except Exception:
        word2vec_model, word2vec_matrix = None, None
    return corpus, vectorizer, matrix, word2vec_model, word2vec_matrix


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(request: QueryRequest):
    corpus, vectorizer, matrix, word2vec_model, word2vec_matrix = get_index()
    return rank_chunks(
        request.query,
        vectorizer=vectorizer,
        matrix=matrix,
        corpus=corpus,
        top_k=request.top_k,
        ranking_method=request.ranking_method,
        word2vec_model=word2vec_model,
        word2vec_matrix=word2vec_matrix,
        tfidf_weight=request.tfidf_weight,
        word2vec_weight=request.word2vec_weight,
    )


@app.post("/structured")
def structured(request: QueryRequest):
    corpus, vectorizer, matrix, word2vec_model, word2vec_matrix = get_index()
    results = rank_chunks(
        request.query,
        vectorizer=vectorizer,
        matrix=matrix,
        corpus=corpus,
        top_k=request.top_k,
        ranking_method=request.ranking_method,
        word2vec_model=word2vec_model,
        word2vec_matrix=word2vec_matrix,
        tfidf_weight=request.tfidf_weight,
        word2vec_weight=request.word2vec_weight,
    )
    output = build_structured_output(request.query, results, request.top_k)
    validate_structured_output(output)
    return output
