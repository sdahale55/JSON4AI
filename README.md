# JSON4AI

JSON4AI is a small information retrieval pipeline that converts raw HTML pages into structured JSON, chunks the cleaned content, and runs TF-IDF search over those chunks.

The project is organized around two main stages:

1. `preprocessing.py`
   - Removes obvious HTML noise such as scripts, navigation, tables, and boilerplate.
   - Extracts sectioned content into a structured JSON format.
   - Chunks sections into overlapping passage-sized units for retrieval.

2. `tfidf.py`
   - Builds a TF-IDF index over the processed chunks.
   - Uses title and heading context to improve ranking.
   - Supports `tfidf`, `word2vec`, and `hybrid` ranking modes.
   - Supports direct querying and benchmark evaluation.

3. `json4ai.py`
   - Takes the top ranked evidence chunks for a query.
   - Produces query-focused structured JSON.
   - Performs a lightweight schema validation step before saving output.

4. `api.py`
   - Exposes retrieval and structured JSON generation through FastAPI.
   - Provides a simple service layer for agent or app integration.

5. `gold_evaluation.py`
   - Compares generated structured JSON against hand-written or LLM-reviewed gold JSON.
   - Reports schema validity, token-level precision/recall/F1, and overlap metrics.

## Repository Layout

- `html_data/`: raw HTML source documents
- `processed_data/`: cleaned JSON outputs from preprocessing
- `tfidf_output/`: saved corpus, retrieval, and evaluation outputs
- `evaluation_queries.json`: small benchmark set for repeatable evaluation
- `preprocessing.py`: HTML to JSON pipeline
- `tfidf.py`: indexing, ranking, and evaluation pipeline
- `json4ai.py`: end-to-end structured JSON generation from ranked evidence
- `api.py`: FastAPI wrapper for search and structured output
- `gold_evaluation.py`: gold JSON comparison workflow
- `sample_gold_queries.json`: sample format for gold structured outputs

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Pipeline

Generate structured JSON from the HTML dataset:

```bash
python preprocessing.py
```

Build the TF-IDF index and run a query:

```bash
python tfidf.py --query "economy of Illinois" --top_k 5
```

Run retrieval with Word2Vec only:

```bash
python tfidf.py --query "economy of Illinois" --ranking_method word2vec --top_k 5
```

Run hybrid TF-IDF + Word2Vec retrieval:

```bash
python tfidf.py --query "economy of Illinois" --ranking_method hybrid --tfidf_weight 0.7 --word2vec_weight 0.3 --top_k 5
```

Run benchmark evaluation:

```bash
python tfidf.py --eval_file evaluation_queries.json --top_k 5
```

Run both a query and evaluation in the same pass:

```bash
python tfidf.py --query "Great Lakes shipping" --eval_file evaluation_queries.json --top_k 5
```

Generate the final structured JSON response for a query:

```bash
python json4ai.py --query "economy of Illinois" --top_k 5
```

Generate structured JSON with hybrid ranking:

```bash
python json4ai.py --query "economy of Illinois" --ranking_method hybrid --tfidf_weight 0.7 --word2vec_weight 0.3 --top_k 5
```

Compare predictions against gold JSON:

```bash
python gold_evaluation.py --gold_file sample_gold_queries.json --ranking_method hybrid --top_k 5
```

Run the API locally:

```bash
uvicorn api:app --reload
```

## Outputs

`processed_data/*.json` contains:

- source file metadata
- cleaned full text
- extracted sections
- overlapping chunks for retrieval

`tfidf_output/` contains:

- `chunk_corpus.json`: flattened chunk corpus used for indexing
- `tfidf_summary.json`: corpus-level summary statistics
- `query_results.json`: top results for the latest query
- `evaluation_results.json`: benchmark metrics such as Precision@K, Recall@K, MRR, and Hit Rate
- `structured_output.json`: schema-checked query-focused JSON response built from top evidence
- `gold_evaluation_results.json`: comparison metrics against gold structured JSON

## API Endpoints

- `GET /health`: basic liveness check
- `POST /search`: returns ranked retrieval chunks
- `POST /structured`: returns the final query-focused structured JSON response

## Gold JSON Format

Each gold example in `sample_gold_queries.json` follows this shape:

```json
{
  "query": "economy of Illinois",
  "gold_output": {
    "query": "...",
    "answer_summary": "...",
    "source_titles": ["..."],
    "key_terms": ["..."],
    "metadata": {
      "top_k": 5,
      "num_evidence_chunks": 5,
      "generator": "gold-reference"
    },
    "evidence": [
      {
        "snippet": "..."
      }
    ]
  }
}
```

The evaluator is forgiving about exact formatting and focuses on overlap-based scoring, which makes it usable for hand-written gold data and LLM-reviewed reference JSON.

## Current Retrieval Improvements

- Citation markers such as `[12]` are stripped during preprocessing.
- Reference-heavy sections like `References` and `External links` are skipped.
- Duplicate blocks are filtered out to reduce repeated text.
- Ranking uses document title and section heading context in addition to chunk text.
- The final stage now converts top evidence into a structured JSON object instead of returning only ranked raw text.
- Retrieval experiments can now compare plain TF-IDF, pure Word2Vec, and a hybrid weighted score.
- Final output evaluation can now be run against gold structured JSON rather than retrieval-only labels.

## Notes

- The current dataset contains 15 HTML pages.
- The benchmark file is intentionally small and lightweight, but it gives the project a concrete evaluation story for a final submission.
- Once you add the 50-100 page dataset, the main next step is to create a matching gold JSON file for a subset of queries and run `gold_evaluation.py` across ranking variants.
