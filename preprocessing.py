from __future__ import annotations

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup, Comment

# Tags that are usually boilerplate / noise for retrieval
REMOVE_TAGS = {
    "script", "style", "noscript", "svg", "path", "iframe", "canvas",
    "footer", "nav", "form", "aside", "button", "input"
}

# Optional selectors for common boilerplate blocks
REMOVE_SELECTORS = [
    "[role='navigation']",
    "[aria-hidden='true']",
    ".nav", ".navbar", ".menu", ".footer", ".sidebar",
    ".advertisement", ".ads", ".promo", ".cookie", ".banner",
    ".breadcrumbs", ".social", ".share"
]


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_html(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # Remove comments
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()

    # Remove noisy tags
    for tag in soup.find_all(REMOVE_TAGS):
        tag.decompose()

    # Remove obvious boilerplate selectors
    for selector in REMOVE_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # Remove elements hidden via inline style
    for tag in soup.find_all(style=True):
        style = tag.get("style", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            tag.decompose()

    # Try to focus on main content first
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )

    title = ""
    if soup.title and soup.title.string:
        title = normalize_whitespace(soup.title.get_text(" ", strip=True))

    # Extract block-level text to preserve some structure
    blocks = []
    for elem in main.find_all(["h1", "h2", "h3", "p", "li", "pre", "code", "blockquote"]):
        text = elem.get_text(" ", strip=True)
        text = normalize_whitespace(text)
        if len(text) >= 20:  # ignore tiny fragments
            blocks.append(text)

    # Fallback if block extraction is too sparse
    if not blocks:
        raw_text = main.get_text("\n", strip=True)
        raw_text = normalize_whitespace(raw_text)
        blocks = [b.strip() for b in raw_text.split("\n\n") if len(b.strip()) >= 20]

    cleaned_text = "\n\n".join(blocks)
    cleaned_text = normalize_whitespace(cleaned_text)

    return {
        "title": title,
        "cleaned_text": cleaned_text,
        "blocks": blocks,
    }


def chunk_text(text: str, max_words: int = 180, overlap_words: int = 40) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_words = []

    for para in paragraphs:
        para_words = para.split()

        # If one paragraph is huge, split it directly
        if len(para_words) > max_words:
            if current_words:
                chunks.append(" ".join(current_words).strip())
                current_words = []

            start = 0
            while start < len(para_words):
                end = min(start + max_words, len(para_words))
                chunk = " ".join(para_words[start:end]).strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(para_words):
                    break
                start = max(0, end - overlap_words)
            continue

        if len(current_words) + len(para_words) <= max_words:
            current_words.extend(para_words)
        else:
            if current_words:
                chunks.append(" ".join(current_words).strip())

            overlap = current_words[-overlap_words:] if overlap_words > 0 else []
            current_words = overlap + para_words

    if current_words:
        chunks.append(" ".join(current_words).strip())

    return chunks


def preprocess_file(file_path: Path, output_dir: Path) -> dict:
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_html(html)
    chunks = chunk_text(cleaned["cleaned_text"])

    result = {
        "source_file": str(file_path),
        "title": cleaned["title"],
        "num_blocks": len(cleaned["blocks"]),
        "num_chunks": len(chunks),
        "cleaned_text": cleaned["cleaned_text"],
        "chunks": [{"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{file_path.stem}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def preprocess_dataset(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    html_files = sorted(
        list(input_path.rglob("*.html")) + list(input_path.rglob("*.htm"))
    )

    if not html_files:
        print("No HTML files found.")
        return

    summary = []
    for file_path in html_files:
        result = preprocess_file(file_path, output_path)
        summary.append({
            "file": result["source_file"],
            "title": result["title"],
            "num_blocks": result["num_blocks"],
            "num_chunks": result["num_chunks"],
        })
        print(f"Processed: {file_path.name} -> {result['num_chunks']} chunks")

    summary_file = output_path / "summary.json"
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Outputs saved to: {output_path}")


if __name__ == "__main__":
    # Example:
    # Put your HTML files in ./html_data
    # Processed JSON files will be saved in ./processed_data
    preprocess_dataset("html_data", "processed_data")
