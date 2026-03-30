from __future__ import annotations

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup, Comment

# -------- helpers --------

def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def is_meaningful(text: str) -> bool:
    text = normalize_text(text)
    if len(text) < 20:
        return False

    bad_starts = [
        "Main menu", "Contents", "Navigation", "Contact us",
        "Privacy policy", "About Wikipedia", "Jump to content"
    ]
    if any(text.startswith(x) for x in bad_starts):
        return False

    # skip blocks that are mostly symbols / code junk
    letters = sum(ch.isalnum() for ch in text)
    return letters / max(len(text), 1) > 0.5


# -------- core preprocessing --------

def remove_noise(soup: BeautifulSoup) -> None:
    # remove comments
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()

    # remove obvious junk tags
    for tag in soup(["script", "style", "noscript", "svg", "path", "iframe", "canvas"]):
        tag.decompose()

    # remove common layout / boilerplate tags
    for tag in soup.find_all(["nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    # remove common boilerplate selectors
    noisy_selectors = [
        "[role='navigation']",
        "[aria-hidden='true']",
        ".nav", ".navbar", ".menu", ".sidebar", ".footer", ".header",
        ".advertisement", ".ads", ".promo", ".cookie", ".banner",
        ".breadcrumbs", ".social", ".share"
    ]
    for selector in noisy_selectors:
        for tag in soup.select(selector):
            tag.decompose()


def find_main_content(soup: BeautifulSoup):
    # try useful article roots first
    candidates = [
        soup.select_one("#mw-content-text"),   # Wikipedia
        soup.find("main"),
        soup.find("article"),
        soup.find(attrs={"role": "main"}),
        soup.select_one("#content"),
        soup.body
    ]
    for c in candidates:
        if c is not None:
            return c
    return soup


def extract_sections(root) -> list[dict]:
    sections = []
    current_heading = "Introduction"
    current_parts = []

    for tag in root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = normalize_text(tag.get_text(" ", strip=True))
        if not is_meaningful(text):
            continue

        if tag.name in ["h1", "h2", "h3"]:
            if current_parts:
                sections.append({
                    "heading": current_heading,
                    "text": normalize_text(" ".join(current_parts))
                })
            current_heading = text
            current_parts = []
        else:
            current_parts.append(text)

    if current_parts:
        sections.append({
            "heading": current_heading,
            "text": normalize_text(" ".join(current_parts))
        })

    return sections


def chunk_sections(sections: list[dict], max_words: int = 180, overlap: int = 30) -> list[dict]:
    chunks = []
    chunk_id = 0

    for sec in sections:
        words = sec["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "heading": sec["heading"],
                    "text": chunk_text
                })
                chunk_id += 1

            if end == len(words):
                break
            start = max(0, end - overlap)

    return chunks


def preprocess_html(html: str, source_file: str = "") -> dict:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)

    remove_noise(soup)
    root = find_main_content(soup)
    sections = extract_sections(root)
    cleaned_text = "\n\n".join(
        f"{sec['heading']}\n{sec['text']}" for sec in sections
    ).strip()
    chunks = chunk_sections(sections)

    return {
        "source_file": source_file,
        "title": title,
        "num_sections": len(sections),
        "num_chunks": len(chunks),
        "cleaned_text": cleaned_text,
        "sections": sections,
        "chunks": chunks
    }


# -------- batch processing for folder --------

def preprocess_file(file_path: Path, output_dir: Path) -> dict:
    html = file_path.read_text(encoding="utf-8", errors="ignore")
    result = preprocess_html(html, source_file=str(file_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{file_path.stem}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def preprocess_dataset(input_dir: str = "html_data", output_dir: str = "processed_data") -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    files = sorted(list(input_path.rglob("*.html")) + list(input_path.rglob("*.htm")))
    if not files:
        print("No HTML files found in", input_dir)
        return

    summary = []
    for fp in files:
        result = preprocess_file(fp, output_path)
        summary.append({
            "file": result["source_file"],
            "title": result["title"],
            "num_sections": result["num_sections"],
            "num_chunks": result["num_chunks"]
        })
        print(f"Processed {fp.name}: {result['num_sections']} sections, {result['num_chunks']} chunks")

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    preprocess_dataset("html_data", "processed_data")
