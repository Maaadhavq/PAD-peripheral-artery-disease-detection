"""Splitting documents into retrievable chunks that remember where they came from."""

import re

CHUNK_WORDS = 220
OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 25

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def split_sections(text):
    """Split markdown-ish text into (section_title, body) pairs."""
    sections = []
    current_title = "Overview"
    current_body = []

    for line in text.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
                current_body = []
            current_title = match.group(2).strip()
        else:
            current_body.append(line)

    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))

    return [(title, body) for title, body in sections if body]


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Sliding word window over a single section."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text] if words else []

    step = max(chunk_words - overlap_words, 1)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_words]
        if len(window) < MIN_CHUNK_WORDS and chunks:
            break
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_document(document, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS):
    """Chunk a document, tagging each chunk with its source and section.

    The section title travels with the chunk so an answer can cite
    "NHLBI — Diagnosis", not just "NHLBI".
    """
    chunks = []
    for section_title, body in split_sections(document["text"]):
        for index, piece in enumerate(chunk_text(body, chunk_words, overlap_words)):
            chunks.append({
                "id": f"{document['id']}::{section_title}::{index}",
                "text": piece,
                "source_id": document["id"],
                "source_title": document["title"],
                "section": section_title,
                "url": document.get("url", ""),
                "publisher": document.get("publisher", ""),
            })
    return chunks
