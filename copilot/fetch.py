"""Fetching the public-domain reference pages listed in sources.yaml.

Pages are cached under knowledge/_cache so repeated ingests do not hammer the
source sites, and so an ingest can run offline once the cache is warm.
"""

from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
CACHE_DIR = KNOWLEDGE_DIR / "_cache"
SOURCES_FILE = Path(__file__).parent / "sources.yaml"

USER_AGENT = "PAD-copilot/1.0 (research project; +https://github.com/Maaadhavq)"
TIMEOUT = 30

# Page furniture that carries no clinical content.
BOILERPLATE = {
    "MENU", "Email", "Print", "Language switcher", "Español",
    "< Back To Health Topics", "< Back To Peripheral Artery Disease",
    "PERIPHERAL ARTERY DISEASE", "KEY POINTS",
}


def load_sources(sources_file=SOURCES_FILE):
    """The source list from sources.yaml."""
    data = yaml.safe_load(Path(sources_file).read_text(encoding="utf-8"))
    return data["sources"]


def html_to_text(html):
    """Strip a page down to readable text, keeping headings as section markers.

    Headings are emitted as markdown so the chunker can attach a section name
    to every chunk, which is what makes a citation specific enough to check.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "form", "noscript"]):
        tag.decompose()

    root = soup.find("article") or soup.find("main") or soup.body or soup

    lines = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text or text in BOILERPLATE or len(text) < 3:
            continue
        if element.name in {"h1", "h2", "h3", "h4"}:
            level = int(element.name[1])
            lines.append(f"\n{'#' * level} {text}\n")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    # Collapse runs of blank lines left behind by removed furniture.
    out, blank = [], False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and blank:
            continue
        out.append(line)
        blank = is_blank
    return "\n".join(out).strip()


def fetch_source(source, cache_dir=CACHE_DIR, refresh=False):
    """Fetch one source to text, using the cache unless refresh is asked for."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{source['id']}.md"

    if cache_file.exists() and not refresh:
        return cache_file.read_text(encoding="utf-8")

    response = requests.get(
        source["url"], headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
    )
    response.raise_for_status()
    text = html_to_text(response.text)
    if not text:
        raise ValueError(f"No readable text extracted from {source['url']}")

    cache_file.write_text(text, encoding="utf-8")
    return text


def local_documents(knowledge_dir=KNOWLEDGE_DIR):
    """The markdown files committed in knowledge/ (model card, data dictionary).

    These are written by this project, not fetched, so they carry no licence
    questions and are always available offline.
    """
    knowledge_dir = Path(knowledge_dir)
    documents = []
    for path in sorted(knowledge_dir.glob("*.md")):
        documents.append({
            "id": path.stem,
            "title": path.stem.replace("_", " ").title(),
            "url": f"knowledge/{path.name}",
            "publisher": "This project",
            "text": path.read_text(encoding="utf-8"),
        })
    return documents
