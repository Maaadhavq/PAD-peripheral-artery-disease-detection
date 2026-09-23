"""Project paths.

Resolved relative to the repository, not the working directory, so
`streamlit run app.py` and `python -m copilot.ingest` find the same artifacts
regardless of where they are launched from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
KNOWLEDGE_INDEX_DIR = ARTIFACT_DIR / "knowledge_index"
