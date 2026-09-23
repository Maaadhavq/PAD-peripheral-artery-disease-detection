"""A small on-disk vector store.

The knowledge base is a few hundred chunks, so an in-memory matrix of unit
vectors and a dot product beat a database on both speed and dependencies. The
index is a .npz of vectors plus a .json of chunk metadata.
"""

import json
from pathlib import Path

import numpy as np

from pad.paths import KNOWLEDGE_INDEX_DIR as DEFAULT_INDEX_DIR

VECTOR_FILE = "vectors.npz"
CHUNK_FILE = "chunks.json"


class VectorStore:
    """Cosine-similarity search over embedded chunks."""

    def __init__(self, vectors, chunks, embedder_name=""):
        if len(vectors) != len(chunks):
            raise ValueError(
                f"{len(vectors)} vectors but {len(chunks)} chunks - index is corrupt."
            )
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.chunks = chunks
        self.embedder_name = embedder_name

    def __len__(self):
        return len(self.chunks)

    def search(self, query_vector, k=6, min_score=0.0):
        """The k best-matching chunks, each with its similarity score."""
        if len(self) == 0:
            return []
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        scores = self.vectors @ query_vector

        k = min(k, len(self))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        results = []
        for index in top:
            score = float(scores[index])
            if score < min_score:
                continue
            results.append({**self.chunks[index], "score": score})
        return results

    def save(self, index_dir=DEFAULT_INDEX_DIR):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(index_dir / VECTOR_FILE, vectors=self.vectors)
        (index_dir / CHUNK_FILE).write_text(
            json.dumps({"embedder": self.embedder_name, "chunks": self.chunks}, indent=1),
            encoding="utf-8",
        )
        return index_dir

    @classmethod
    def load(cls, index_dir=DEFAULT_INDEX_DIR):
        index_dir = Path(index_dir)
        vector_path = index_dir / VECTOR_FILE
        chunk_path = index_dir / CHUNK_FILE
        if not vector_path.exists() or not chunk_path.exists():
            raise FileNotFoundError(
                f"No knowledge index at {index_dir}. Run: python -m copilot.ingest"
            )
        vectors = np.load(vector_path)["vectors"]
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
        return cls(vectors, payload["chunks"], payload.get("embedder", ""))
