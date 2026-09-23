"""Embedding providers.

Ollama is the real one — it runs locally, so nothing derived from MIMIC-IV
leaves the machine, which is what the data use agreement requires.

The hashing provider is a deterministic stand-in used only by the tests. It is
not a silent fallback: callers ask for it by name, because a hashed bag of
words retrieves far worse than a real embedding model and quietly substituting
it would make the copilot look like it works when it does not.
"""

import hashlib
import os
import re

import numpy as np
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("PAD_EMBED_MODEL", "nomic-embed-text")
TIMEOUT = 120

HASH_DIMENSIONS = 512
TOKEN_RE = re.compile(r"[a-z0-9]+")


class OllamaEmbedder:
    """Embeddings from a local Ollama server."""

    name = "ollama"

    def __init__(self, model=EMBED_MODEL, host=OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def is_available(self):
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def embed(self, texts):
        vectors = []
        for text in texts:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return _normalize(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """Deterministic hashed bag-of-words. Test fixture only — see module docstring."""

    name = "hashing"

    def __init__(self, dimensions=HASH_DIMENSIONS):
        self.dimensions = dimensions

    def is_available(self):
        return True

    def embed(self, texts):
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in TOKEN_RE.findall(text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self.dimensions
                vectors[row, index] += 1.0
        return _normalize(vectors)


def _normalize(vectors):
    """Unit-length rows, so a dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def get_embedder(provider="ollama", **kwargs):
    """Build an embedder by name. Unknown names fail loudly rather than degrade."""
    providers = {"ollama": OllamaEmbedder, "hashing": HashingEmbedder}
    if provider not in providers:
        raise ValueError(
            f"Unknown embedding provider {provider!r}. Choose one of {sorted(providers)}."
        )
    return providers[provider](**kwargs)
