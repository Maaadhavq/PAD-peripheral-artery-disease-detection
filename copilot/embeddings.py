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


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding backend cannot be reached or lacks the model.

    Retrieval runs against the same Ollama server as generation, so it fails the
    same way. Callers that already degrade gracefully on OllamaUnavailable need
    an equivalent here, rather than a raw requests exception escaping.
    """


class OllamaEmbedder:
    """Embeddings from a local Ollama server."""

    name = "ollama"

    def __init__(self, model=EMBED_MODEL, host=OLLAMA_HOST):
        self.model = model
        self.host = host.rstrip("/")

    def _unreachable(self, cause=None):
        return EmbeddingUnavailable(
            f"Could not reach Ollama at {self.host} to embed the query. Is it running?\n"
            "  Install:  https://ollama.com/download\n"
            f"  Pull:     ollama pull {self.model}"
        )

    def is_available(self):
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def embed(self, texts):
        vectors = []
        for text in texts:
            try:
                response = requests.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=TIMEOUT,
                )
            except requests.RequestException as error:
                raise self._unreachable() from error

            if response.status_code == 404:
                raise EmbeddingUnavailable(
                    f"Ollama has no embedding model named {self.model!r}. Pull it with:\n"
                    f"    ollama pull {self.model}"
                )
            try:
                response.raise_for_status()
                vectors.append(response.json()["embedding"])
            except (requests.RequestException, KeyError, ValueError) as error:
                raise EmbeddingUnavailable(
                    f"Ollama returned an unusable embedding response: {error}"
                ) from error

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
