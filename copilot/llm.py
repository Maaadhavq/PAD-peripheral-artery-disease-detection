"""Local LLM client.

Generation runs on a local Ollama server. That is a requirement rather than a
preference: anything derived from MIMIC-IV must not be sent to a third-party
API under the PhysioNet data use agreement.

Talks to the Ollama REST API directly so the project does not depend on the
Ollama Python SDK.
"""

import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_MODEL = os.environ.get("PAD_LLM_MODEL", "llama3.1:8b")
TIMEOUT = 300


class OllamaUnavailable(RuntimeError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaClient:
    """Minimal chat client for a local Ollama server."""

    def __init__(self, model=CHAT_MODEL, host=OLLAMA_HOST, temperature=0.2):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def is_available(self):
        try:
            return requests.get(f"{self.host}/api/tags", timeout=5).ok
        except requests.RequestException:
            return False

    def installed_models(self):
        """Names of the models this server has pulled."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            return [model["name"] for model in response.json().get("models", [])]
        except requests.RequestException:
            return []

    def chat(self, system, user):
        """One completion. Raises OllamaUnavailable if the server is not up."""
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
                timeout=TIMEOUT,
            )
        except requests.RequestException as error:
            raise OllamaUnavailable(
                f"Could not reach Ollama at {self.host}. Is it running?\n"
                "  Install:  https://ollama.com/download\n"
                f"  Pull:     ollama pull {self.model}"
            ) from error

        if response.status_code == 404:
            raise OllamaUnavailable(
                f"Ollama has no model named {self.model!r}. Pull it with:\n"
                f"    ollama pull {self.model}"
            )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


class EchoClient:
    """Test double. Returns a fixed, citation-bearing answer without a server."""

    def __init__(self, reply=None):
        self.reply = reply or "Stub answer grounded in the references [1]."
        self.model = "echo"
        self.calls = []

    def is_available(self):
        return True

    def installed_models(self):
        return ["echo"]

    def chat(self, system, user):
        self.calls.append({"system": system, "user": user})
        return self.reply
