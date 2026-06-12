"""Ollama LLM configuration for the AI Goal Navigator."""

import os
from typing import List

import requests
from langchain_community.chat_models import ChatOllama

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _available_models(base_url: str) -> List[str]:
    try:
        # Reduce timeout to 1 second since localhost queries should be instant.
        # This avoids long startup delays if Ollama is starting, offline, or slow.
        response = requests.get(f"{base_url}/api/tags", timeout=1)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model.get("name", "") for model in models if model.get("name")]
    except Exception:
        return []


def _select_model() -> str:
    # If the user has explicitly set OLLAMA_MODEL, use it directly to bypass network checks.
    preferred = os.getenv("OLLAMA_MODEL")
    if preferred:
        return preferred

    default_model = "qwen3:8b"
    installed = set(_available_models(OLLAMA_BASE_URL))
    if not installed:
        return default_model

    candidates = [
        default_model,
        "llama3:latest",
        "phi3:latest",
        "phi3:mini",
    ]

    for candidate in candidates:
        if candidate in installed:
            return candidate

    return default_model


llm = ChatOllama(
    model=_select_model(),
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
)

