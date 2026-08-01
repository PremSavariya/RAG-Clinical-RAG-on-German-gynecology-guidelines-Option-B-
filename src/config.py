from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def load_yaml(path: Path | None = None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml()


def env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Missing env var: {key}")
    return val


OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = env("OLLAMA_API_KEY", "ollama")
LLM_MODEL_NAME = env("LLM_MODEL_NAME", "qwen2.5:3b")
EMBEDDINGS_MODEL_NAME = env("EMBEDDINGS_MODEL_NAME", "bge-m3:latest")

PDF_DIR = ROOT / env("PDF_DIR", CFG["paths"]["pdf_dir"])
ARTIFACTS_DIR = ROOT / env("ARTIFACTS_DIR", CFG["paths"]["artifacts_dir"])
CHROMA_DIR = ROOT / env("CHROMA_DIR", CFG["paths"]["chroma_dir"])


def is_e5_model(name: str | None = None) -> bool:
    n = (name or EMBEDDINGS_MODEL_NAME).lower()
    return "e5" in n
