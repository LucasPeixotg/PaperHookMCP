# api/__init__.py
from .open_alex_client import OpenAlexClient
from .semantic_scholar_client import SemanticScholarClient

__all__ = ["OpenAlexClient", "SemanticScholarClient"]