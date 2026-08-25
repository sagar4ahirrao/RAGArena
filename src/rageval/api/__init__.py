"""FastAPI serving layer for RAGEval — dashboard + REST API."""
from .server import app, start_server

__all__ = ["app", "start_server"]
