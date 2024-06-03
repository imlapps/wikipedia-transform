from enum import Enum


class EmbeddingModelType(str, Enum):
    """An enum of embedding model types."""

    OPEN_AI = "OpenAI"
