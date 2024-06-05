from enum import Enum


class OpenAiEmbeddingModelName(str, Enum):
    """An enum of OpenAI embedding model names."""

    ADA_V2: str = "text-embedding-ada-002"
    TEXT_EMBEDDING_MODEL_SMALL: str = "text-embedding-3-small"
    TEXT_EMBEDDING_MODEL_LARGE: str = "text-embedding-3-large"
