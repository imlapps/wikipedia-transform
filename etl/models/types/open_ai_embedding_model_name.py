from enum import Enum


class OpenAiEmbeddingModelName(str, Enum):
    """An enum of OpenAI embedding model names."""

    TEXT_EMBEDDING_ADA_002: str = "text-embedding-ada-002"
    TEXT_EMBEDDING_3_SMALL: str = "text-embedding-3-small"
    TEXT_EMBEDDING_3_LARGE: str = "text-embedding-3-large"
