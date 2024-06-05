from pydantic import BaseModel

from wikipedia_transform.models.types import OpenAiEmbeddingModelName


class OpenAiEmbeddingModel(BaseModel):
    """Pydantic Model to hold the parameters of an OpenAI embedding model"""

    name: OpenAiEmbeddingModelName = OpenAiEmbeddingModelName.ADA_V2
