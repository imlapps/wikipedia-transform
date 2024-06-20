from dagster import Config
from pydantic import Field

from etl.models.types import ApiKey, OpenAiEmbeddingModelName, OpenAiGenerativeModelName


class OpenAiSettings(Config):  # type: ignore
    openai_api_key: ApiKey = Field(default=..., description="OpenAI API key")
    embedding_model_name: OpenAiEmbeddingModelName = Field(
        default=OpenAiEmbeddingModelName.TEXT_EMBEDDING_3_LARGE
    )
    generative_model_name: OpenAiGenerativeModelName = Field(
        default=OpenAiGenerativeModelName.GPT_4O
    )
    temperature: int = Field(default=0)
