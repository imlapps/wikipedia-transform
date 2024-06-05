from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from wikipedia_transform.models.types import (
    ApiKey,
    EmbeddingModelType,
    EnhancementType,
    GenerativeModelType,
    RecordType,
)

CONFIG_FILE_PATH = Path(__file__).parent.parent.parent.absolute()


class Settings(BaseSettings):
    """A Pydantic BaseSetting to hold environment variables."""

    record_type: RecordType = RecordType.WIKIPEDIA
    enhancements: frozenset[EnhancementType] = frozenset()

    embedding_model_type: EmbeddingModelType = EmbeddingModelType.OPEN_AI
    generative_model_type: GenerativeModelType = GenerativeModelType.OPEN_AI

    data_file_paths: frozenset[Path] = frozenset()
    output_path: Path = Path(__file__).parent.parent.absolute() / "data" / "output"

    openai_api_key: ApiKey | None = None

    model_config = SettingsConfigDict(
        env_file=(CONFIG_FILE_PATH / ".env.local", CONFIG_FILE_PATH / ".env.secret"),
        extra="ignore",
        env_file_encoding="utf-8",
        validate_default=False,
    )

    @field_validator("data_file_paths", mode="before")
    @classmethod
    def convert_to_list_of_file_paths(
        cls, file_names: frozenset[str]
    ) -> frozenset[Path]:
        """Convert the list of file names in environment variables into a list of Path objects."""
        return frozenset(
            [
                Path(__file__).parent.parent.absolute() / "data" / file_name
                for file_name in file_names
            ]
        )


settings = Settings()
