from dagster import Config

from etl.models.open_ai_settings import OpenAiSettings
from etl.models.types import EnrichmentType, RecordType


class OpenAiResourceParams(Config):  # type: ignore
    openai_settings: OpenAiSettings
    record_type: RecordType
    enrichment_type: EnrichmentType
