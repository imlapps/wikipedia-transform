from dagster import Config

from etl.models.open_ai_settings import OpenAiSettings
from etl.models.types import EnrichmentType, RecordType


class OpenAiResourceParams(Config):
    openai_settings: OpenAiSettings
    record_type: RecordType | None = RecordType.WIKIPEDIA
    enrichment_type: EnrichmentType | None= EnrichmentType.SUMMARY
