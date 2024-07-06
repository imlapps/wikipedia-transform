from dagster import ConfigurableResource

from etl.models.types import EnrichmentType, RecordType
from etl.resources.open_ai_settings import OpenAiSettings


class OpenAiPipelineConfig(ConfigurableResource):  # type: ignore[misc]
    """A ConfigurableResource that holds the shared parameters of OpenAI Pipelines."""

    openai_settings: OpenAiSettings
    record_type: RecordType
    enrichment_type: EnrichmentType
