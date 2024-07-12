from dagster import ConfigurableResource

from etl.models.types import EnrichmentType
from etl.resources import OpenAiSettings


class OpenAiPipelineConfig(ConfigurableResource):  # type: ignore[misc]
    """A ConfigurableResource that holds the shared parameters of OpenAI Pipelines."""

    openai_settings: OpenAiSettings
    enrichment_type: EnrichmentType
