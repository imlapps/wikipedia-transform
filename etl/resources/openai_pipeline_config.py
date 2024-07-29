from dagster import ConfigurableResource

from etl.models.types import EnrichmentType
from etl.resources import OpenaiSettings


class OpenaiPipelineConfig(ConfigurableResource):  # type: ignore[misc]
    """A ConfigurableResource that holds the shared parameters of OpenAI Pipelines."""

    openai_settings: OpenaiSettings
    enrichment_type: EnrichmentType
