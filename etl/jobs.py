from dagster import EnvVar, RunConfig, define_asset_job

from .assets import wikipedia_articles_embeddings
from .models import OpenAiPipelineConfig, OpenAiSettings
from .models.types import EnrichmentType, RecordType

openai_pipeline_config = OpenAiPipelineConfig(
    openai_settings=OpenAiSettings(
        openai_api_key=EnvVar("OPENAI_API_KEY").get_value("")
    ),
    record_type=EnvVar("RECORD_TYPE").get_value(default=RecordType.WIKIPEDIA),
    enrichment_type=EnvVar("ENRICHMENT_TYPE").get_value(default=EnrichmentType.SUMMARY),
)


embedding_job = define_asset_job(
    "embedding_job",
    selection=["*" + wikipedia_articles_embeddings.key.path[0]],
    config=RunConfig(
        ops={
            "wikipedia_articles_with_summaries": openai_pipeline_config,
            "documents_of_wikipedia_articles_with_summaries": openai_pipeline_config,
            "wikipedia_articles_embeddings": OpenAiSettings(
                openai_api_key=EnvVar("OPENAI_API_KEY").get_value("")
            ),
        }
    ),
)
