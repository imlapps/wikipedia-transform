from dagster import EnvVar, RunConfig, define_asset_job

from .assets import wikipedia_articles_embeddings, wikipedia_articles_with_summaries
from .models import OpenAiPipelineConfig, OpenAiSettings
from .models.types import RecordType, EnrichmentType

embedding_job = define_asset_job(
    "embedding_job",
    selection=["*" + wikipedia_articles_embeddings.key.path[0]],
    config=RunConfig(
        ops={
            "wikipedia_articles_with_summaries": OpenAiPipelineConfig(
                openai_settings=OpenAiSettings(
                    openai_api_key=EnvVar("OPENAI_API_KEY").get_value("")
                ),
                record_type=EnvVar("RECORD_TYPE").get_value(
                    default=RecordType.WIKIPEDIA
                ),
                enrichment_type=EnvVar("ENRICHMENT_TYPE").get_value(
                    default=EnrichmentType.SUMMARY
                ),
            ),
            "wikipedia_articles_embeddings": OpenAiSettings(
                openai_api_key=EnvVar("OPENAI_API_KEY").get_value("")
            ),
        }
    ),
)
