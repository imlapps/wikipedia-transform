import json
import os
from pathlib import Path

from dagster import asset, define_asset_job
from etl.readers import WikipediaReader
from etl.generative_model_pipelines import OpenAiGenerativeModelPipeline
from etl.embedding_model_pipelines import OpenAiEmbeddingModelPipeline
from etl.models import (
    OpenAiSettings,
    OpenAiPipelineConfig,
    data_files_config_from_env_vars,
    output_config_from_env_vars,
)


@asset
def wikipedia_articles_from_storage():
    """Materialize an asset of Wikipedia articles."""
    return tuple(
        WikipediaReader(data_files_config=data_files_config_from_env_vars).read()
    )


@asset
def wikipedia_articles_with_summaries(
    wikipedia_articles_from_storage, config: OpenAiPipelineConfig
):
    """Materialize an asset of Wikipedia articles with summaries."""

    return tuple(
        OpenAiGenerativeModelPipeline(config).enrich_record(wikipedia_article)
        for wikipedia_article in wikipedia_articles_from_storage
    )


@asset
def wikipedia_articles_with_summaries_to_json(wikipedia_articles_with_summaries):
    """Store the asset of Wikipedia articles with summaries as JSON."""

    output_directory = Path(__file__).parent.absolute() / "data" / "output"
    output_directory.mkdir(exist_ok=True)

    enriched_wikipedia_output_path = (
        output_directory / "wikipedia_articles_with_summaries.txt"
    )

    with enriched_wikipedia_output_path.open(
        mode="w"
    ) as enriched_wikipedia_output_file:
        enriched_wikipedia_articles_as_json = [
            json.dumps(enriched_wikipedia_article.model_dump(by_alias=True))
            for enriched_wikipedia_article in wikipedia_articles_with_summaries
        ]

        enriched_wikipedia_output_file.writelines(enriched_wikipedia_articles_as_json)


@asset
def wikipedia_articles_embeddings(
    wikipedia_articles_with_summaries, config: OpenAiSettings
):
    """Materialize an asset of Wikipedia articles embeddings."""
    return OpenAiEmbeddingModelPipeline(
        openai_settings=config,
        output_config=output_config_from_env_vars,
    ).create_embedding_store(wikipedia_articles_with_summaries)
