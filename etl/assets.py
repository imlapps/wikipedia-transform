import json
from pathlib import Path

from dagster import asset, define_asset_job

from etl.resources import (OpenAiEmbeddingModelResource,
                           OpenAiGenerativeModelResource,
                           WikipediaReaderResource)


@asset
def wikipedia_articles_from_storage(
    wikipedia_reader_resource: WikipediaReaderResource,
):
    """Materialize an asset of Wikipedia articles."""

    return tuple(wikipedia_reader_resource.read())


@asset
def wikipedia_articles_with_summaries(
    wikipedia_articles_from_storage,
    openai_generative_model_resource: OpenAiGenerativeModelResource,
):
    """Materialize an asset of Wikipedia articles with summaries."""

    enriched_wikipedia_articles = [
        openai_generative_model_resource.enrich_record(wikipedia_article)
        for wikipedia_article in wikipedia_articles_from_storage
    ]

    enriched_wikipedia_output_directory = Path(__file__).parent / "data" / "output"

    with enriched_wikipedia_output_directory.open(
        mode="w"
    ) as enriched_wikipedia_output_file:
        enriched_wikipedia_articles_as_json = [
            json.dumps(enriched_wikipedia_article.model_dump(by_alias=True))
            for enriched_wikipedia_article in enriched_wikipedia_articles
        ]

        enriched_wikipedia_output_file.writelines(enriched_wikipedia_articles_as_json)

    return enriched_wikipedia_articles


@asset
def wikipedia_articles_embeddings(
    wikipedia_articles_with_summaries,
    openai_embedding_model_resource: OpenAiEmbeddingModelResource,
):
    """Materialize an asset of Wikipedia articles embeddings."""

    return openai_embedding_model_resource.create_embedding_store(
        wikipedia_articles_with_summaries
    )


embedding_job = define_asset_job(
    name="embedding_job",
    selection="*wikipedia_articles_embeddings",
)
