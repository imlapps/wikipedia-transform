import json

from dagster import asset
from langchain_core.vectorstores import VectorStore

from etl.embedding_model_pipelines import OpenAiEmbeddingModelPipeline
from etl.generative_model_pipelines import OpenAiGenerativeModelPipeline
from etl.models import (
    DocumentTuple,
    OpenAiPipelineConfig,
    OpenAiSettings,
    RecordTuple,
    data_files_config_from_env_vars,
    output_config_from_env_vars,
)
from etl.readers import WikipediaReader
from etl.utils import create_documents


@asset
def wikipedia_articles_from_storage() -> RecordTuple:
    """Materialize an asset of Wikipedia articles."""

    return RecordTuple(
        records=tuple(
            WikipediaReader(data_files_config=data_files_config_from_env_vars).read()
        )
    )


@asset
def wikipedia_articles_with_summaries(
    wikipedia_articles_from_storage: RecordTuple, config: OpenAiPipelineConfig
) -> RecordTuple:
    """Materialize an asset of Wikipedia articles with summaries."""

    return RecordTuple(
        records=tuple(
            OpenAiGenerativeModelPipeline(config).enrich_record(wikipedia_article)
            for wikipedia_article in wikipedia_articles_from_storage.records
        )
    )


@asset
def wikipedia_articles_with_summaries_json_file(
    wikipedia_articles_with_summaries: RecordTuple,
) -> None:
    """Store the asset of Wikipedia articles with summaries as JSON."""

    output_directory_path = output_config_from_env_vars.parse().directory_path

    output_directory_path.mkdir(exist_ok=True)

    enriched_wikipedia_output_file_path = (
        output_directory_path / "wikipedia_articles_with_summaries.txt"
    )

    with enriched_wikipedia_output_file_path.open(
        mode="w"
    ) as enriched_wikipedia_output_file:

        enriched_wikipedia_output_file.writelines(
            [
                json.dumps(enriched_wikipedia_article.model_dump(by_alias=True))
                for enriched_wikipedia_article in wikipedia_articles_with_summaries.records
            ]
        )


@asset
def documents_of_wikipedia_articles_with_summaries(
    wikipedia_articles_with_summaries: RecordTuple, config: OpenAiPipelineConfig
) -> DocumentTuple:
    """Materialize an asset of Documents of Wikipedia articles with summaries."""

    return DocumentTuple(
        documents=create_documents(
            records=wikipedia_articles_with_summaries.records,
            record_type=config.record_type,
            enrichment_type=config.enrichment_type,
        )
    )


@asset
def wikipedia_articles_embeddings(
    documents_of_wikipedia_articles_with_summaries: DocumentTuple,
    config: OpenAiSettings,
) -> VectorStore:
    """Materialize an asset of Wikipedia articles embeddings."""

    return OpenAiEmbeddingModelPipeline(
        openai_settings=config,
        output_config=output_config_from_env_vars,
    ).create_embedding_store(documents_of_wikipedia_articles_with_summaries.documents)
