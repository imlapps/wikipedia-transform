import json

from dagster import asset, EnvVar
from langchain_core.vectorstores import VectorStore

from etl.pipelines import OpenAiEmbeddingPipeline, RecordEnrichmentPipeline

from etl.models import DocumentTuple, RecordTuple
from etl.readers import WikipediaReader
from etl.resources import InputDataFilesConfig
from etl.resources.open_ai_pipeline_config import OpenAiPipelineConfig
from etl.resources.open_ai_settings import OpenAiSettings
from etl.utils import create_documents


@asset
def wikipedia_articles_from_storage(
    input_data_files_config: InputDataFilesConfig,
) -> RecordTuple:
    """Materialize an asset of Wikipedia articles."""

    return RecordTuple(
        records=tuple(
            WikipediaReader(
                data_file_paths=input_data_files_config.parse().data_file_paths
            ).read()
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

    # output_directory_path = output_config_from_env_vars.parse()

    # output_directory_path.directory_path.mkdir(exist_ok=True)

    # with output_directory_path.wikipedia_articles_with_summaries_file_path.open(
    #     mode="w"
    # ) as wikipedia_articles_with_summaries_file:

    #     wikipedia_articles_with_summaries_file.writelines(
    #         [
    #             json.dumps(enriched_wikipedia_article.model_dump(by_alias=True))
    #             for enriched_wikipedia_article in wikipedia_articles_with_summaries.records
    #         ]
    #     )


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

    # return OpenAiEmbeddingModelPipeline(
    #     openai_settings=config,
    #     output_config=output_config_from_env_vars,
    # ).create_embedding_store(documents_of_wikipedia_articles_with_summaries.documents)
