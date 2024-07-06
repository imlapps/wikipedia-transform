import json

from dagster import asset
from langchain_core.vectorstores import VectorStore

from etl.models import DocumentTuple, RecordTuple
from etl.pipelines import OpenAiEmbeddingPipeline, OpenAiRecordEnrichmentPipeline
from etl.readers import WikipediaReader
from etl.resources import (
    InputDataFilesConfig,
    OpenAiPipelineConfig,
    OpenAiSettings,
    OutputConfig,
)


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
    wikipedia_articles_from_storage: RecordTuple,
    openai_pipeline_config: OpenAiPipelineConfig,
) -> RecordTuple:
    """Materialize an asset of Wikipedia articles with summaries."""

    return RecordTuple(
        records=tuple(
            OpenAiRecordEnrichmentPipeline(openai_pipeline_config).enrich_record(
                wikipedia_article
            )
            for wikipedia_article in wikipedia_articles_from_storage.records
        )
    )


@asset
def wikipedia_articles_with_summaries_json_file(
    wikipedia_articles_with_summaries: RecordTuple, output_config: OutputConfig
) -> None:
    """Store the asset of Wikipedia articles with summaries as JSON."""

    with output_config.parse().wikipedia_articles_with_summaries_file_path.open(
        mode="w"
    ) as wikipedia_articles_with_summaries_file:

        wikipedia_articles_with_summaries_file.writelines(
            [
                json.dumps(enriched_wikipedia_article.model_dump(by_alias=True))
                for enriched_wikipedia_article in wikipedia_articles_with_summaries.records
            ]
        )


@asset
def documents_of_wikipedia_articles_with_summaries(
    wikipedia_articles_with_summaries: RecordTuple,
    openai_pipeline_config: OpenAiPipelineConfig,
) -> DocumentTuple:
    """Materialize an asset of Documents of Wikipedia articles with summaries."""

    return DocumentTuple.from_records(
        records=wikipedia_articles_with_summaries.records,
        record_type=openai_pipeline_config.record_type,
        enrichment_type=openai_pipeline_config.enrichment_type,
    )


@asset
def wikipedia_articles_embeddings(
    documents_of_wikipedia_articles_with_summaries: DocumentTuple,
    openai_settings: OpenAiSettings,
    output_config: OutputConfig,
) -> VectorStore:
    """Materialize an asset of Wikipedia articles embeddings."""

    return OpenAiEmbeddingPipeline(
        openai_settings=openai_settings,
        output_config=output_config,
    ).create_embedding_store(documents_of_wikipedia_articles_with_summaries.documents)
