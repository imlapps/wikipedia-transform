import json

from dagster import asset

from etl.models import AntiRecommendationsByKeyTuple, DocumentTuple, RecordTuple
from etl.pipelines import (
    AntiRecommendationRetrievalPipeline,
    OpenAiEmbeddingPipeline,
    OpenAiRecordEnrichmentPipeline,
)
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
            json.dumps(enriched_wikipedia_article.model_dump(by_alias=True))
            for enriched_wikipedia_article in wikipedia_articles_with_summaries.records
        )


@asset
def documents_of_wikipedia_articles_with_summaries(
    wikipedia_articles_with_summaries: RecordTuple,
) -> DocumentTuple:
    """Materialize an asset of Documents of Wikipedia articles with summaries."""

    return DocumentTuple.from_records(
        records=wikipedia_articles_with_summaries.records,
        record_content=lambda record: str(record.model_dump().get("summary")),
    )


@asset
def wikipedia_articles_embedding_store(
    documents_of_wikipedia_articles_with_summaries: DocumentTuple,
    openai_settings: OpenAiSettings,
    output_config: OutputConfig,
) -> None:
    """Materialize an asset of Wikipedia articles embeddings."""

    OpenAiEmbeddingPipeline(
        openai_settings=openai_settings,
        output_config=output_config,
    ).create_embedding_store(documents_of_wikipedia_articles_with_summaries.documents)


@asset
def retrievals_of_wikipedia_anti_recommendations(
    wikipedia_articles_from_storage: RecordTuple,
    documents_of_wikipedia_articles_with_summaries: DocumentTuple,
    openai_settings: OpenAiSettings,
    output_config: OutputConfig,
) -> AntiRecommendationsByKeyTuple:
    """Materialize an asset of Wikipedia anti-recommendations."""

    vector_store = OpenAiEmbeddingPipeline(
        openai_settings=openai_settings,
        output_config=output_config,
    ).create_embedding_store(documents_of_wikipedia_articles_with_summaries.documents)

    return AntiRecommendationsByKeyTuple(
        anti_recommendations_by_key=tuple(
            {
                record.key: tuple(
                    document_float_tuple[0].metadata["source"][30:]
                    for document_float_tuple in AntiRecommendationRetrievalPipeline(
                        vector_store
                    ).retrieve_documents(record_key=record.key, k=6)
                )
            }
            for record in wikipedia_articles_from_storage.records
        )
    )


@asset
def retrievals_of_wikipedia_anti_recommendations_json_file(
    retrievals_of_wikipedia_anti_recommendations: AntiRecommendationsByKeyTuple,
    output_config: OutputConfig,
) -> None:
    """Store the asset of Wikipedia anti-recommendations as JSON."""

    with output_config.parse().wikipedia_anti_recommendations_file_path.open(
        mode="w"
    ) as wikipedia_anti_recommendations_file:
        wikipedia_anti_recommendations_file.writelines(
            json.dumps(anti_recommendations_by_key_dict)
            for anti_recommendations_by_key_dict in retrievals_of_wikipedia_anti_recommendations.anti_recommendations_by_key
        )
