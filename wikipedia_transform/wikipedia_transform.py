import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import StrictBool

from wikipedia_transform import RecordTransform
from wikipedia_transform.embedding_pipelines import (
    EmbeddingPipeline,
    OpenAiEmbeddingPipeline,
)
from wikipedia_transform.generative_ai_pipelines import GenAiPipeline, OpenAiPipeline
from wikipedia_transform.models import Record, settings, wikipedia
from wikipedia_transform.models.types import (
    EmbeddingModelType,
    EnhancementType,
    RecordType,
)
from wikipedia_transform.readers import WikipediaReader


class WikipediaTransform(RecordTransform):
    """A transformer for Wikipedia articles.

    WikipediaTransform takes in a frozenset of file paths, and uses an embedding model pipeline to convert Wikipedia articles 
    stored at the file paths into embeddings. When enhancements are set in the environment variables, WikipediaTransform uses a generative AI pipeline to
    enhance the articles before they are converted into embeddings.

    """

    def __init__(self, file_paths: frozenset[Path]):
        self.__file_paths: frozenset[Path] = file_paths
        self.__embedding_pipeline: EmbeddingPipeline | None = (
            self.__select_embedding_pipeline()
        )

    def __select_embedding_pipeline(self) -> EmbeddingPipeline | None:
        """Select and return an embedding model type based on environment variables."""

        if (
            settings.embedding_model_type == EmbeddingModelType.OPEN_AI
            and settings.openai_api_key
        ):
            return OpenAiEmbeddingPipeline()

        return None

    def __select_generative_ai_pipeline(self) -> GenAiPipeline | None:
        """Select and return a generative ai model based on environmnet variables."""

        if settings.generative_model_type and settings.openai_api_key:
            return OpenAiPipeline()

        return None

    def __enhance_articles_with_summaries(
        self,
        *,
        generative_ai_pipeline: GenAiPipeline,
        wikipedia_articles: tuple[wikipedia.Article, ...],
        store: StrictBool = True
    ) -> Iterable[Record]:
        """Use a generative AI model to enhance Wikipedia articles with summaries."""

        enhanced_wikipedia_articles = [
            next(
                iter(
                    generative_ai_pipeline.enhance_record(
                        record=wikipedia_article,
                        record_type=RecordType.WIKIPEDIA,
                        enhancement_type=EnhancementType.SUMMARY,
                    )
                )
            )
            for wikipedia_article in wikipedia_articles
        ]

        if store:
            summary_store_path = (
                settings.output_path / "summaries_of_wikipedia_articles.txt"
            )

            with summary_store_path.open(mode="w") as summary_output:
                enhanced_wikipedia_articles_as_json = [
                    json.dumps(
                        enhanced_wikipedia_article.model_dump(by_alias=True))
                    for enhanced_wikipedia_article in enhanced_wikipedia_articles
                ]

                summary_output.writelines(enhanced_wikipedia_articles_as_json)

        yield from enhanced_wikipedia_articles

    def transform(self):
        """Transform Wikipedia articles into embeddings."""

        wikipedia_articles = [
            WikipediaReader(file_path).read() for file_path in self.__file_paths
        ]

        if settings.enhancements:
            generative_ai_pipeline = self.__select_generative_ai_pipeline()

            if generative_ai_pipeline:
                for enhancement in settings.enhancements:
                    if enhancement == EnhancementType.SUMMARY:
                        wikipedia_articles = self.__enhance_articles_with_summaries(
                            generative_ai_pipeline, tuple(wikipedia_articles)
                        )

        self.__embedding_pipeline.create_embedding_store(
            tuple(wikipedia_articles))
