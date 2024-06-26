import os
from pathlib import Path

import pytest

from etl.models import OpenAiResourceParams, OpenAiSettings, wikipedia
from etl.models.types import EnrichmentType, ModelResponse, RecordKey, RecordType
from etl.resources import (
    DataFilesConfig,
    OpenAiEmbeddingModelResource,
    OpenAiGenerativeModelResource,
    WikipediaReaderResource,
)


@pytest.fixture(scope="session")
def data_files_config() -> DataFilesConfig:

    return DataFilesConfig.default(frozenset(["mini-wikipedia.output.txt"]))


@pytest.fixture(scope="session")
def wikipedia_reader(data_files_config: DataFilesConfig) -> WikipediaReaderResource:
    """Return a WikipediaReaderobject."""

    return WikipediaReaderResource(data_files_config=data_files_config)


@pytest.fixture(scope="session")
def openai_settings() -> OpenAiSettings:
    """
    Return an OpenAiSettings object.
    Skip all tests that use this fixture if OPENAI_API_KEY is not present in the environment variables.
    """

    if "OPENAI_API_KEY" in os.environ:
        return OpenAiSettings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    pytest.skip(reason="don't have OpenAI key.")


@pytest.fixture(scope="session")
def record_type() -> RecordType:
    """Return a sample record type."""

    return RecordType.WIKIPEDIA


@pytest.fixture(scope="session")
def enrichment_type() -> EnrichmentType:
    """Return a summary enrichment type."""

    return EnrichmentType.SUMMARY


@pytest.fixture(scope="session")
def openai_resource_params(
    openai_settings: OpenAiSettings,
    record_type: RecordType,
    enrichment_type: EnrichmentType,
) -> OpenAiResourceParams:
    """Return an OpenAiResourceParams object."""
    return OpenAiResourceParams(
        openai_settings=openai_settings,
        record_type=record_type,
        enrichment_type=enrichment_type,
    )


@pytest.fixture(scope="session")
def openai_generative_model_resource(
    openai_resource_params: OpenAiResourceParams,
) -> OpenAiGenerativeModelResource:
    """Return an OpenAIGenerativeModelResource object."""

    return OpenAiGenerativeModelResource(openai_resource_params=openai_resource_params)


@pytest.fixture(scope="session")
def openai_embedding_model_resource(
    openai_resource_params: OpenAiResourceParams,
) -> OpenAiEmbeddingModelResource:
    """Return an OpenAIEmbedddingModelResource object."""

    return OpenAiEmbeddingModelResource(openai_resource_params=openai_resource_params)


@pytest.fixture(scope="session")
def record_key() -> RecordKey:
    """Return a sample record key."""

    return "Mouseion"


@pytest.fixture(scope="session")
def article(record_key: RecordKey) -> wikipedia.Article:
    """Return a Wikipedia Article object."""

    return wikipedia.Article(
        title=record_key,
        url="https://en.wikipedia.org/wiki/" + record_key.replace(" ", "_"),
    )


@pytest.fixture(scope="session")
def openai_model_response() -> ModelResponse:
    """Return a sample OpenAI summary response."""

    return """ 
               The Mouseion, established in Alexandria, Egypt, in the 3rd century BCE, was an ancient center of 
               learning and research associated with the Library of Alexandria. Founded by Ptolemy I Soter, it 
               functioned as a scholarly community akin to a modern university, hosting scholars and scientists. 
               The Mouseion featured lecture halls, laboratories, and communal dining for resident scholars, fostering 
               intellectual exchange. It significantly contributed to advancements in various fields, including mathematics, 
               astronomy, medicine, and literature. The institution's decline began with the Roman conquest and other
               sociopolitical changes, but its legacy endures as a symbol of classical knowledge and scholarship.
           """


@pytest.fixture(scope="session")
def article_with_summary(
    article: wikipedia.Article, openai_model_response: ModelResponse
) -> wikipedia.Article:
    """Return a wikipedia Article with a set summary field."""

    article.summary = openai_model_response
    return article


@pytest.fixture(scope="session")
def tuple_of_article_with_summary(
    article_with_summary: wikipedia.Article,
) -> tuple[wikipedia.Article, ...]:
    """Return a tuple of Wikipedia articles."""

    return (article_with_summary,)
