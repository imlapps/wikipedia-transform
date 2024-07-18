import os
from pathlib import Path

import pytest
from faiss import IndexFlatL2
from langchain.docstore.document import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from etl.models import wikipedia
from etl.models.types import DataFileName, EnrichmentType, ModelResponse, RecordKey
from etl.pipelines import (
    OpenAiEmbeddingPipeline,
    OpenAiRecordEnrichmentPipeline,
    OpenAiRetrievalPipeline,
)
from etl.readers import WikipediaReader
from etl.resources import (
    InputDataFilesConfig,
    OpenAiPipelineConfig,
    OpenAiSettings,
    OutputConfig,
)


@pytest.fixture(scope="session")
def data_file_names() -> tuple[DataFileName, ...]:
    """Return a tuple of data file names."""

    return ("mini-wikipedia.output.txt",)


@pytest.fixture(scope="session")
def input_data_files_config(
    data_file_names: tuple[DataFileName, ...]
) -> InputDataFilesConfig:
    """Return an InputDataFilesConfig object."""

    return InputDataFilesConfig.default(
        data_files_directory_path_default=Path(__file__).parent.parent.absolute()
        / "etl"
        / "data"
        / "input"
        / "data_files",
        data_file_names_default=data_file_names,
    )


@pytest.fixture(scope="session")
def output_config() -> OutputConfig:
    """Return an OutputConfig object."""

    return OutputConfig.default(
        output_directory_path_default=Path(__file__).parent.parent.absolute()
        / "etl"
        / "data"
        / "output"
    )


@pytest.fixture(scope="session")
def wikipedia_reader(
    input_data_files_config: InputDataFilesConfig,
) -> WikipediaReader:
    """Return a WikipediaReaderobject."""

    return WikipediaReader(
        data_file_paths=input_data_files_config.parse().data_file_paths
    )


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
def enrichment_type() -> EnrichmentType:
    """Return a summary enrichment type."""

    return EnrichmentType.SUMMARY


@pytest.fixture(scope="session")
def openai_pipeline_config(
    openai_settings: OpenAiSettings,
    enrichment_type: EnrichmentType,
) -> OpenAiPipelineConfig:
    """Return an OpenAiPipelineConfig object."""

    return OpenAiPipelineConfig(
        openai_settings=openai_settings,
        enrichment_type=enrichment_type,
    )


@pytest.fixture(scope="session")
def openai_record_enrichment_pipeline(
    openai_pipeline_config: OpenAiPipelineConfig,
) -> OpenAiRecordEnrichmentPipeline:
    """Return an OpenAiRecordEnrichmentPipeline object."""

    return OpenAiRecordEnrichmentPipeline(openai_pipeline_config=openai_pipeline_config)


@pytest.fixture(scope="session")
def openai_embedding_pipeline(
    openai_settings: OpenAiSettings, output_config: OutputConfig
) -> OpenAiEmbeddingPipeline:
    """Return an OpenAiEmbedddingPipeline object."""

    return OpenAiEmbeddingPipeline(
        openai_settings=openai_settings, output_config=output_config
    )


@pytest.fixture(scope="session")
def record_key() -> RecordKey:
    """Return a sample record key."""

    return "Mouseion"


@pytest.fixture(scope="session")
def openai_model_response() -> ModelResponse:
    """Return a sample OpenAI summary."""

    return """\
               The Mouseion, established in Alexandria, Egypt, in the 3rd century BCE, was an ancient center of
               learning and research associated with the Library of Alexandria. Founded by Ptolemy I Soter, it
               functioned as a scholarly community akin to a modern university, hosting scholars and scientists.
               The Mouseion featured lecture halls, laboratories, and communal dining for resident scholars, fostering
               intellectual exchange. It significantly contributed to advancements in various fields, including mathematics,
               astronomy, medicine, and literature. The institution's decline began with the Roman conquest and other
               sociopolitical changes, but its legacy endures as a symbol of classical knowledge and scholarship.
           """


@pytest.fixture(scope="session")
def article(record_key: RecordKey) -> wikipedia.Article:
    """Return a wikipedia.Article object."""

    return wikipedia.Article(
        title=record_key,
        url="https://en.wikipedia.org/wiki/" + record_key.replace(" ", "_"),
    )


@pytest.fixture(scope="session")
def article_with_summary(
    article: wikipedia.Article, openai_model_response: ModelResponse
) -> wikipedia.Article:
    """Return a wikipedia.Article object with a set summary field."""

    article.summary = openai_model_response
    return article


@pytest.fixture(scope="session")
def document_of_article_with_summary(
    article_with_summary: wikipedia.Article,
) -> Document:
    """Return a Document of a wikipedia.Article object with a set summary field."""
    return Document(
        page_content=str(article_with_summary.model_dump().get("summary")),
        metadata={
            "source": f"https://en.wikipedia.org/wiki/{article_with_summary.key}"
        },
    )


@pytest.fixture(scope="session")
def tuple_of_articles_with_summaries(
    article_with_summary: wikipedia.Article,
) -> tuple[wikipedia.Article, ...]:
    """Return a tuple of wikipedia.Articles that have summaries."""

    return (article_with_summary,)


@pytest.fixture(scope="session")
def faiss() -> FAISS:
    """Return a FAISS object."""

    return FAISS(
        embedding_function=OpenAIEmbeddings(),
        docstore=InMemoryDocstore(),
        index=IndexFlatL2(42),
        index_to_docstore_id={},
        normalize_L2=False,
    )


@pytest.fixture(scope="session")
def open_ai_retrieval_pipeline(faiss: FAISS) -> OpenAiRetrievalPipeline:
    """Return an OpenAiRetrievalPipeline object."""
    return OpenAiRetrievalPipeline(faiss)
