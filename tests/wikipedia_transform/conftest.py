from pathlib import Path

import pytest

from wikipedia_transform.embedding_pipelines import OpenAiEmbeddingPipeline
from wikipedia_transform.generative_ai_pipelines import OpenAiPipeline
from wikipedia_transform.models import wikipedia
from wikipedia_transform.models.types import (
    EnhancementType,
    ModelResponse,
    RecordKey,
    RecordType,
)
from wikipedia_transform.readers import WikipediaReader


@pytest.fixture(scope="session")
def wikipedia_output_path() -> Path:
    """Return the Path of the Wikipedia output file."""

    return Path(__file__).parent / "data" / "test-wikipedia.output.txt"


@pytest.fixture(scope="session")
def wikipedia_reader(wikipedia_output_path: Path) -> WikipediaReader:
    """Yield a WikipediaReader object."""

    return WikipediaReader(file_path=wikipedia_output_path)


@pytest.fixture(scope="session")
def record_key() -> RecordKey:
    """Return a sample record key."""

    return "Mouseion"


@pytest.fixture(scope="session")
def record_type() -> RecordType:
    """Return a sample record type."""

    return RecordType.WIKIPEDIA


@pytest.fixture(scope="session")
def open_ai_pipeline() -> OpenAiPipeline:
    """Return an OpenAiPipeline object."""

    return OpenAiPipeline()


@pytest.fixture(scope="session")
def article(record_key: RecordKey) -> wikipedia.Article:
    """Return a Wikipedia Article object."""

    return wikipedia.Article(
        title=record_key,
        url="https://en.wikipedia.org/wiki/" + record_key.replace(" ", "_"),
    )


@pytest.fixture(scope="session")
def tuple_of_article(article: wikipedia.Article) -> tuple[wikipedia.Article, ...]:
    """Return a tuple of Wikipedia articles."""

    return tuple([article])


@pytest.fixture(scope="session")
def open_ai_model_response() -> ModelResponse:
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
def enhancement_type() -> EnhancementType:
    """Return a summary enhancement type."""

    return EnhancementType.SUMMARY


@pytest.fixture(scope="session")
def article_with_summary(
    article: wikipedia.Article, open_ai_model_response: ModelResponse
) -> wikipedia.Article:
    """Return a wikipedia Article with a set summary field."""

    article.summary = open_ai_model_response
    return article


@pytest.fixture(scope="session")
def open_ai_embedding_pipeline() -> OpenAiEmbeddingPipeline:
    return OpenAiEmbeddingPipeline()
