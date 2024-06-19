import os
import pytest 

from etl.models import OpenAiSettings, wikipedia
from etl.models.types import RecordType, RecordKey, EnhancementType, ModelResponse
from etl.resources import WikipediaReaderResource, OpenAiGenerativeModelResource

@pytest.fixture(scope="session")
def wikipedia_reader_resource() -> WikipediaReaderResource:
    """Return a WikipediaReaderResource object."""

    return WikipediaReaderResource(data_file_names=["mini-wikipedia.output.txt"])

@pytest.fixture(scope="session")
def open_ai_settings() -> OpenAiSettings:
    """
        Return an OpenAISettings object. 
        Skip all tests that use this fixture if API_KEY is not present in the environment variables.
    """

    if "OPENAI_API_KEY" in os.environ:
        return OpenAiSettings(
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        )
    
    pytest.skip(reason = "don't have OpenAI key.")



@pytest.fixture(scope="session")
def record_type() -> RecordType:
    """Return a sample record type."""

    return RecordType.WIKIPEDIA

@pytest.fixture(scope="session")
def enhancement_type() -> EnhancementType:
    """Return a summary enhancement type."""

    return EnhancementType.SUMMARY

@pytest.fixture(scope="session")
def open_ai_generative_model_resource(open_ai_settings: OpenAiSettings, record_type: RecordType, enhancement_type: EnhancementType) -> OpenAiGenerativeModelResource:
    """Return an OpenAIGenerativeModelResource object."""
    
    return OpenAiGenerativeModelResource(
        open_ai_settings=open_ai_settings,
        record_type=record_type,
        enhancement_type=enhancement_type,
    )


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
def article_with_summary(
    article: wikipedia.Article, open_ai_model_response: ModelResponse
) -> wikipedia.Article:
    """Return a wikipedia Article with a set summary field."""

    article.summary = open_ai_model_response
    return article

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
