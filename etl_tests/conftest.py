import pytest 

from etl.resources import WikipediaReaderResource

@pytest.fixture(scope="session")
def wikipedia_reader_resource() -> WikipediaReaderResource:
    """Return a WikipediaReaderResource object."""

    return WikipediaReaderResource(data_file_names=["mini-wikipedia.output.txt"])