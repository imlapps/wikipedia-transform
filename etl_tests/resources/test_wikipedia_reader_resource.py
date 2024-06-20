from pathlib import Path

from etl.models import wikipedia
from etl.resources import WikipediaReaderResource


def test_wikipedia_reader_resource(
    wikipedia_reader_resource: WikipediaReaderResource,
) -> None:
    """Test that WikipediaReaderResource.read yields wikipedia.Article objects."""

    assert isinstance(next(iter(wikipedia_reader_resource.read())), wikipedia.Article)
