from pathlib import Path
from etl.resources import WikipediaReaderResource
from etl.models import wikipedia

def test_wikipedia_reader_resource(wikipedia_reader_resource: WikipediaReaderResource) -> None:
    """Test that WikipediaReaderResource.read yields wikipedia.Article objects."""

    assert isinstance(next(iter(wikipedia_reader_resource.read())), wikipedia.Article)