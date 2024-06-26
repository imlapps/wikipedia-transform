from etl.readers import WikipediaReader
from etl.models import wikipedia


def test_read(wikipedia_reader: WikipediaReader) -> None:
    """Test that WikipediaReader.read yields wikipedia.Article objects."""

    assert isinstance(next(iter(wikipedia_reader.read())), wikipedia.Article)
