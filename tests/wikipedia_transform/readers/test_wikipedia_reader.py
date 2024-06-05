from wikipedia_transform.models import wikipedia
from wikipedia_transform.readers import WikipediaReader


def test_read(wikipedia_reader: WikipediaReader) -> None:
    """Test that WikipediaReader.read yields a wikipedia.Article object."""

    assert isinstance(next(iter(wikipedia_reader.read())), wikipedia.Article)
