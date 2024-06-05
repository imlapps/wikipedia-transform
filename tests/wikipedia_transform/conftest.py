from pathlib import Path

import pytest

from wikipedia_transform.readers import WikipediaReader


@pytest.fixture(scope="session")
def wikipedia_output_path() -> Path:
    """Return the Path of the Wikipedia output file."""

    return Path(__file__).parent / "data" / "test-wikipedia.output.txt"


@pytest.fixture(scope="session")
def wikipedia_reader(wikipedia_output_path: Path) -> WikipediaReader:
    """Yield a WikipediaReader object."""

    return WikipediaReader(file_path=wikipedia_output_path)
