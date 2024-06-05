import pytest

from wikipedia_transform import WikipediaTransform


@pytest.fixture(scope="session")
def wikipedia_transform() -> WikipediaTransform:
    return WikipediaTransform()


def test_transform(wikipedia_transform: WikipediaTransform) -> None:
    """Test that WikipediaTransform.transform performs the necessary operations."""
    pass
