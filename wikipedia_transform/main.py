from wikipedia_transform import WikipediaTransform
from wikipedia_transform.models import settings
from wikipedia_transform.models.types import RecordType


def main() -> None:
    """
    Invoke WikipediaTransform.transform() to transform Wikipedia data files into embeddings.
    """
    if settings.record_type == RecordType.WIKIPEDIA:
        WikipediaTransform(settings.data_file_paths).transform()


if __name__ == "__main__":
    main()
