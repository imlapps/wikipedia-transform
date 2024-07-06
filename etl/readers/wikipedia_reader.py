import json
from collections.abc import Iterable
from pathlib import Path

from unidecode import unidecode

from etl.models import wikipedia
from etl.readers import Reader
from etl.resources import InputDataFilesConfig


class WikipediaReader(Reader):
    """
    A concrete implementation of Reader.

    Read in Wikipedia data and yield them as wikipedia.Articles.
    """

    def __init__(self, data_file_paths: frozenset[Path]) -> None:
        self.wikipedia_jsonl_file_paths = data_file_paths

    def read(self) -> Iterable[wikipedia.Article]:
        """Read in Wikipedia data and yield them as wikipedia.Articles."""

        for wikipedia_jsonl_file_path in self.wikipedia_jsonl_file_paths:
            if wikipedia_jsonl_file_path:
                with wikipedia_jsonl_file_path.open(encoding="utf-8") as json_file:

                    for json_line in json_file:
                        record_json = json.loads(json_line)

                        if record_json["type"] != "RECORD":
                            continue

                        json_obj = json.loads(
                            unidecode(
                                json.dumps(record_json["record"], ensure_ascii=False)
                            )
                        )

                        yield wikipedia.Article(**(json_obj["abstract_info"]))
