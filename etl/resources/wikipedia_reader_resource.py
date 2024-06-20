import json
from collections.abc import Iterable
from pathlib import Path

from dagster import ConfigurableResource
from unidecode import unidecode

from etl.models import wikipedia
from etl.models.types import DataFileName


class WikipediaReaderResource(ConfigurableResource):  # type: ignore
    """A ConfigurableResource to read Wikipedia data from storage."""

    data_file_names: list[DataFileName]

    def read(self) -> Iterable[wikipedia.Article]:
        """Read in Wikipedia data and yield them as wikipedia.Article objects."""

        for data_file_name in self.data_file_names:

            data_file_path = Path(__file__).parent.parent / "data" / data_file_name

            if data_file_path:
                with data_file_path.open(encoding="utf-8") as json_file:

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
