import json
from pathlib import Path
from etl.readers import Reader
from etl.models import wikipedia, DataFilesConfig
from collections.abc import Iterable
from dagster import ConfigurableResource
from unidecode import unidecode


class WikipediaReader(Reader):
    """
    A concrete implementation of Reader.

    Read in Wikipedia data and yield them as wikipedia.Article objects.
    """

    def __init__(self, data_files_config: DataFilesConfig) -> None:
        self.__parsed_data_files_config: DataFilesConfig.Parsed = (
            data_files_config.parse()
        )

    def read(self) -> Iterable[wikipedia.Article]:
        """Read in Wikipedia data and yield them as wikipedia.Article objects."""

        for data_file_path in self.__parsed_data_files_config.data_file_paths:
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
