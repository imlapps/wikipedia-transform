from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dagster import ConfigurableResource, EnvVar

from etl.models.types import DataFileName


class DataFilesConfig(ConfigurableResource):  # type: ignore
    @dataclass(frozen=True)
    class Parsed:
        data_file_paths: frozenset[Path]

    data_file_names: list[DataFileName]

    @classmethod
    def default(
        cls, *, data_file_names_default: tuple[DataFileName, ...]
    ) -> DataFilesConfig:
        return DataFilesConfig(data_file_names=list(data_file_names_default))

    @classmethod
    def from_env_vars(
        cls, *, data_file_names_default: tuple[DataFileName, ...]
    ) -> DataFilesConfig:
        return cls(
            data_file_names=json.loads(
                EnvVar("DATA_FILE_NAMES").get_value(
                    json.dumps(list(data_file_names_default))
                )
            )
        )

    def parse(self) -> Parsed:

        return DataFilesConfig.Parsed(
            data_file_paths=frozenset(
                [
                    Path(__file__).parent.parent.absolute() / "data" / data_file_name
                    for data_file_name in self.data_file_names
                ]
            )
        )


data_files_config_from_env_vars: DataFilesConfig = DataFilesConfig.from_env_vars(
    data_file_names_default=("mini-wikipedia.output.txt",)
)
