from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dagster import ConfigurableResource, EnvVar

if TYPE_CHECKING:
    from etl.models.types import DataFileName


class InputDataFilesConfig(ConfigurableResource):  # type: ignore[misc]
    """
    A ConfigurableResource that holds the Paths of data files.
    """

    @dataclass(frozen=True)
    class Parsed:
        # """A dataclass that contains a frozenset of data file paths."""

        data_files_directory_path: Path
        data_file_paths: frozenset[Path]

    data_files_directory_path: str
    data_file_names: list[str]

    @classmethod
    def default(
        cls,
        *,
        data_files_directory_path_default: Path,
        data_file_names_default: tuple[DataFileName, ...],
    ) -> InputDataFilesConfig:
        """Return an InputConfig object."""

        return InputDataFilesConfig(
            data_files_directory_path=str(data_files_directory_path_default),
            data_file_names=list(data_file_names_default),
        )

    @classmethod
    def from_env_vars(
        cls,
        *,
        data_files_directory_path_default: Path,
        data_file_names_default: tuple[DataFileName, ...],
    ) -> InputDataFilesConfig:
        # """Return a DataFilesConfig object, with data_file_names obtained from environment variables."""

        return cls(
            data_files_directory_path=EnvVar("DATA_FILES_DIRECTORY_PATH").get_value(
                str(data_files_directory_path_default)
            ),
            data_file_names=json.loads(
                str(
                    EnvVar("DATA_FILE_NAMES").get_value(
                        json.dumps(list(data_file_names_default))
                    )
                )
            ),
        )

    def parse(self) -> Parsed:

        # Return a Parsed dataclass object.
        # Convert list of file names into frozenset of Paths, and store in Parsed.

        return InputDataFilesConfig.Parsed(
            data_files_directory_path=Path(self.data_files_directory_path),
            data_file_paths=frozenset(
                [
                    Path(self.data_files_directory_path) / data_file_name
                    for data_file_name in self.data_file_names
                ]
            ),
        )
