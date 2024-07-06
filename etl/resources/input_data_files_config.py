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
    A ConfigurableResource that holds the directory path of input data files,
    and a list of data file names.
    """

    @dataclass(frozen=True)
    class Parsed:
        """
        A dataclass that contains the directory path of data files,
        and a frozenset of data file paths.
        """

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
        """Return an InputDataFilesConfig object using only default parameters."""

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
        """Return an InputDataFilesConfig object, with parameter values obtained from environment variables."""

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
        """
        Return a Parsed dataclass object that contains a data_files_directory_path
        that has been converted to a Path, and a list of data_file_names that have been
        converted to a frozenset of Paths.
        """

        return InputDataFilesConfig.Parsed(
            data_files_directory_path=Path(self.data_files_directory_path),
            data_file_paths=frozenset(
                [
                    Path(self.data_files_directory_path) / data_file_name
                    for data_file_name in self.data_file_names
                ]
            ),
        )
