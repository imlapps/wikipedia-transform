from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from dagster import ConfigurableResource, EnvVar
from langchain_community.vectorstores.utils import DistanceStrategy
from pydantic import Field

if TYPE_CHECKING:
    from etl.models.types import DataFileName, Iri, ScoreThreshold


class InputConfig(ConfigurableResource):  # type: ignore[misc]
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
        distance_strategy: DistanceStrategy
        score_threshold: ScoreThreshold
        etl_base_iri: Iri

    data_files_directory_path: str
    data_file_names: list[str]
    distance_strategy: str
    score_threshold: float
    etl_base_iri: str

    @classmethod
    def default(
        cls,
        *,
        data_files_directory_path_default: Path,
        data_file_names_default: tuple[DataFileName, ...],
        distance_strategy_default: DistanceStrategy,
        score_threshold_default: ScoreThreshold,
        etl_base_iri_default: Iri,
    ) -> InputConfig:
        """Return an InputConfig object using only default parameters."""

        return InputConfig(
            data_files_directory_path=str(data_files_directory_path_default),
            data_file_names=list(data_file_names_default),
            distance_strategy=distance_strategy_default.value,
            score_threshold=score_threshold_default,
            etl_base_iri=etl_base_iri_default,
        )

    @classmethod
    def from_env_vars(
        cls,
        *,
        data_files_directory_path_default: Path,
        data_file_names_default: tuple[DataFileName, ...],
        distance_strategy_default: DistanceStrategy,
        score_threshold_default: ScoreThreshold,
        etl_base_iri_default: Iri,
    ) -> InputConfig:
        """Return an InputConfig object, with parameter values obtained from environment variables."""

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
            distance_strategy=EnvVar("DISTANCE_STRATEGY").get_value(
                distance_strategy_default.value
            ),
            score_threshold=EnvVar("SCORE_THRESHOLD").get_value(
                score_threshold_default
            ),
            etl_base_iri=EnvVar("ETL_BASE_IRI").get_value(etl_base_iri_default),
        )

    def parse(self) -> Parsed:
        """
        Return a Parsed dataclass object that contains a data_files_directory_path
        that has been converted to a Path, and a list of data_file_names that have been
        converted to a frozenset of Paths.
        """

        return InputConfig.Parsed(
            data_files_directory_path=Path(self.data_files_directory_path),
            data_file_paths=frozenset(
                [
                    Path(self.data_files_directory_path) / data_file_name
                    for data_file_name in self.data_file_names
                ]
            ),
            distance_strategy=self.distance_strategy,
            score_threshold=self.score_threshold,
            etl_base_iri=self.etl_base_iri,
        )
