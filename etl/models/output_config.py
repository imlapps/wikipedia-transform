from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dagster import ConfigurableResource, EnvVar


class OutputConfig(ConfigurableResource):  # type: ignore[misc]
    """
    A ConfigurableResource that holds the output path of the ETL.
    """

    @dataclass(frozen=True)
    class Parsed:
        """A dataclass that contains the output directory Path of the ETL."""

        directory_path: Path

    directory_path: str

    @classmethod
    def default(cls, *, directory_path_default: Path) -> OutputConfig:
        """Return an OutputConfig object."""

        return OutputConfig(directory_path=str(directory_path_default))

    @classmethod
    def from_env_vars(cls, *, directory_path_default: Path) -> OutputConfig:
        """Return an OutputConfig object, with a directory_path obtained from environment variables."""

        return cls(
            directory_path=EnvVar("OUTPUT_DIRECTORY_PATH").get_value(
                str(directory_path_default)
            ),
        )

    def parse(self) -> Parsed:
        """
        Return a Parsed dataclass object.
        Convert directory_path str into a Path, and store in Parsed.
        """

        return OutputConfig.Parsed(directory_path=Path(self.directory_path))


output_config_from_env_vars = OutputConfig.from_env_vars(
    directory_path_default=Path(__file__).parent.parent.absolute() / "data" / "output"
)
