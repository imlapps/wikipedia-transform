from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dagster import ConfigurableResource, EnvVar


class OutputConfig(ConfigurableResource):  # type: ignore
    @dataclass(frozen=True)
    class Parsed:
        directory_path: Path

    directory_path: str

    @classmethod
    def default(cls, *, directory_path_default: Path) -> OutputConfig:
        return OutputConfig(directory_path=str(directory_path_default))

    @classmethod
    def from_env_vars(cls, *, directory_path_default: Path) -> OutputConfig:
        return cls(
            directory_path=EnvVar("OUTPUT_DIRECTORY_PATH").get_value(str(directory_path_default)),  # type: ignore
        )

    def parse(self) -> Parsed:
        return OutputConfig.Parsed(directory_path=Path(self.directory_path))


output_config_from_env_vars = OutputConfig.from_env_vars(
    directory_path_default=Path(__file__).parent.parent.absolute() / "data" / "output"
)
