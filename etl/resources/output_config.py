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

        output_directory_path: Path

        @property
        def openai_embeddings_cache_directory_path(self) -> Path:

            return self.output_directory_path / "openai_embeddings_cache"

        @property
        def record_enrichment_directory_path(self) -> Path:

            return self.output_directory_path / "enriched_records"

        @property
        def wikipedia_articles_with_summaries_file_path(self) -> Path:
            """Return the path of the file that contains Wikipedia articles with summaries."""

            if not self.record_enrichment_directory_path.exists():
                self.record_enrichment_directory_path.mkdir(exist_ok=True)

            return (
                self.record_enrichment_directory_path
                / "wikipedia_articles_with_summaries.jsonl"
            )

    output_directory_path: str

    @classmethod
    def default(cls, *, output_directory_path_default: Path) -> OutputConfig:
        """Return an OutputConfig object."""

        return OutputConfig(output_directory_path=str(output_directory_path_default))

    @classmethod
    def from_env_vars(cls, *, output_directory_path_default: Path) -> OutputConfig:
        """Return an OutputConfig object, with a directory_path obtained from environment variables."""

        return cls(
            output_directory_path=EnvVar("OUTPUT_DIRECTORY_PATH").get_value(
                str(output_directory_path_default)
            ),
        )

    def parse(self) -> Parsed:
        """
        Return a Parsed dataclass object.
        Convert directory_path str into a Path, and store in Parsed.
        """

        return OutputConfig.Parsed(
            output_directory_path=Path(self.output_directory_path)
        )
