#!/bin/bash 

set -e

cd "$(dirname "$0")/.."

poetry run black --check wikipedia_transform
poetry run mypy wikipedia_transform
poetry run ruff check wikipedia_transform


mkdir -p test-results
poetry run pytest --junitxml=test-results/junit.xml