#!/bin/bash 

set -e

cd "$(dirname "$0")/.."

poetry run black --check etl
poetry run mypy etl
poetry run ruff check etl


mkdir -p test-results
poetry run pytest --cov=etl_tests --cov-report=term-missing:skip-covered --junitxml=test-results/junit.xml -p no:warnings tests| tee test-results/coverage.txt