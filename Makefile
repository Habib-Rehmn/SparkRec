.PHONY: help setup build test test-unit lint smoke download ingest fixture demo-small

export SPARKREC_UID ?= $(shell id -u)
export SPARKREC_GID ?= $(shell id -g)
COMPOSE := docker compose

help:
	@echo "setup       Build the development image and create output directories"
	@echo "test        Build the test stage (lint, unit, Spark integration, ALS smoke)"
	@echo "test-unit   Run unit tests in the development container"
	@echo "smoke       Check Spark and ALS in the development container"
	@echo "fixture     Ingest the bundled synthetic dataset"
	@echo "download    Download/cache and verify MovieLens small"
	@echo "ingest      Validate MovieLens small and write Parquet"
	@echo "demo-small  Download then ingest MovieLens small (M1 data demo)"

setup:
	mkdir -p data artifacts
	$(MAKE) build

build:
	docker build --target development -t sparkrec:dev -f docker/spark.Dockerfile .

test:
	docker build --target test -t sparkrec:test -f docker/spark.Dockerfile .

test-unit:
	$(COMPOSE) run --rm --entrypoint pytest pipeline -q -m "not spark"

lint:
	$(COMPOSE) run --rm --entrypoint ruff pipeline check src tests

smoke:
	$(COMPOSE) run --rm pipeline smoke --config configs/fixture.yaml

download:
	$(COMPOSE) run --rm pipeline download --config configs/dev.yaml

ingest:
	$(COMPOSE) run --rm pipeline ingest --config configs/dev.yaml

fixture:
	$(COMPOSE) run --rm pipeline ingest --config configs/fixture.yaml

demo-small: download
	$(MAKE) ingest
