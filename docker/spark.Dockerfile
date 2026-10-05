FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS base

RUN pip install --no-cache-dir uv==0.9.7
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless ca-certificates \
    && ln -s /usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture) /opt/java \
    && rm -rf /var/lib/apt/lists/*

FROM base AS runtime

ENV JAVA_HOME=/opt/java \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    SPARK_LOCAL_IP=127.0.0.1 \
    PYSPARK_PYTHON=/app/.venv/bin/python \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY src ./src
COPY configs ./configs
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen \
    && useradd --create-home --uid 1000 sparkrec \
    && mkdir -p /app/data /app/artifacts \
    && chown sparkrec:sparkrec /app /app/data /app/artifacts
USER sparkrec
ENTRYPOINT ["sparkrec"]
CMD ["--help"]

FROM runtime AS test
RUN ruff check src tests && ruff format --check src tests \
    && pytest -q --junitxml=/app/artifacts/tests.xml \
    && sparkrec ingest --config configs/fixture.yaml --run-id docker-fixture

FROM scratch AS test-results
COPY --from=test /app/artifacts/ /

FROM runtime AS development
