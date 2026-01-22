FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/
COPY configs configs/
# Copy DVC config files for runtime model pulling
COPY .dvc/config .dvc/config
# Copy models.dvc (must exist - create with: dvc add models/)
COPY models.dvc models.dvc
# Models directory will be created at runtime if not present
# API will automatically pull models from DVC on startup if models/ doesn't exist

WORKDIR /
RUN uv sync --frozen

# Cloud Run defaults to 8080
EXPOSE 8080

# Environment variable for model name (defaults to EfficientNet)
ENV MODEL_NAME=EfficientNet

# Entrypoint using shell to expand PORT variable
# API will automatically pull models from DVC on startup if they don't exist
ENTRYPOINT ["sh", "-c", "uv run uvicorn mlops_project.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
