FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/
COPY configs configs/
# Ensure models are copied (User must run 'dvc pull' first!)
COPY models models/

WORKDIR /
RUN uv sync --frozen

# Cloud Run defaults to 8080
EXPOSE 8080

# Environment variable for model path inside container
ENV MODEL_PATH=/models/demo_model.onnx

# Entrypoint using shell to expand PORT variable
ENTRYPOINT ["sh", "-c", "uv run uvicorn mlops_project.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
