# 1. Use Debian-based image (better compatibility with PyTorch and scientific packages)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 2. Copy project files
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src/

# 3. Change work dir and sync dependencies with uv
WORKDIR /
RUN uv sync --frozen

# 4. Set entrypoint to training script
ENTRYPOINT ["uv", "run", "src/mlops_project/evaluate.py"]

# Docker commands to build and run the training container:
# docker build -f dockerfiles/evaluate.dockerfile -t evaluate:latest .
# docker run --name experimentName -v ${PWD}/models:/models/ -v ${PWD}/data/raw:/data/raw -v ${PWD}/data/processed:/data/processed evaluate:latest