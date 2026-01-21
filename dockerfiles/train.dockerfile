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
COPY configs configs/

# 3. Copy DVC configuration files for data access from GCS and model tracking
# Only copy config, not the cache (cache is huge and not needed in image)
COPY .dvc/config .dvc/config
COPY data.dvc data.dvc
COPY raw.dvc raw.dvc
COPY models.dvc models.dvc

# 4. Copy entrypoint script
COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 5. Change work dir and sync dependencies with uv
WORKDIR /
RUN uv sync --frozen

# 6. Set entrypoint to use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]

# Docker commands to build and run the training container:
# docker build -f dockerfiles/train.dockerfile -t train:latest .
# docker run --name experimentName -v ${PWD}/models:/models/ -v ${PWD}/data/raw:/data/raw -v ${PWD}/data/processed:/data/processed train:latest
