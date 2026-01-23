# CI/CD Pipeline

## GitHub Actions Workflows

### Tests (`tests.yaml`)
- Runs on every push and PR
- Tests on Ubuntu, macOS, Windows
- Python 3.12

### Docker (`docker.yaml`)
- Builds Docker images on push to main
- Pushes to Google Container Registry

### Documentation (`deploy_docs.yaml`)
- Builds and deploys docs to GitHub Pages
- Triggers on changes to `docs/`

## Workflow Overview

```
Push to main
    │
    ├── Run tests (all platforms)
    │
    ├── Build Docker image
    │   └── Push to GCR
    │
    ├── Deploy to Cloud Run
    │
    └── Update documentation
```

## Local CI Simulation

```bash
# Run what CI runs
uv run pytest tests/ -v
uv run ruff check src/ tests/
docker build -f dockerfiles/api.dockerfile -t test .
```
