import os

from invoke.context import Context
from invoke.tasks import task

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.12"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def create_demo_model(ctx: Context) -> None:
    """Create a dummy ONNX model for testing the pipeline."""
    ctx.run("uv run python -m mlops_project.create_demo_model", echo=True, pty=not WINDOWS)


@task
def deploy_to_cloud_run(
    ctx: Context, project_id: str, region: str = "europe-west1", artifact_registry: str = "dtu-vibe-ops"
) -> None:
    """Deploy API to Google Cloud Run."""
    print(f"Deploying to project: {project_id}, region: {region}, registry: {artifact_registry}")

    # Configure Docker for Artifact Registry (suppress output)
    ctx.run(
        f"gcloud auth configure-docker {region}-docker.pkg.dev --quiet",
        echo=False,
        pty=not WINDOWS,
    )

    # Tag locally with Artifact Registry format
    image_name = f"{region}-docker.pkg.dev/{project_id}/{artifact_registry}/skin-lesion-api:latest"

    # Build image (suppress most output, only show errors and final status)
    print("Building Docker image...")
    ctx.run(
        f"docker build -t {image_name} -f dockerfiles/api.dockerfile . --quiet",
        echo=False,
        pty=not WINDOWS,
    )
    print("✓ Image built successfully")

    # Push to Artifact Registry (suppress verbose output)
    print("Pushing image to Artifact Registry...")
    ctx.run(f"docker push {image_name} --quiet", echo=False, pty=not WINDOWS)
    print("✓ Image pushed successfully")

    # Deploy to Cloud Run
    print("Deploying to Cloud Run...")
    ctx.run(
        f"gcloud run deploy skin-lesion-api "
        f"--image {image_name} "
        f"--platform managed "
        f"--region {region} "
        f"--allow-unauthenticated "
        f"--port 8080 "
        f"--quiet",
        echo=True,  # Keep echo for deployment status
        pty=not WINDOWS,
    )
