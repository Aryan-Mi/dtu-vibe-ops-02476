# CI/CD Pipeline Documentation

Complete guide to the Continuous Integration and Continuous Deployment pipeline for the MLOps Skin Lesion Classification project.

## Pipeline Overview

Our CI/CD pipeline ensures code quality, automated testing, and seamless deployment:

- **Continuous Integration**: Automated testing, linting, and quality checks
- **Continuous Deployment**: Automated model training, deployment, and monitoring
- **Multi-platform Support**: Tests on Linux, Windows, and macOS
- **Security**: Vulnerability scanning and dependency checks

## GitHub Actions Workflows

### Current Workflows

```
.github/workflows/
├── tests.yaml              # Unit and integration tests
├── linting.yaml            # Code quality checks
├── docker-build.yaml       # Container building and testing
├── api_deployment.yaml     # API deployment to cloud
├── train-manual.yaml       # Manual model training triggers
└── pre-commit-update.yaml  # Keep pre-commit hooks updated
```

## Core CI Workflow

### Test Pipeline (`tests.yaml`)

```yaml
name: Tests and Quality Checks

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    name: Test Python ${{ matrix.python-version }} on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.12"]
        
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
        with:
          version: "latest"
      
      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: uv sync --all-extras
      
      - name: Run pre-commit hooks
        run: uv run pre-commit run --all-files
        if: matrix.os == 'ubuntu-latest'
      
      - name: Run unit tests
        run: |
          uv run pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --junitxml=pytest.xml \
            -v \
            -m "not slow and not gpu"
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        if: matrix.os == 'ubuntu-latest'
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
          verbose: true
      
      - name: Upload test results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: test-results-${{ matrix.os }}
          path: |
            pytest.xml
            htmlcov/
            coverage.xml

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run integration tests
        run: |
          uv run pytest tests/ \
            -v \
            -m integration \
            --tb=short
      
      - name: Test API endpoints
        run: |
          # Start API server in background
          uv run uvicorn src.mlops_project.api:app --host 0.0.0.0 --port 8000 &
          sleep 10
          
          # Test endpoints
          uv run pytest tests/test_api.py -v
          
          # Cleanup
          pkill -f uvicorn

  performance-tests:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync
      
      - name: Run performance tests
        run: |
          uv run pytest tests/test_performance.py \
            -v \
            --benchmark-json=benchmark.json
      
      - name: Store benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

### Code Quality Pipeline (`linting.yaml`)

```yaml
name: Code Quality

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  quality:
    name: Code Quality Checks
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync
      
      - name: Check code formatting
        run: |
          uv run ruff format --check src/ tests/
          
      - name: Run linting
        run: |
          uv run ruff check src/ tests/ --output-format=github
      
      - name: Type checking with mypy
        run: |
          uv run mypy src/ --strict
        continue-on-error: true
      
      - name: Security scan with bandit
        run: |
          uv run bandit -r src/ -f json -o bandit-report.json
        continue-on-error: true
      
      - name: Upload security scan results
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: security-scan
          path: bandit-report.json
      
      - name: Dependency vulnerability check
        run: |
          uv run safety check --json --output safety-report.json
        continue-on-error: true
      
      - name: Upload vulnerability scan
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: vulnerability-scan
          path: safety-report.json

  documentation:
    name: Documentation Build
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync
      
      - name: Build documentation
        run: |
          cd docs
          uv run mkdocs build --strict
      
      - name: Upload documentation artifacts
        uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/site/
```

## Docker CI/CD Pipeline

### Container Build and Test (`docker-build.yaml`)

```yaml
name: Docker Build and Test

on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    name: Build Docker Images
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    strategy:
      matrix:
        target: [training, api, docs]
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.target }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: ${{ matrix.target }}
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Test Docker image
        run: |
          docker run --rm \
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.target }}:${{ github.sha }} \
            --help

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: build
    if: github.event_name != 'pull_request'
    
    strategy:
      matrix:
        target: [training, api]
    
    steps:
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}-${{ matrix.target }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

## Deployment Pipelines

### API Deployment (`api_deployment.yaml`)

```yaml
name: Deploy API to Cloud Run

on:
  push:
    branches: [main]
    tags: ['v*']
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production

jobs:
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'staging')
    environment: staging
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
      
      - name: Configure Docker for GCR
        run: gcloud auth configure-docker
      
      - name: Build and push Docker image
        run: |
          docker build -t gcr.io/${{ secrets.GCP_PROJECT_ID }}/skin-lesion-api:${{ github.sha }} .
          docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/skin-lesion-api:${{ github.sha }}
      
      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy skin-lesion-api-staging \
            --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/skin-lesion-api:${{ github.sha }} \
            --platform managed \
            --region europe-west1 \
            --allow-unauthenticated \
            --memory 2Gi \
            --cpu 2 \
            --concurrency 100 \
            --max-instances 10 \
            --set-env-vars ENVIRONMENT=staging
      
      - name: Run deployment tests
        run: |
          # Wait for deployment
          sleep 30
          
          # Get service URL
          SERVICE_URL=$(gcloud run services describe skin-lesion-api-staging \
            --platform managed \
            --region europe-west1 \
            --format 'value(status.url)')
          
          # Test health endpoint
          curl -f $SERVICE_URL/health
          
          # Test prediction endpoint with sample image
          # (would include actual integration test here)

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v') || (github.event_name == 'workflow_dispatch' && github.event.inputs.environment == 'production')
    environment: production
    needs: deploy-staging
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_PROD_SA_KEY }}
      
      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1
      
      - name: Deploy to Production Cloud Run
        run: |
          # Use the same image from staging
          gcloud run deploy skin-lesion-api \
            --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/skin-lesion-api:${{ github.sha }} \
            --platform managed \
            --region europe-west1 \
            --allow-unauthenticated \
            --memory 4Gi \
            --cpu 4 \
            --concurrency 50 \
            --max-instances 50 \
            --set-env-vars ENVIRONMENT=production
      
      - name: Run production health checks
        run: |
          SERVICE_URL=$(gcloud run services describe skin-lesion-api \
            --platform managed \
            --region europe-west1 \
            --format 'value(status.url)')
          
          # Comprehensive health checks
          curl -f $SERVICE_URL/health
          curl -f $SERVICE_URL/health/detailed
          curl -f $SERVICE_URL/model/info
          
      - name: Update monitoring dashboards
        run: |
          # Update monitoring alerts and dashboards
          # (implementation specific to monitoring setup)
          echo "Updating production monitoring..."
```

### Model Training Pipeline (`train-manual.yaml`)

```yaml
name: Model Training Pipeline

on:
  workflow_dispatch:
    inputs:
      model_type:
        description: 'Model architecture to train'
        required: true
        default: 'efficientnet'
        type: choice
        options:
        - baseline_cnn
        - resnet
        - efficientnet
      variant:
        description: 'Model variant (for EfficientNet)'
        required: false
        default: 'b3'
      max_epochs:
        description: 'Maximum training epochs'
        required: true
        default: '25'
      experiment_name:
        description: 'Experiment name for tracking'
        required: true
        default: 'automated-training'

jobs:
  train-model:
    name: Train ${{ github.event.inputs.model_type }} Model
    runs-on: ubuntu-latest-gpu  # Assuming GPU-enabled runner
    timeout-minutes: 480  # 8 hours max
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      
      - name: Download training data
        run: |
          # Download data from GCS
          gsutil -m cp -r gs://mlops-skin-lesion-data/ham10000 data/
      
      - name: Configure Weights & Biases
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          uv run wandb login $WANDB_API_KEY
      
      - name: Train model
        run: |
          uv run src/mlops_project/train.py \
            model=${{ github.event.inputs.model_type }} \
            model.variant=${{ github.event.inputs.variant || 'b0' }} \
            training.max_epochs=${{ github.event.inputs.max_epochs }} \
            wandb.enabled=true \
            wandb.project="automated-training" \
            wandb.name="${{ github.event.inputs.experiment_name }}" \
            wandb.tags=["automated","github-actions"] \
            training.accelerator=gpu \
            training.devices=1
      
      - name: Upload model artifacts
        run: |
          # Upload trained models to GCS
          gsutil -m cp -r models/ gs://mlops-skin-lesion-models/automated-training/
      
      - name: Model validation
        run: |
          # Run model validation tests
          uv run pytest tests/test_model_validation.py -v
      
      - name: Notify completion
        if: always()
        run: |
          # Send notification (Slack, email, etc.)
          echo "Training completed for ${{ github.event.inputs.model_type }}"
```

## Quality Gates

### Branch Protection Rules

```yaml
# GitHub branch protection configuration
branch_protection:
  main:
    required_status_checks:
      strict: true
      contexts:
        - "Tests and Quality Checks / Test Python 3.12 on ubuntu-latest"
        - "Tests and Quality Checks / Test Python 3.12 on windows-latest"
        - "Tests and Quality Checks / Test Python 3.12 on macos-latest"
        - "Code Quality / Code Quality Checks"
        - "Code Quality / Documentation Build"
    enforce_admins: true
    required_pull_request_reviews:
      required_approving_review_count: 1
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    restrictions:
      users: []
      teams: ["maintainers"]
```

### Quality Thresholds

```python
# Quality gates configuration
QUALITY_GATES = {
    "test_coverage": 90,          # Minimum test coverage percentage
    "code_quality": "A",          # Minimum code quality grade
    "security_score": 8,          # Minimum security score (1-10)
    "performance_regression": 5,   # Maximum performance regression percentage
    "documentation_coverage": 80,  # Minimum documentation coverage
}
```

## Monitoring and Observability

### Pipeline Monitoring

```yaml
# .github/workflows/monitor.yaml
name: Pipeline Monitoring

on:
  schedule:
    - cron: '0 8 * * *'  # Daily at 8 AM UTC

jobs:
  health-check:
    name: System Health Check
    runs-on: ubuntu-latest
    
    steps:
      - name: Check API endpoints
        run: |
          # Check staging environment
          curl -f https://staging-api.example.com/health
          
          # Check production environment
          curl -f https://production-api.example.com/health
      
      - name: Check model performance
        run: |
          # Run automated model validation
          python scripts/validate_deployed_model.py
      
      - name: Check data pipeline
        run: |
          # Validate data freshness and quality
          python scripts/check_data_pipeline.py
      
      - name: Generate monitoring report
        run: |
          python scripts/generate_health_report.py > health_report.md
      
      - name: Upload monitoring results
        uses: actions/upload-artifact@v3
        with:
          name: health-report
          path: health_report.md
```

### Alerting Configuration

```yaml
# Monitoring alerts configuration
alerts:
  api_health:
    condition: "health_check_failed"
    channels: ["slack", "email"]
    severity: "critical"
    
  model_performance:
    condition: "accuracy < 0.85"
    channels: ["slack"]
    severity: "warning"
    
  deployment_failure:
    condition: "deployment_failed"
    channels: ["slack", "pagerduty"]
    severity: "critical"
    
  test_failure:
    condition: "test_success_rate < 0.95"
    channels: ["slack"]
    severity: "warning"
```

## Security and Compliance

### Secret Management

```yaml
# Required repository secrets
secrets:
  # Google Cloud
  GCP_SA_KEY: "Service account key for staging"
  GCP_PROD_SA_KEY: "Service account key for production"
  GCP_PROJECT_ID: "Google Cloud project ID"
  
  # Weights & Biases
  WANDB_API_KEY: "W&B API key for experiment tracking"
  
  # Container Registry
  DOCKER_USERNAME: "Docker Hub username"
  DOCKER_PASSWORD: "Docker Hub password"
  
  # Monitoring
  SLACK_WEBHOOK: "Slack webhook for notifications"
  PAGERDUTY_KEY: "PagerDuty integration key"
```

### Security Scanning

```yaml
# Security scanning configuration
security:
  dependency_scanning:
    tools: ["safety", "pip-audit"]
    schedule: "daily"
    
  code_scanning:
    tools: ["bandit", "semgrep"]
    on_events: ["push", "pull_request"]
    
  container_scanning:
    tools: ["trivy", "clair"]
    registries: ["ghcr.io", "gcr.io"]
    
  infrastructure_scanning:
    tools: ["checkov", "tfsec"]
    paths: ["terraform/", "kubernetes/"]
```

## Performance and Optimization

### Pipeline Optimization Strategies

1. **Parallel Execution**: Run independent jobs in parallel
2. **Caching**: Cache dependencies and build artifacts
3. **Incremental Builds**: Only rebuild changed components
4. **Matrix Builds**: Test across multiple configurations efficiently
5. **Artifact Reuse**: Share build artifacts between jobs

### Build Cache Configuration

```yaml
# Optimized caching strategy
cache_strategy:
  dependencies:
    key: "${{ runner.os }}-uv-${{ hashFiles('uv.lock') }}"
    restore_keys: |
      ${{ runner.os }}-uv-
      
  docker_layers:
    type: "gha"
    mode: "max"
    
  test_data:
    key: "test-data-${{ hashFiles('tests/fixtures/**') }}"
    path: "tests/fixtures/"
```

---

*This completes our comprehensive CI/CD pipeline documentation. The pipeline ensures code quality, automated testing, and reliable deployment of the MLOps skin lesion classification system.*