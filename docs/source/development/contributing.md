# Contributing

## Development Setup

```bash
git clone https://github.com/Aryan-Mi/dtu-vibe-ops-02476.git
cd dtu-vibe-ops-02476
uv sync
uv run pre-commit install
```

## Code Style

We use Ruff for linting and formatting:

```bash
# Check code
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Submit PR with clear description
5. Address review feedback

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `refactor:` Code refactoring
