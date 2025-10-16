# pixelverse

A modern Python package using uv for dependency management.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) installed

### Installation

```bash
# Install dependencies (including dev dependencies)
uv sync --all-groups

# Or install just the package
uv sync
```

### Development Tools

This project uses the following tools for code quality:

- **black**: Code formatting (line-length: 100)
- **ruff**: Fast linting and import sorting
- **ty**: Type checking (experimental Astral type checker)

### Running Development Tools

```bash
# Format code with black
uv run black src/

# Lint with ruff
uv run ruff check src/

# Auto-fix ruff issues
uv run ruff check --fix src/

# Type check with ty
uv run ty check src/

# Format imports with ruff
uv run ruff check --select I --fix src/
```

### Running the CLI

```bash
# Run the pixelverse command
uv run pixelverse
```

### Project Structure

```
pixelverse/
├── src/
│   └── pixelverse/
│       ├── __init__.py
│       └── py.typed
├── pyproject.toml
├── README.md
└── .python-version
```

## Contributing

1. Install development dependencies: `uv sync --all-groups`
2. Make your changes
3. Format code: `uv run black src/`
4. Check linting: `uv run ruff check src/`
5. Check types: `uv run ty check src/`
