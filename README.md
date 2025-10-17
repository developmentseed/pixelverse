# pixelverse

Cloud Native Tooling to generate and store pixelwise geospatial embeddings.

Please note, very much in prototyping form and under development

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) and pre-commit for development.

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) installed

### Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-extras
uv run pre-commit install

# optionally run pre-commit hooks manually
uv run pre-commit run --all-files
```
