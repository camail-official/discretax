# Contributing to Linax

Thank you for your interest in contributing to Linax! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

---

## Getting Started

Before contributing, please:

1. **Check existing issues** to see if someone is already working on it
2. **Open an issue** to discuss major changes before implementing them
3. **Read our [Code of Conduct](CODE_OF_CONDUCT.md)** and follow community standards

## Development Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation Steps

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/linax.git
cd linax
```

2. **Install dependencies (including dev tools):**

```bash
uv sync --extra dev
```

3. **Install pre-commit hooks:**

```bash
uv run pre-commit install
```

This will automatically run linters and formatters before each commit.

### Optional: CUDA Support

If you have a CUDA-enabled GPU:

```bash
uv sync --extra cu12 --extra dev
```

---

## Code Style

We use the following tools to maintain code quality:

- **Ruff** - Linting and formatting
- **Pre-commit hooks** - Automatic checks before commits

### Running Code Quality Checks

```bash
# Run all pre-commit hooks manually
uv run pre-commit run --all-files

# Run ruff linter
uv run ruff check src/ tests/

# Run ruff formatter
uv run ruff format src/ tests/
```

### Style Guidelines

- Follow **PEP 8** conventions
- Use **Google-style docstrings**
- Maximum line length: **99 characters**
- Type hints are **required** for all public functions
- Use `jaxtyping` for array shape annotations

**Example:**

```python
from jaxtyping import Array, Float

def my_function(x: Float[Array, "batch features"]) -> Float[Array, "batch"]:
    """Brief description of the function.

    Args:
        x: Input array with shape (batch, features).

    Returns:
        Processed array with shape (batch,).
    """
    return x.sum(axis=-1)
```

---

## Testing

We use **pytest** for testing.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_smoke.py

# Run with coverage
uv run pytest --cov=linax
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Include docstrings explaining what the test validates

---

## Submitting Changes

### Pull Request Process

1. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure all checks pass:**

```bash
uv run pre-commit run --all-files
uv run pytest
```

4. **Push to your fork:**

```bash
git push origin feature/your-feature-name
```

5. **Open a Pull Request:**
   - Use a clear, descriptive title
   - Fill out the PR template completely
   - Link any related issues

### Commit Message Guidelines

Write clear, descriptive commit messages that explain what changed:

**Good:**
```
Add LRU sequence mixer implementation
Fix gradient computation in LinOSS
Update README installation instructions
Add tests for S5 model
```

**Not so good:**
```
fix
update
wip
changes
```

---

## Community

- **Discord**: [Join our Discord server](https://discord.gg/VazrGCxeT7)
- **Issues**: [GitHub Issues](https://github.com/camail-official/linax/issues)
- **Website**: [camail.org/linax](https://camail.org/linax/)

---

## Questions?

If you have questions about contributing, feel free to:

- Open a [GitHub Discussion](https://github.com/camail-official/linax/discussions)
- Ask in our [Discord server](https://discord.gg/VazrGCxeT7)
- Email the maintainers (see `pyproject.toml` for contact info)

---

Thank you for contributing to Linax! 🚀
