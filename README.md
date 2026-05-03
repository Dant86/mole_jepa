# MoLeJEPA

An implementation of the MoLeJEPA model along with train/evaluation scripts and exploratory notebooks.

## Development

### Setup

Install [uv](https://docs.astral.sh/uv/), then install all dependencies including dev tools:

```bash
uv sync --group dev
uv run pre-commit install
```

### Code Style

This project enforces consistent style via **ruff** (lint + format) and **pyright** (static type checking). Pre-commit hooks run both tools automatically on every commit. The same checks run in CI on every push and pull request.

#### Imports

Imports follow Google style — stdlib first, then third-party, then local — each group separated by a blank line. `ruff` (isort rules) enforces the ordering automatically.

Import **modules**, not names from modules. The only exception is the `typing` module, whose members may be imported directly.

```python
# stdlib
import os
from typing import Optional

# third-party
import torch
import datasets

# local
from mole_jepa import model
```

#### Docstrings

All public functions, classes, and methods require Google-style docstrings. Module and package-level docstrings are optional.

```python
def encode(smiles: str, max_len: int = 512) -> torch.Tensor:
    """Encode a SMILES string into a token tensor.

    Args:
        smiles: The input SMILES string.
        max_len: Maximum sequence length; longer sequences are truncated.

    Returns:
        A 1-D integer tensor of token IDs.

    Raises:
        ValueError: If `smiles` is empty.
    """
```

#### Running checks manually

```bash
uv run ruff check .          # lint
uv run ruff format --check . # format check (drop --check to auto-fix)
uv run pyright               # type check
```
