# MoLeJEPA

[![codecov](https://codecov.io/github/Dant86/mole_jepa/graph/badge.svg?token=EFA7KKVN22)](https://codecov.io/github/Dant86/mole_jepa)

<div align="center">
  <img src="assets/mole.svg" width="320" alt="pixel art mole">
</div>

A multimodal JEPA that predicts text embeddings from image embeddings,
regularized with SIGReg to prevent representational collapse.

## Architecture

Image encoder `f` (ViT) and text encoder `h` (LM) both project to a shared
embedding space `ℝᵈ`. A lightweight MLP predictor `g` is trained to predict
`h(y)` from `f(x)` for paired image-text inputs `(x, y)`. SIGReg is applied
to both encoder outputs to enforce isotropic Gaussian structure:

```
L = MSE(g(f(x)), h(y))  +  λ · (SIGReg(f(x)) + SIGReg(h(y)))
```

## Project Structure

```
src/mole_jepa/
├── models/
│   ├── encoders.py       # ImageEncoder (ViT), TextEncoder (LM)
│   ├── predictor.py      # MLP predictor g
│   └── mole_jepa.py      # Top-level model and MoLeJEPAOutput
├── regularizers/
│   └── sig_reg.py        # SIGReg
└── test_statistics/
    └── epps_pulley.py    # EppsPulley base + Gaussian / Laplace variants
```

## Development

### Setup

Install [uv](https://docs.astral.sh/uv/), then install all dependencies
including dev tools:

```bash
uv sync --group dev
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest
```

### Code Style

This project enforces consistent style via **ruff** (lint + format) and
**pyright** (static type checking). Pre-commit hooks run both tools
automatically on every commit. The same checks run in CI on every push and
pull request.

#### Imports

Imports follow Google style — stdlib first, then third-party, then local —
each group separated by a blank line. `ruff` (isort rules) enforces the
ordering automatically.

Import **modules**, not names from modules. The only exception is the
`typing` module, whose members may be imported directly.

```python
# stdlib
import dataclasses
from typing import Literal

# third-party
import torch
import transformers

# local
from mole_jepa import models
from mole_jepa import test_statistics
```

#### Docstrings

All public functions, classes, and methods require Google-style docstrings.
Module and package-level docstrings are optional.

```python
def encode(text: str, max_len: int = 512) -> torch.Tensor:
    """Encode a text string into a token tensor.

    Args:
        text: The input string.
        max_len: Maximum sequence length; longer sequences are truncated.

    Returns:
        A 1-D integer tensor of token IDs.

    Raises:
        ValueError: If `text` is empty.
    """
```

#### Running checks manually

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # format check (drop --check to auto-fix)
uv run pyright                # type check
```
