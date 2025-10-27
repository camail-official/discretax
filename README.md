<div align="center">
  <img alt="Linax Banner" src="https://raw.githubusercontent.com/camail-official/linax/refs/heads/feature/docs/assets/logo.png" style="padding-bottom: 2rem;" />
</div>


# linax - State Space Models in Jax

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/VazrGCxeT7)

</div>


`linax` is a collection of State space models in `JAX`.

## Package Installation
**Not yet available**
```
pip install linax
```

## Full Library Installation
```bash
uv sync
```

To include development tooling (pre-commit, Ruff), install with the `--dev` flag:

```bash
uv sync --extra dev
```

After installing the development dependencies (activate your environment if needed), enable the git hooks:

```bash
uv run pre-commit install
```

### On device with CUDA

```bash
uv sync --extra cu12
```

To combine the CUDA extras with development dependencies:

```bash
uv sync --extra cu12 --extra dev
```

## Core Contributors

This repository has been created and is maintained by:

- [Benedict Armstrong](https://github.com/benedict-armstrong)
- [Philipp Nazari](https://phnazari.github.io)
- [Francesco Maria Ruscio](https://github.com/francescoshox)

This work has been carried out within the [Computational Applied Mathematics & AI Lab](https://camail.org),
led by [T. Konstantin Rusch](https://github.com/tk-rusch).
