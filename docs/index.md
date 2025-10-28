# Getting Started

<div align="center">
  <img alt="Linax Banner" src="https://raw.githubusercontent.com/camail-official/linax/refs/heads/main/assets/logo.png" style="padding-bottom: 2rem;" />
</div>

[linax](https://github.com/camail-official/linax) is a collection of state space models implemented in JAX. It is

- easy to use
- lightning-fast
- highly modular
- easily accessible.

## Table of contents
- [Just get me Going](#just-get-me-going)
- [Join the Community](#join-the-community)
- [Installation](#installation)
- [Full Library Installation](#full-library-installation)
- [Supported Models](#supported-models)
- [Contributing](#contributing)
- [Core Contributors](#core-contributors)
- [Citation](#citation)

## Just get me Going
🥱 If you don't care about the details, we provide [example notebooks](examples/) that are ready to use.


## Join the Community

To join our growing community of JAX and state space model enthusiasts, join our [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/VazrGCxeT7) server. Feel free to write us a message (either there or to our personal email, see the bottom of this page) if you have any questions, comments, or just want to say hi! 👋

🤫 Psssst! Rumor has it we are also developing an end-to-end JAX training pipeline. Stay tuned for JAX Lightning. So join the discord server to be the first to hear about our newest project(s)! 🚀

## Installation
[linax](https://github.com/camail-official/linax) is available as a PyPI package. To install it via uv, just run
```bash
uv add linax
```
or
```bash
uv add linax[cu12]
```

If pip is your package manager of choice, run
```bash
pip install linax
```
or
```bash
pip install linax[cu12]
```

## Full Library Installation
If you want to install the full library, especially if you want to **contribute** to the project, clone the [linax](https://github.com/camail-official/linax) repository and cd into it
```bash
git clone https://github.com/camail-official/linax.git
cd linax
```

If you want to install dependencies for CPU, run
```bash
uv sync
```
for GPU run
```bash
uv sync --extra cu12
```

To include development tooling (pre-commit, Ruff), install:
```bash
uv sync --extra dev
```
After installing the development dependencies (activate your environment if needed), enable the git hooks:
```bash
pre-commit install
```

## Supported Models

| Year | Venue | Model | Paper | Code | Our implementation |
|------|-------|-------|-------|------|---------------------|
| 2024 | arXiv | LinOSS | [Oscillatory State Space Models](https://arxiv.org/abs/2410.03943) | [tk-rusch/linoss](https://github.com/tk-rusch/linoss) | [linax](https://github.com/camail-official/linax/blob/main/src/linax/models/linoss.py) |
| 2023 | arXiv | LRU | [Resurrecting Recurrent Neural Networks for Long Sequences](https://arxiv.org/abs/2303.06349) | N/A | [linax](https://github.com/camail-official/linax/blob/main/src/linax/models/lru.py) |
| 2022 | arXiv | S5 | [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933) | [lindermanlab/S5](https://github.com/lindermanlab/S5) | [linax](https://github.com/camail-official/linax/blob/main/src/linax/models/s5.py) |
| 2022 | arXiv | S4D | [On the Parameterization and Initialization of Diagonal State Space Models](https://arxiv.org/abs/2206.11893) | N/A | [linax](https://github.com/camail-official/linax/blob/main/src/linax/sequence_mixers/s4d.py) |

## Contributing
If you want to contribute to the project, please check out [contributing](docs/contributing.md)

## Core Contributors

This repository has been created and is maintained by:

- [Benedict Armstrong](https://github.com/benedict-armstrong)
- [Philipp Nazari](https://phnazari.github.io)
- [Francesco Maria Ruscio](https://github.com/francescoshox)

This work has been carried out within the [Computational Applied Mathematics & AI Lab](https://camail.org),
led by [T. Konstantin Rusch](https://github.com/tk-rusch).

## Citation
If you find this repository useful, please consider citing it.

```bib
@software{linax2025,
  title  = {Linax: A Lightweight Collection of State Space Models in JAX},
  author = {Armstrong, Benedict and Nazari, Philipp and Ruscio, Francesco Maria},
  url    = {https://github.com/camail-official/linax},
  month  = oct,
  year   = {2025}
}
```
