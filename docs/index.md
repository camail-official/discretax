<div align="center">
  <img alt="Linax Banner" src="https://raw.githubusercontent.com/camail-official/linax/refs/heads/feature/docs/assets/logo.png" style="padding-bottom: 2rem;" />
</div>

# Getting Started

[linax](https://github.com/camail-official/linax) is a collection of state space models implemented in JAX. It is

- is easy to use
- ⚡ lightning fast
- highly accessible.

Our aim as core developers is to provide researchers and scientists with a tool that just works out of the box, while simultaneously allowing for a high modularity and accessibility for anybody who wants to dig deeper.

## Just get me Going
🥱 If you don't care about the details, we provide [example notebooks](FILL IN!!) that are ready to use.

## Join the Community

To join our growing community of JAX and state space model enthusiasts, join our [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/VazrGCxeT7) server and our [mailing list](ADD ME). Feel free to just write us a message (either there or to our personal email, see the bottom of this page) if you have any questions, comments, or want to say hi! 👋

🤫 Psssst! Rumor has it we are also developing an end-to-end JAX training pipeline. Stay tuned for JAX Lightning. So join the discord server and our mailing list to be the first to hear about our newest project(s)! 🚀

## Installation
[linax](https://github.com/camail-official/linax) is available as a PyPi package. To install it via uv, just run
```bash
uv add linax
```
or
```bash
uv add linax[cuda]
```

If pip is your package manager of choice, run
```bash
pip install linax
```
or
```bash
pip install linax[cuda]
```

## Full Library Installation
If you want to install the full library, especially if you want to **contribute** to the project, clone the [linax](https://github.com/camail-official/linax) repository and cd into it
```bash
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

## Contributing
If you want to contribute to the project, please check out [contributing](/docs/contributing.md)

## Core Contributors

This repository has been created and is maintained by:

- [Benedict Armstrong](https://github.com/benedict-armstrong)
- [Philipp Nazari](https://phnazari.github.io)
- [Francesco Maria Ruscio](https://github.com/francescoshox)

This work has been carried out within the [Computational Applied Mathematics & AI Lab](https://camail.org),
led by [T. Konstantin Rusch](https://github.com/tk-rusch).
