# notebooks/

One notebook only: `phase2.ipynb`, a **driver**.

It imports `src/` code to launch runs and look at results — VRAM probing (T-205),
tokeniser output, the T-207 baseline gap, learning curves. It defines no pipeline
logic. The loader, training loop, config system and baseline runs live in `src/`
modules with tests and are run through `python -m scripts.t20x_*`; nothing on the
reproducibility path may exist only here (CLAUDE2.md "Code layout", `hard rule 5`).

Strip outputs before committing: `nbstripout notebooks/phase2.ipynb`, or pair with
jupytext and commit the `.py`.
