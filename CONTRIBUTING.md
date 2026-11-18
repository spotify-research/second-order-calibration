# How to Contribute

Thanks for your interest in second-order-calibration!

This repository is a **research prototype**: the code accompanying the NeurIPS
2026 paper "The Minimax Rate of Perturbed Second-Order Calibration"
([arXiv:2605.07808](https://arxiv.org/abs/2605.07808)). Its main purpose is to
let others reproduce and build on the paper's results. We do not plan active
feature development, and we can't promise that every request will be addressed.

That said, contributions are welcome, especially:

- **Bug reports**, for example a script that fails, or results that differ from
  the paper.
- **Small fixes**: bug fixes, compatibility fixes for newer dependency versions,
  and documentation improvements.

For larger changes (new methods, refactors, new experiments), please open an
issue first so we can discuss whether it fits the scope of the repository.
Usually such work is better done in a fork.

## Getting Started

The [README](README.md) explains the repository structure, installation and how
to reproduce each experiment. Each experiment folder (`exp1/` to `exp4/`) has
its own `README.md` with configuration details.

## Building dependencies

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The plotting scripts render figures with LaTeX (`text.usetex: True`), so a
working LaTeX installation with the `lmodern` font is needed to produce figures.

## Building the Project

There is nothing to build. The experiments are plain Python scripts; see
[Reproducing the experiments](README.md#reproducing-the-experiments) in the
README.

## Workflow

We follow the [GitHub Flow Workflow](https://guides.github.com/introduction/flow/):

1.  Fork the project
1.  Check out the `main` branch
1.  Create a feature branch
1.  Write code and tests for your change
1.  From your branch, make a pull request against `spotify-research/second-order-calibration` `main`
1.  Work with repo maintainers to get your change reviewed
1.  Wait for your change to be merged into `main`
1.  Delete your feature branch

## Testing

The unit tests are written with pytest:

```bash
cd exp1 && python -m pytest test_units.py
```

Please make sure the tests pass before opening a pull request, and add tests
for bug fixes where practical. If your change affects an experiment's results,
say so in the pull request and describe how you checked it.

## Style

Follow the style of the surrounding code (PEP 8, with readable names). Keep
changes focused, and avoid unrelated reformatting.

## Issues

When creating an issue please try to adhere to the following format:

    exp-name: One line summary of the issue (less than 72 characters)

    ### Expected behavior

    As concisely as possible, describe the expected behavior.

    ### Actual behavior

    As concisely as possible, describe the observed behavior.

    ### Steps to reproduce the behavior

    List all relevant steps to reproduce the observed behavior, including the
    command you ran, your Python version and the versions of key dependencies.

## Pull Requests

Comments should be formatted to a width no greater than 80 columns.

Files should be exempt of trailing spaces.

Please write commit messages along these guidelines, with a line width no
greater than 80 columns (you can use `fmt -n -p -w 80` to accomplish this):

    exp-name: One line description of your change (less than 72 characters)

    Problem

    Explain the context and why you're making that change.  What is the problem
    you're trying to solve? In some cases there is not a problem and this can be
    thought of being the motivation for your change.

    Solution

    Describe the modifications you've done.

    Result

    What will change as a result of your pull request? Note that sometimes this
    section is unnecessary because it is self-explanatory based on the solution.

Some important notes regarding the summary line:

* Describe what was done; not the result
* Use the active voice
* Use the present tense
* Capitalize properly
* Do not end in a period — this is a title/subject
* Prefix the subject with its scope (for example `exp2:` or `docs:`)

## Code Review

Pull requests are reviewed by the maintainers on GitHub. Because this is a
research prototype maintained alongside other work, reviews may take some time.

## Documentation

We also welcome improvements to the project documentation. Please file an
[issue](https://github.com/spotify-research/second-order-calibration/issues/new)
or open a pull request.

# License

By contributing your code, you agree to license your contribution under the
terms of the [LICENSE](LICENSE).

# Code of Conduct

Read our [Code of Conduct](CODE_OF_CONDUCT.md) for the project.
