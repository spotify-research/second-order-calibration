# second-order-calibration

[![Lifecycle: alpha](https://img.shields.io/badge/lifecycle-alpha-a0c3d2.svg)](#development-status)

Research code for estimating and correcting second-order calibration error,
accompanying the NeurIPS 2026 paper "The Minimax Rate of Perturbed Second-Order
Calibration".

> **The Minimax Rate of Perturbed Second-Order Calibration**
> Kamil Ciosek, Banafsheh Rafiee, Sina Ghiassian, Nicolò Felicioni
> *Advances in Neural Information Processing Systems (NeurIPS), 2026*
> [arXiv:2605.07808](https://arxiv.org/abs/2605.07808)

The paper was accepted at NeurIPS 2026.

## Getting Started

- **Repository:** <https://github.com/spotify-research/second-order-calibration>
- **Paper:** <https://arxiv.org/abs/2605.07808>
- **Install:** see [Installation](#installation) below.
- **Questions and bug reports:** open a
  [GitHub issue](https://github.com/spotify-research/second-order-calibration/issues/new).

## Project Description

### Why this project exists

Many models report not just a probability but also how uncertain they are about
it. A second-order predictor outputs a score pair `S = (m, s)`: a probability `m`
and a variance `s` that describes epistemic uncertainty about that probability.
Second-order calibration asks whether `m` and `m² + s` match the conditional
first and second moments of the label. These moments can be identified from
two independent label snapshots `Y⁽¹⁾, Y⁽²⁾`.

The paper establishes the minimax rate at which second-order calibration error
can be estimated after the scores are perturbed. This repository contains the
code behind its results.

### What is in this repository

- **An estimator of second-order calibration error (CE₂).** It perturbs the
  scores with a smooth sech-shaped kernel of bandwidth `h`, then estimates the
  conditional moments `η₁(S) = E[Y⁽¹⁾ | S]` and `η₂(S) = E[Y⁽¹⁾Y⁽²⁾ | S]` with
  ridge-regularised polynomial (tensor-Chebyshev) regression. Bucketing and
  Nadaraya–Watson baselines are included for comparison.
- **Perturbed second-order Platt scaling**, a recalibration method for score
  pairs `(m, s)`.
- **The experiments in the paper.** These include synthetic experiments that
  confirm the convergence rate of the estimator, and downstream decision
  experiments where second-order recalibration matters.

### Intended use

The code is meant for researchers who want to:

- reproduce the experiments and figures in the paper;
- understand how the estimator and the recalibration method work in practice;
- use the implementation as a starting point for their own research on
  second-order calibration.

### Caveats

This is a **research prototype** accompanying the NeurIPS 2026 paper "The
Minimax Rate of Perturbed Second-Order Calibration"
([arXiv:2605.07808](https://arxiv.org/abs/2605.07808)). It is **not a
production library**:

- The code is organised as experiment scripts, not as an installable package
  with a stable API.
- The scripts are tuned to the paper's settings and have not been hardened for
  other data, scales or deployment.
- Results may vary slightly across hardware, PyTorch versions and CUDA versions.

### Repository structure

```
.
├── exp1/                    # Experiment 1: convergence rate of CE₂ estimators (Fig. 2)
│   ├── dataset.py           #   synthetic DGP, deep ensemble, perturbation, ground-truth CE₂
│   ├── bucketing.py         #   bucketing (2D histogram) estimator
│   ├── nadaraya_watson.py   #   Nadaraya–Watson estimator
│   ├── polynomial.py        #   polynomial / Chebyshev ridge estimator
│   ├── plot.py              #   error-vs-n log-log plots
│   ├── common.py            #   shared utilities
│   ├── test_units.py        #   unit tests
│   ├── run_all.sh           #   runs the full experiment
│   └── README.md            #   experiment details and configuration
├── exp2/                    # Experiment 2: second-order recalibration of an ensemble (Fig. 3)
│   ├── dataset.py, polynomial.py, common.py
│   ├── experiment2.py       #   recalibration, diagnostics, plots
│   ├── replot.py            #   regenerate plots from saved results
│   ├── run_experiment2.sh
│   └── README.md
├── exp3/                    # Experiment 3: synthetic specialist referral (Fig. 4)
│   ├── experiment_med.py
│   ├── run_exp_med.sh
│   └── README.md
├── exp4/                    # Experiment 4: crowd disagreement audit, Weather Sentiment (Fig. 5)
│   ├── experiment_crowd_weather_disagreement.py
│   ├── replot.py
│   ├── run_exp_crowd_weather_disagreement.sh
│   └── README.md
├── requirements.txt
├── .github/CODEOWNERS
├── catalog-info.yaml
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── README.md
```

Each experiment folder has a `README.md` describing the setup, outputs,
defaults and environment-variable overrides.

## Project Direction

### Development status

**Alpha.** The code is complete for reproducing the paper, and that is the
scope of the repository. We do not plan further feature development. We will
fix bugs that affect reproducibility, and we welcome issues and small fixes
(see [Contributing](#contributing)). The BibTeX entry below will be updated
once the NeurIPS proceedings version is published.

## Develop

### Installation

The code was tested with Python 3.10. Python 3.10 or later is recommended.

```bash
# install dependencies
git clone https://github.com/spotify-research/second-order-calibration.git
cd second-order-calibration
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All figures are rendered with matplotlib's LaTeX backend (`text.usetex: True`).
You therefore need a working LaTeX installation with the `lmodern` font, for
example `texlive-latex-extra` and `cm-super` on Debian/Ubuntu.

A CUDA GPU is optional. The `dataset.py` scripts use one for ensemble training
if one is available and fall back to the CPU otherwise.

### Datasets

#### Synthetic datasets (Experiments 1 and 2)

Experiments 1 and 2 use synthetic data. The cached `data/dataset.npz` files are
**not** included in the repository because they are large (69 MB and 115 MB).
Regenerate them deterministically with fixed seeds and the default settings:

```bash
# Experiment 1  ->  exp1/data/dataset.npz
cd exp1 && python dataset.py && cd ..

# Experiment 2  ->  exp2/data/dataset.npz
cd exp2 && python dataset.py && cd ..
```

Each script must be run from inside its own experiment folder, because paths are
relative (`./data`, `./results`). `run_all.sh` and `run_experiment2.sh` also
generate the dataset as their first step, so you do not need to run the
commands above separately.

#### Specialist referral (Experiment 3)

This experiment is fully synthetic. Its scores are simulated inside
`experiment_med.py`, so no dataset is needed.

#### Weather Sentiment (Experiment 4)

The crowdsourcing experiment uses the public **Weather Sentiment** Amazon
Mechanical Turk dataset (CrowdFlower data, released by Venanzi et al.).
It is not included in this repository. On the first run, the script downloads
it automatically to `exp4/data/WeatherSentiment_amt.csv`
from <https://eprints.soton.ac.uk/376543/1/WeatherSentiment_amt.csv>.

### Reproducing the experiments

Run every command from the repository root. Outputs (PDF/PNG figures and `.npz`
metrics) are written to each experiment's `results/` folder.

| Paper | Experiment | Command | Main outputs |
|---|---|---|---|
| Exp. 1, Fig. 2 | Convergence rate of CE₂ estimators (bucketing, Nadaraya–Watson, polynomial) | `bash exp1/run_all.sh` | `exp1/results/experiment1_h_1_{4,16,64}.pdf` |
| Exp. 2, Fig. 3 | Perturbed second-order Platt scaling of a deep ensemble | `bash exp2/run_experiment2.sh` | `exp2/results/experiment2_h_*_{before,after,binned,ce2}.pdf`, `experiment2_rankdiag_h_*.pdf` |
| Exp. 3, Fig. 4 | Synthetic specialist referral | `bash exp3/run_exp_med.sh` | `exp3/results/exp_med_strong_{decision_curve,decision_curve_legend,pgi_distribution,score_space}.pdf` |
| Exp. 4, Fig. 5 | Crowd disagreement audit (Weather Sentiment) | `bash exp4/run_exp_crowd_weather_disagreement.sh` | `exp4/results/exp_crowd_weather_disagreement_{curves,contrasts}.pdf` |

The defaults in each script are the settings used for the figures in the
paper. Experiments 1 and 2 are run for the perturbation bandwidths
`h ∈ {1/4, 1/16, 1/64}`.

After an experiment has run, you can redraw its figures from the saved results
without recomputing them:

```bash
cd exp2 && python replot.py && cd ..
cd exp4 && python replot.py && cd ..
cd exp1 && python plot.py && cd ..
```

### Running the tests

```bash
cd exp1 && python -m pytest test_units.py
```

## Support

Create a [new issue](https://github.com/spotify-research/second-order-calibration/issues/new).
As this is a research prototype, responses may take some time.

## Contributing

We feel that a welcoming community is important and we ask that you follow Spotify's
[Open Source Code of Conduct](CODE_OF_CONDUCT.md) in all interactions with the
community. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to report issues and
submit fixes, and [SECURITY.md](SECURITY.md) for how to report security
vulnerabilities.

## Authors

* Kamil Ciosek
* Banafsheh Rafiee
* Sina Ghiassian
* Nicolò Felicioni

A full list of [contributors](https://github.com/spotify-research/second-order-calibration/graphs/contributors?type=a) can be found on GitHub.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ciosek2026minimax,
  title     = {The Minimax Rate of Perturbed Second-Order Calibration},
  author    = {Ciosek, Kamil and Rafiee, Banafsheh and Ghiassian, Sina and Felicioni, Nicol{\`o}},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2605.07808},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2605.07808}
}
```

## License

Copyright 2026 Spotify AB.

Licensed under the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0.
See [LICENSE](LICENSE).

## Security Issues?

Please report sensitive security issues via Spotify's bug-bounty program (https://hackerone.com/spotify) rather than GitHub. See [SECURITY.md](SECURITY.md).
