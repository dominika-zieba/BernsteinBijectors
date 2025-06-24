# BernsteinBijectors

This repository contains an implementation of Bernstein Polynomial normalising flow bijectors in [JAX](https://docs.jax.dev/en/latest/quickstart.html). The bijectors are implemented as a [distrax](https://github.com/google-deepmind/distrax) [Bijector class](https://github.com/google-deepmind/distrax/blob/master/distrax/_src/bijectors/bijector.py). Implementation is based on [Ramasinghe et al. (2022)](https://arxiv.org/abs/2102.03509).

Contents of the repository:

- BernsteinBijector.py contains the implementation of the Bernstein Bijectors as a distrax class.
- vi_routines.py contains a JAX implementation of a coupling normalising flow with Bernstein Bijectors.
- VI_BP.ipynb provides an example of training the Bernstein polynomial flow to approximate a Gaussian mixture target using variational inference (reverse KL optimisation).

## Installation

The package is not in PyPI yet, but to install directly from github, run
```
pip install git+https://github.com/dominika-zieba/BernsteinBijectors.git
```

If you want to edit, then clone the repository and `pip install .` from the top level directory.

## Dependencies

The JAX ecosystem is frequently changing and has been known break upon upgrades. Sometimes incompatibilities can arise between GPU and CPU versions.
This package has been tested on a fresh environment with `pip install jax[cuda12]`, pulling jax 0.6.2. There are known issues with v0.6.1 and v0.6.0.

