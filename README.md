# pyTDGL

![pyTDGL Logo](docs/images/logo-transparent-small.png)

Time-dependent Ginzburg-Landau in Python

![PyPI](https://img.shields.io/pypi/v/tdgl)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/loganbvh/py-tdgl/lint-and-test.yml?branch=main)
[![Documentation Status](https://readthedocs.org/projects/py-tdgl/badge/?version=latest)](https://py-tdgl.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/loganbvh/py-tdgl/branch/main/graph/badge.svg?token=VXdxJKP6Ag)](https://codecov.io/gh/loganbvh/py-tdgl)
![GitHub](https://img.shields.io/github/license/loganbvh/py-tdgl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/535746543.svg)](https://zenodo.org/badge/latestdoi/535746543)



## Motivation
`pyTDGL` solves a 2D generalized time-dependent Ginzburg-Landau (TDGL) equation, enabling simulations of vortex and phase dynamics in thin film superconducting devices.

## Learn `pyTDGL`

The documentation for `pyTDGL` can be found at [py-tdgl.readthedocs.io](https://py-tdgl.readthedocs.io/en/latest/).

## Try `pyTDGL`

Click the badge below to try `pyTDGL` interactively online via [Google Colab](https://colab.research.google.com/):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/loganbvh/py-tdgl/blob/main/docs/notebooks/quickstart.ipynb)

## Install `pyTDGL`

`pyTDGL` requires `python` `3.8`, `3.9`, `3.10`, or `3.11`. We recommend installing `pyTDGL` in a [`conda` environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), e.g.

```bash
conda create --name tdgl python="3.10"
conda activate tdgl
```

### Install via `pip`

From  [PyPI](https://pypi.org/project/tdgl/), the Python Package index:
    
```bash
pip install tdgl
```

From this [GitHub repository](https://github.com/loganbvh/py-tdgl/):

```bash
pip install git+https://github.com/loganbvh/py-tdgl.git
```

Editable installation:

```bash
git clone https://github.com/loganbvh/py-tdgl.git
cd py-tdgl
pip install -e ".[dev,docs]"
```
## About `pyTDGL`

### Authors

- Primary author and maintainer: [@loganbvh](https://github.com/loganbvh/).

### Citing `pyTDGL`

`pyTDGL` is described in the following paper:

>*pyTDGL: Time-dependent Ginzburg-Landau in Python*, Computer Physics Communications **291**, 108799 (2023), DOI: [10.1016/j.cpc.2023.108799](https://doi.org/10.1016/j.cpc.2023.108799).

If you use `pyTDGL` in your research, please cite the paper linked above.

    % BibTeX citation
    @article{
        Bishop-Van_Horn2023-wr,
        title    = "{pyTDGL}: Time-dependent {Ginzburg-Landau} in Python",
        author   = "Bishop-Van Horn, Logan",
        journal  = "Comput. Phys. Commun.",
        volume   =  291,
        pages    = "108799",
        month    =  may,
        year     =  2023,
        url      = "http://dx.doi.org/10.1016/j.cpc.2023.108799",
        issn     = "0010-4655",
        doi      = "10.1016/j.cpc.2023.108799"
    }


### Acknowledgments

Parts of this package have been adapted from [`SuperDetectorPy`](https://github.com/afsa/super-detector-py), a GitHub repo authored by [Mattias Jönsson](https://github.com/afsa). Both `SuperDetectorPy` and `py-tdgl` are released under the open-source MIT License. If you use either package in an academic publication or similar, please consider citing the following in addition to the `pyTDGL` paper:

- Mattias Jönsson, Theory for superconducting few-photon detectors (Doctoral dissertation), KTH Royal Institute of Technology (2022) ([Link](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-312132))
- Mattias Jönsson, Robert Vedin, Samuel Gyger, James A. Sutton, Stephan Steinhauer, Val Zwiller, Mats Wallin, Jack Lidmar, Current crowding in nanoscale superconductors within the Ginzburg-Landau model, Phys. Rev. Applied 17, 064046 (2022) ([Link](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.17.064046))

The user interface is adapted from [`SuperScreen`](https://github.com/loganbvh/superscreen).

