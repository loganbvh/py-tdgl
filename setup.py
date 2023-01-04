"""
# pyTDGL

Time-dependent Ginzburg-Landau in Python

![PyPI](https://img.shields.io/pypi/v/tdgl)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/loganbvh/py-tdgl/lint-and-test.yml?branch=main)
[![Documentation Status](https://readthedocs.org/projects/py-tdgl/badge/?version=latest)](https://py-tdgl.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/loganbvh/py-tdgl/branch/main/graph/badge.svg?token=VXdxJKP6Ag)](https://codecov.io/gh/loganbvh/py-tdgl)
![GitHub](https://img.shields.io/github/license/loganbvh/py-tdgl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Motivation
`pyTDGL` solves a 2D generalized time-dependent Ginzburg-Landau (TDGL) equation, enabling simulations of vortex and phase dynamics in thin film superconducting devices.

## Learn `pyTDGL`

The documentation for `pyTDGL` can be found at [py-tdgl.readthedocs.io](https://py-tdgl.readthedocs.io/en/latest/).

## Try `pyTDGL`

Click the badge below and navigate to `docs/notebooks/` to try `pyTDGL` interactively online via [Binder](https://mybinder.org/)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/loganbvh/py-tdgl/HEAD)

## Acknowledgments

Parts of this package have been adapted from [`SuperDetectorPy`](https://github.com/afsa/super-detector-py), a GitHub repo authored by [Mattias Jönsson](https://github.com/afsa). Both `SuperDetectorPy` and `py-tdgl` are released under the open-source MIT License. If you use either package in an academic publication or similar, please consider citing the following:

- Mattias Jönsson, Theory for superconducting few-photon detectors (Doctoral dissertation), KTH Royal Institute of Technology (2022) ([Link](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-312132))
- Mattias Jönsson, Robert Vedin, Samuel Gyger, James A. Sutton, Stephan Steinhauer, Val Zwiller, Mats Wallin, Jack Lidmar, Current crowding in nanoscale superconductors within the Ginzburg-Landau model, Phys. Rev. Applied 17, 064046 (2022) ([Link](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.17.064046))

The user interface is adapted from [`SuperScreen`](https://github.com/loganbvh/superscreen).
"""

from setuptools import find_packages, setup

DESCRIPTION = "pyTDGL: Time-dependent Ginzburg-Landau in Python."
LONG_DESCRIPTION = __doc__

NAME = "tdgl"
AUTHOR = "Logan Bishop-Van Horn"
AUTHOR_EMAIL = "logan.bvh@gmail.com"
URL = "https://github.com/loganbvh/py-tdgl"
LICENSE = "MIT"
PYTHON_VERSION = ">=3.8, <3.11"

INSTALL_REQUIRES = [
    "cloudpickle",
    "h5py",
    "joblib",
    "jupyter",
    "matplotlib",
    "meshpy",
    "numpy",
    "pint",
    "pytest",
    "pytest-cov",
    "scipy",
    "shapely",
    "tqdm",
]

EXTRAS_REQUIRE = {
    "dev": [
        "black",
        "isort",
        "pre-commit",
    ],
    "docs": [
        "IPython",
        # https://stackoverflow.com/a/71069918
        "sphinx>=4.3.0",
        "sphinx-rtd-theme>=0.5.1",
        "sphinx-autodoc-typehints",
        "nbsphinx",
        "pillow",
        "sphinx_toolbox",
        "enum_tools",
        "sphinx-argparse",
        "sphinxcontrib-bibtex",
    ],
    "jax": [
        "jax[cpu]",
    ],
}

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
]

PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
KEYWORDS = "superconductor vortex Ginzburg-Landau"

exec(open("tdgl/version.py").read())

setup(
    name=NAME,
    version=__version__,  # noqa: F821
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    python_requires=PYTHON_VERSION,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
