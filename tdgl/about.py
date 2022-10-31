import inspect
import os
import sys
import time
from typing import Dict, Optional

import IPython
import joblib
import matplotlib
import numpy
import scipy
from IPython.display import HTML

try:
    import jax

    jax_version = jax.__version__
except (ModuleNotFoundError, ImportError):
    jax_version = None

import tdgl


def _blas_info() -> str:
    # https://github.com/qutip/qutip/blob/3ff3f42d2cd950c99af8936b12d9221ed79de5e3/
    # qutip/utilities.py#L335-L355
    config = numpy.__config__
    if hasattr(config, "blas_ilp64_opt_info"):
        blas_info = config.blas_ilp64_opt_info
    elif hasattr(config, "blas_opt_info"):
        blas_info = config.blas_opt_info
    else:
        blas_info = {}
    _has_lib_key = "libraries" in blas_info
    blas = None
    if hasattr(config, "mkl_info") or (
        _has_lib_key and any("mkl" in lib for lib in blas_info["libraries"])
    ):
        blas = "INTEL MKL"
    elif hasattr(config, "openblas_info") or (
        _has_lib_key and any("openblas" in lib for lib in blas_info["libraries"])
    ):
        blas = "OPENBLAS"
    elif "extra_link_args" in blas_info and (
        "-Wl,Accelerate" in blas_info["extra_link_args"]
    ):
        blas = "Accelerate"
    else:
        blas = "Generic"
    return blas


def version_dict() -> Dict[str, str]:
    """Returns a dictionary containing the versions of important dependencies."""
    cpu_count = [joblib.cpu_count(only_physical_cores=b) for b in (True, False)]
    return {
        "tdgl": tdgl.__version__,
        "Numpy": numpy.__version__,
        "SciPy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "jax": str(jax_version),
        "IPython": IPython.__version__,
        "Python": sys.version,
        "OS": f"{os.name} [{sys.platform}]",
        "Number of CPUs": f"Physical: {cpu_count[0]}, Logical: {cpu_count[1]}",
        "BLAS Info": _blas_info(),
    }


def version_table(
    version_info: Optional[Dict[str, str]] = None, verbose: bool = False
) -> HTML:
    """Returns an HTML table with the versions of important depedencies."""

    # Adapted from: https://github.com/qutip/qutip/blob/
    # 88919ce50880dadbc1a817a3e6059c82c23a83f9/qutip/ipynbtools.py#L47

    html = [
        "<table>",
        "<tr><th>Software</th><th>Version</th></tr>",
    ]
    if version_info is None:
        version_info = version_dict()

    for name, version in version_info.items():
        html.append(f"<tr><td>{name}</td><td>{version}</td></tr>")

    if verbose:
        html.append("<tr><th colspan='2'>Additional information</th></tr>")
        install_path = os.path.dirname(inspect.getsourcefile(tdgl))
        html.append(f"<tr><td>Installation path</td><td>{install_path}</td></tr>")

    html.append(
        f"<tr><td colspan='2'>{time.strftime('%a %b %d %H:%M:%S %Y %Z')}</td></tr>"
    )
    html.append("</table>")

    return HTML("".join(html))
