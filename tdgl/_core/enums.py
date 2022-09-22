from enum import Enum
from typing import Sequence

import numpy as np


class Observable(Enum):
    COMPLEX_FIELD = "Complex field"
    PHASE = "Phase"
    SUPERCURRENT = "Supercurrent density"
    NORMAL_CURRENT = "Normal current density"
    VORTICITY = "Vorticity"
    SCALAR_POTENTIAL = "Scalar potential"
    APPLIED_VECTOR_POTENTIAL = "Applied vector potential"
    INDUCED_VECTOR_POTENTIAL = "Induced vector potential"
    TOTAL_VECTOR_POTENTIAL = "Total vector potential"
    ALPHA = "Alpha"

    @classmethod
    def get_keys(cls) -> Sequence[str]:
        return list(item.name for item in Observable)

    @classmethod
    def from_key(cls, key: str) -> "Observable":
        return Observable[key]


class SparseFormat(Enum):
    CSC = "csc"
    CSR = "csr"


class Operator(Enum):
    AND = np.all
    OR = np.any

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.value(*args, **kwargs)


class MatrixType(Enum):
    LAPLACIAN = "laplacian"
    NEUMANN_BOUNDARY_LAPLACIAN = "neumann_boundary_laplacian"
    DIVERGENCE = "divergence"
    GRADIENT = "gradient"
